//! Routed SwiGLU experts whose projections hold block-scaled E4M3 codes: each live row's selected
//! experts, one selection at a time.
//!
//! - An expert is its checkpoint's `gate_proj`, `up_proj` and `down_proj`, each a 128 x 128
//!   block-scaled E4M3 weight ([`fp8::Block`]); a projection quantizes its input at run time in
//!   groups of 128 inputs, as the block linear of one row states.
//! - A selection's expert is read from the router's indices on the device, so a launch's routing
//!   never leaves it; the selections of rows past the live ones are not computed, and their output
//!   rows hold whatever was there.
//! - The SiLU of the gate rounds to BF16 before it multiplies the up projection, and the product
//!   rounds to BF16 before the down projection quantizes it, as for dense rows.

use crate::Op;
use crate::load::Load;
use crate::ops::linear::experts::{Indices, Rows};
use crate::ops::pointwise::{activation::Silu, multiply::Multiply};
use crate::tensor::form::{BlockScaled, fp8};
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;

/// A block-scaled E4M3 projection of `In` features to `Out` on one chip.
pub type Projection<In, Out> =
    BlockScaled<In, Out, f8e4m3, f32, fp8::Scales<In, Out>, fp8::Block, Chip>;

/// One expert's gated feed-forward of `Hidden` features over `Intermediate` ones.
#[derive(DeviceSend, Load)]
pub struct Expert<Hidden: AxisName, Intermediate: AxisName> {
    pub gate_proj: Projection<Hidden, Intermediate>,
    pub up_proj: Projection<Hidden, Intermediate>,
    pub down_proj: Projection<Intermediate, Hidden>,
}

impl<H: AxisName, I: AxisName> Expert<H, I> {
    /// `down(round(silu(gate(row))) * up(row))`, each projection rounded once to BF16.
    pub fn row(
        &self,
        device: &mut Device,
        row: &HbmTensor<bf16, Chip, m![H]>,
    ) -> HbmTensor<bf16, Chip, m![H]> {
        let gate = self.gate_proj.forward(&mut *device, row);
        let up = self.up_proj.forward(&mut *device, row);
        let gate = Silu.forward(&mut *device, &gate);
        let product = Multiply.forward(&mut *device, (&gate, &up));
        self.down_proj.forward(device, &product)
    }
}

/// `output[r][k] = experts[indices[r][k]](input[r])` for each live row `r` of `R` and each of its
/// `K` selections; an index names one of the `E` experts.
impl<'x, H: AxisName, I: AxisName, R: AxisName, K: AxisName, const E: usize>
    Op<(
        &'x HbmTensor<bf16, Chip, m![R, H]>,
        &'x Indices<R, K>,
        Live<(), R>,
    )> for [Expert<H, I>; E]
{
    type Output = Rows<R, K, H>;

    fn forward(
        &self,
        device: &mut Device,
        (input, indices, live): (
            &'x HbmTensor<bf16, Chip, m![R, H]>,
            &'x Indices<R, K>,
            Live<(), R>,
        ),
    ) -> Self::Output {
        const { assert!(E > 0 && K::SIZE > 0) };
        let mut output = Rows::<R, K, H>::new();
        for r in live.start()..live.start() + live.rows() {
            for k in 0..K::SIZE {
                let mut at: HbmTensor<i32, Chip, m![1]> = HbmTensor::new();
                indices
                    .view()
                    .tile::<m![R], 1, m![R = 1 # R, K, 1 # 2]>(r)
                    .tile::<m![K], 1, m![R = 1 # R, K = 1 # K, 1 # 2]>(k)
                    .to_hbm_view(&mut device.tdma, at.view_mut());
                let expert = at.view().to_spm(&mut device.tdma) as usize;
                // SAFETY: the tile is row `r`'s features in order; only the row axis drops.
                let tile = unsafe {
                    input
                        .view()
                        .tile::<m![R], 1, m![R = 1 # R, H]>(r)
                        .reshape::<Chip, m![H]>()
                };
                let mut row: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
                tile.to_hbm_view(&mut device.tdma, row.view_mut());
                let routed = self[expert].row(&mut *device, &row);
                let into = output
                    .view_mut()
                    .tile::<m![R], 1, m![R = 1 #{!} R, K, H]>(r)
                    .tile::<m![K], 1, m![R = 1 #{!} R, K = 1 #{!} K, H]>(k);
                routed.view().to_hbm_view(&mut device.tdma, into);
            }
        }
        output
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ops::linear::fp8::block::tests::{hash, reference};
    use crate::tensor::form::fp8::{GROUP, layout};
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use core::marker::PhantomData;
    use furiosa_opt_std::runtime::launch;

    axes![Width = 512, Inner = 256, Eight = 8, Two = 2];

    /// Each row's two selections among four experts; rows past the live ones select too.
    const SELECTED: [i32; 16] = [2, 0, 1, 3, 3, 2, 0, 1, 1, 1, 2, 0, 3, 0, 0, 2];

    /// The live rows of the test's eight.
    const LIVE: usize = 6;

    #[device(chip = 1)]
    fn routed(
        device: &mut Device,
        experts: &[Expert<Width, Inner>; 4],
        input: &HbmTensor<bf16, Chip, m![Eight, Width]>,
        indices: &Indices<Eight, Two>,
        count: &Value<Chip>,
    ) -> Rows<Eight, Two, Width> {
        let live = Live::<(), Eight>::read(count);
        experts.forward(device, (input, indices, live))
    }

    /// A projection's E4M3 codes and its `[out / 128, in / 128]` grid of block scales.
    pub(crate) type Weights = (Vec<f8e4m3>, Vec<f32>);

    /// A projection's codes and `[out / 128, in / 128]` grid of scales, from `seed`.
    pub(crate) fn weights(rows: usize, columns: usize, seed: u32) -> Weights {
        let codes = (0..rows * columns)
            .map(|at| {
                let bits = hash(at, seed);
                f8e4m3::from_f32(((bits % 241) as f32 - 120.0) / 16.0)
            })
            .collect();
        let grid = (0..(rows / GROUP) * (columns / GROUP))
            .map(|at| 0.002 + (hash(at, seed ^ 0x55) % 13) as f32 * 0.0007)
            .collect();
        (codes, grid)
    }

    /// `count` experts' gate, up and down weights of `w` features over `i` intermediate ones.
    pub(crate) fn banks(count: u32, w: usize, i: usize) -> Vec<[Weights; 3]> {
        (0..count)
            .map(|e| {
                [
                    weights(i, w, 0x100 + e),
                    weights(i, w, 0x200 + e),
                    weights(w, i, 0x300 + e),
                ]
            })
            .collect()
    }

    fn rounded(values: Vec<f32>) -> Vec<bf16> {
        values.into_iter().map(bf16::from_f32).collect()
    }

    pub(crate) async fn projection<In: AxisName, Out: AxisName>(
        device: &mut Device,
        (codes, grid): &Weights,
    ) -> Projection<In, Out> {
        BlockScaled {
            weight: HostTensor::<f8e4m3, m![Out, In]>::from_vec(codes.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f32, fp8::Scales<In, Out>>::from_vec(layout::<In, Out>(
                grid,
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap(),
            contract: fp8::Block,
            shape: PhantomData,
        }
    }

    /// The FP32 output of `row` through the expert of `gate`, `up` and `down` weights, before its
    /// rounding: the reference block linears, the SiLU and the product each rounded to BF16.
    pub(crate) fn oracle(row: &[bf16], [gate, up, down]: &[Weights; 3]) -> Vec<f32> {
        let gated = rounded(reference(row, &gate.0, &gate.1));
        let up = rounded(reference(row, &up.0, &up.1));
        let product: Vec<bf16> = gated
            .iter()
            .zip(&up)
            .map(|(g, u)| {
                let g = g.to_f32();
                let silu = bf16::from_f32(g / (1.0 + (-g).exp())).to_f32();
                bf16::from_f32(silu * u.to_f32())
            })
            .collect();
        reference(&product, &down.0, &down.1)
    }

    /// The experts of `banks` on the device.
    pub(crate) async fn experts<H: AxisName, I: AxisName, const E: usize>(
        device: &mut Device,
        banks: &[[Weights; 3]],
    ) -> [Expert<H, I>; E] {
        let mut experts = Vec::new();
        for [gate, up, down] in banks {
            experts.push(Expert {
                gate_proj: projection::<H, I>(device, gate).await,
                up_proj: projection::<H, I>(device, up).await,
                down_proj: projection::<I, H>(device, down).await,
            });
        }
        experts.try_into().ok().unwrap()
    }

    /// Each live row's selection runs through its expert as the reference block linears and a
    /// BF16 SwiGLU give it, to a step of the output's rounding; rows past the live ones are not
    /// written.
    #[tokio::test(flavor = "current_thread")]
    async fn runs_each_live_selection_through_its_expert() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(routed.topology()).unwrap();
        let (w, i) = (Width::SIZE, Inner::SIZE);
        let banks = banks(4, w, i);
        let experts = experts::<Width, Inner, 4>(&mut device, &banks).await;
        let rows: Vec<bf16> = (0..Eight::SIZE * w)
            .map(|at| bf16::from_f32(((hash(at, 0x77) % 1001) as f32 - 500.0) / 250.0))
            .collect();
        let input = HostTensor::<bf16, m![Eight, Width]>::from_vec(rows.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let indices = HostTensor::<i32, m![Eight, Two]>::from_vec(SELECTED)
            .to_hbm::<Chip, m![Eight, Two, 1 # 2]>(&mut device.pdma)
            .await
            .unwrap();
        let count = HostTensor::<i32, m![1]>::from_vec(vec![LIVE as i32])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(routed, (&mut device, &experts, &input, &indices, &count))
            .await
            .unwrap()
            .to_host::<m![Eight, Two, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, &expert) in SELECTED.iter().enumerate().take(LIVE * Two::SIZE) {
            let row = &rows[at / Two::SIZE * w..][..w];
            let want = oracle(row, &banks[expert as usize]);
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let got = &output[at * w..][..w];
            for (o, (got, want)) in got.iter().zip(&want).enumerate() {
                // A product a step off its reference's may quantize to a neighboring code.
                let error = (got.to_f32() - want).abs();
                assert!(
                    error <= scale * 2f32.powi(-6),
                    "selection {at} (expert {expert}) output {o}: {got:?} vs {want}"
                );
            }
        }
    }
}
