//! Routed SwiGLU experts whose projections hold block-scaled E4M3 codes: each live row's selected
//! experts, one selection at a time.
//!
//! - An expert is its checkpoint's `gate_proj`, `up_proj` and `down_proj`, each a 128 x 128
//!   block-scaled E4M3 weight ([`fp8::Block`]); a projection quantizes its input at run time in
//!   groups of 128 inputs, as the block linear of one row states.
//! - A [`Bank`] stacks every expert's codes and scales of one projection in one tensor each, so a
//!   selection's expert is an offset into them, read from the router's indices on the device: a
//!   launch's routing never leaves it, and no buffer is chosen at run time.
//! - The selections of rows past the live ones are not computed, and their output rows hold
//!   whatever was there.
//! - An expert's gate and up projections are packed into one, gate first, as a dense gated MLP's
//!   are; the SiLU of the gate rounds to BF16 before it multiplies the up projection, and the
//!   product rounds to BF16 before the down projection quantizes it, as for dense rows.

use crate::Op;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::linear::experts::{Indices, Rows};
use crate::ops::pointwise::{activation::Silu, multiply::Multiply};
use crate::ops::swiglu::GateUp;
use crate::tensor::axis::Single;
use crate::tensor::form::{BlockScaled, Part, fp8};
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

// The gate, then the up projection, in a packed row.
axes![Term = 2];

/// A block-scaled E4M3 projection of `In` features to `Out` on one chip.
pub type Projection<In, Out> =
    BlockScaled<In, Out, f8e4m3, f32, fp8::Scales<In, Out>, fp8::Block, Chip>;

/// One projection of each of `X` experts: their codes in `[expert, Out, In]` order and their block
/// scales, each expert's laid out as [`fp8::Scales`].
#[derive(DeviceSend)]
pub struct Bank<In: AxisName, Out: AxisName, X: AxisName> {
    pub weight: HbmTensor<f8e4m3, Chip, m![X, Out, In]>,
    pub weight_scale: HbmTensor<f32, Chip, m![X, { fp8::Scales<In, Out> }]>,
}

impl<In: AxisName, Out: AxisName, X: AxisName> Bank<In, Out, X> {
    /// Expert `expert`'s projection, a view of its codes and scales.
    pub fn expert(&self, expert: usize) -> Projection<In, Out> {
        // SAFETY: the weights are read-only; the tile is one expert's codes, then its scales, in
        // order.
        let (weight, weight_scale) = unsafe {
            (
                self.weight
                    .view()
                    .tile::<m![X], 1, m![X = 1 # X, Out, In]>(expert)
                    .alias::<Chip, m![Out, In]>(),
                self.weight_scale
                    .view()
                    .tile::<m![X], 1, m![X = 1 # X, { fp8::Scales<In, Out> }]>(expert)
                    .alias::<Chip, fp8::Scales<In, Out>>(),
            )
        };
        BlockScaled::new(weight, weight_scale, fp8::Block)
    }
}

/// The projections `names` of every expert under the context's name, `0` to `X - 1`, packed in
/// order along their outputs, as a gated MLP packs its gate and up projections: each its codes
/// under `weight` and its scales under either spelling, stacked in one tensor each.
fn bank<In: AxisName, Out: AxisName, X: AxisName, const N: usize>(
    context: &mut Context<'_>,
    from: &Checkpoint,
    names: [&str; N],
) -> Result<Bank<In, Out, X>, LoadError> {
    type P<Out, const N: usize> = Part<Out, N>;
    let mut children = Children::default();
    let (mut codes, mut scales) = (Vec::new(), Vec::new());
    for expert in 0..X::SIZE {
        let at = expert.to_string();
        for name in names {
            let read = context.field(&at, &at, |cx| {
                cx.field(name, name, |cx| {
                    let weight = cx.field("weight", "weight", |cx| {
                        from.host::<f8e4m3, m![{ Symbol<P<Out, N>> }, In], m![{ Symbol<P<Out, N>> }, In]>(cx)
                    });
                    let scale = cx.field("weight_scale", "", |cx| {
                        fp8::scales::<In, P<Out, N>, Chip>(from, cx)
                    });
                    Ok::<_, LoadError>((weight?, scale?))
                })
            });
            // The scales are laid out output-major, so the packed weight's are each part's in turn.
            if let Some((weight, scale)) = children.ask(read)? {
                codes.extend(weight.into_vec());
                scales.extend(scale.into_vec());
            }
        }
    }
    children.node(Some(()))?;
    let tree = names.join("_");
    let weight = context.field(&tree, "", |cx| {
        HbmTensor::load(cx, HostTensor::<f8e4m3, m![X, Out, In]>::from_vec(codes))
    })?;
    let scales = HostTensor::<f32, m![X, { fp8::Scales<In, Out> }]>::from_vec(scales);
    let weight_scale = context.field(&tree, "", |cx| HbmTensor::load(cx, scales))?;
    Ok(Bank {
        weight,
        weight_scale,
    })
}

/// Every expert's gated feed-forward of `Hidden` features over `Intermediate` ones, for `X`
/// experts: their checkpoint's `{expert}.{gate,up,down}_proj`.
#[derive(DeviceSend)]
pub struct Experts<Hidden: AxisName, Intermediate: AxisName, X: AxisName> {
    /// The gate and up projections packed, gate first.
    pub gate_up: Bank<Hidden, GateUp<Intermediate>, X>,
    pub down_proj: Bank<Intermediate, Hidden, X>,
}

impl<'a, H: AxisName, I: AxisName, X: AxisName> Load<&'a Checkpoint> for Experts<H, I, X> {
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let gate_up = children.ask(bank(cx, from, ["gate_proj", "up_proj"]))?;
            let down = children.ask(bank(cx, from, ["down_proj"]))?;
            let (gate_up, down_proj) = children.node(gate_up.zip(down))?;
            Ok(Self { gate_up, down_proj })
        })
    }
}

impl<H: AxisName, I: AxisName, X: AxisName> Experts<H, I, X> {
    /// `down(round(silu(gate(row))) * up(row))` through expert `expert`, each projection rounded
    /// once to BF16.
    pub fn row(
        &self,
        device: &mut Device,
        expert: usize,
        row: &HbmTensor<bf16, Chip, m![H]>,
    ) -> HbmTensor<bf16, Chip, m![H]> {
        let gate_up = self.gate_up.expert(expert).forward(&mut *device, row);
        // SAFETY: the packed features are the gate's, then the up projection's, in order.
        let gate_up: HbmTensor<bf16, Chip, m![Term, I]> = unsafe { gate_up.reshape() };
        // SAFETY: each tile is one half's features in order, read as their own vector.
        let half = |term: usize| unsafe {
            gate_up
                .view()
                .tile::<m![Term], 1, m![Term = 1 # Term, I]>(term)
                .alias::<Chip, m![I]>()
        };
        let (gate, up) = (half(0), half(1));
        let gate = Silu.forward(&mut *device, &gate);
        let product = Multiply.forward(&mut *device, (&gate, &up));
        self.down_proj.expert(expert).forward(device, &product)
    }
}

/// `output[r][k] = experts[indices[r][k]](input[r])` for each live row `r` of `R` and each of its
/// `K` selections; an index names one of the `X` experts.
impl<'x, H: AxisName, I: AxisName, X: AxisName, R: AxisName, K: AxisName>
    Op<(
        &'x HbmTensor<bf16, Chip, m![R, H]>,
        &'x Indices<R, K>,
        Live<(), R>,
    )> for Experts<H, I, X>
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
        const { assert!(X::SIZE > 0 && K::SIZE > 0) };
        let mut output = Rows::<R, K, H>::new();
        for r in live.start()..live.start() + live.rows() {
            let mut row: HbmTensor<bf16, Chip, m![Single, H]> = HbmTensor::new();
            input
                .view()
                .tile::<m![R], 1, m![R = 1 # R, H]>(r)
                .to_hbm_view(&mut device.tdma, row.view_mut());
            // SAFETY: one row's features in order; only the row axis of one drops.
            let row: HbmTensor<bf16, Chip, m![H]> = unsafe { row.reshape() };
            for k in 0..K::SIZE {
                let mut at: HbmTensor<i32, Chip, m![1]> = HbmTensor::new();
                indices
                    .view()
                    .tile::<m![R], 1, m![R = 1 # R, K, 1 # 2]>(r)
                    .tile::<m![K], 1, m![R = 1 # R, K = 1 # K, 1 # 2]>(k)
                    .to_hbm_view(&mut device.tdma, at.view_mut());
                let expert = at.view().to_spm(&mut device.tdma) as usize;
                let routed = self.row(&mut *device, expert, &row);
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
    use furiosa_opt_std::runtime::launch;

    axes![Width = 512, Inner = 256, Eight = 8, Two = 2, Four = 4];

    /// Each row's two selections among four experts; rows past the live ones select too.
    const SELECTED: [i32; 16] = [2, 0, 1, 3, 3, 2, 0, 1, 1, 1, 2, 0, 3, 0, 0, 2];

    /// The live rows of the test's eight.
    const LIVE: usize = 6;

    #[device(chip = 1)]
    fn routed(
        device: &mut Device,
        experts: &Experts<Width, Inner, Four>,
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

    /// The projections `parts` of every expert of `banks`, packed along their outputs in order
    /// and stacked on the device.
    async fn stacked<In: AxisName, Out: AxisName, X: AxisName>(
        device: &mut Device,
        banks: &[[Weights; 3]],
        parts: &[usize],
    ) -> Bank<In, Out, X> {
        let (mut codes, mut scales) = (Vec::<f8e4m3>::new(), Vec::<f32>::new());
        for bank in banks {
            for &at in parts {
                codes.extend(bank[at].0.iter().copied());
                let rows = Out::SIZE / parts.len();
                scales.extend(match parts.len() {
                    1 => layout::<In, Out>(&bank[at].1),
                    _ => part_layout::<In>(&bank[at].1, rows),
                });
            }
        }
        Bank {
            weight: HostTensor::<f8e4m3, m![X, Out, In]>::from_vec(codes)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f32, m![X, { fp8::Scales<In, Out> }]>::from_vec(scales)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        }
    }

    /// The scale layout of one projection of `rows` outputs, as [`layout`] gives it.
    fn part_layout<In: AxisName>(grid: &[f32], rows: usize) -> Vec<f32> {
        #[derive(Clone, Debug)]
        struct Rows768;
        impl AxisName for Rows768 {
            const NAME: Ident = Ident::new("Rows768");
            const SIZE: usize = 768;
        }
        #[derive(Clone, Debug)]
        struct Rows256;
        impl AxisName for Rows256 {
            const NAME: Ident = Ident::new("Rows256");
            const SIZE: usize = 256;
        }
        match rows {
            768 => layout::<In, Rows768>(grid),
            256 => layout::<In, Rows256>(grid),
            _ => unreachable!("the tests' projections have 256 or 768 outputs"),
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
    pub(crate) async fn experts<H: AxisName, I: AxisName, X: AxisName>(
        device: &mut Device,
        banks: &[[Weights; 3]],
    ) -> Experts<H, I, X> {
        assert_eq!(banks.len(), X::SIZE);
        Experts {
            gate_up: stacked::<H, GateUp<I>, X>(device, banks, &[0, 1]).await,
            down_proj: stacked::<I, H, X>(device, banks, &[2]).await,
        }
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
        let experts = experts::<Width, Inner, Four>(&mut device, &banks).await;
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
