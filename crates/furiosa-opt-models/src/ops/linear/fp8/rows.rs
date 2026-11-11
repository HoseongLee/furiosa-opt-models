//! The FP8 block linear of a band of rows: E4M3 codes whose activations quantize at run time in
//! groups of 128 inputs, each block's sum scaled before the blocks add, as [`fp8`] states, the
//! numbers of the one-row linear ([`super::block`]) for every row.
//!
//! - Each slice holds every row's half of the inputs it contracts, quantized once, and streams
//!   them through the Contraction Engine against one four-output tile of its weight at a time,
//!   resident in the TRF a lane an output: a tile's codes are read once for all the band's rows.
//! - A row's group scale and a block's weight scale reach the sums as VRF operands coarser than
//!   the stream, so the four output lanes of a row share them.
//! - The two halves of a row's inputs add across the slices, and each output rounds once.

use super::super::projection::{Block, Cluster, Reduced, Slices};
use super::block::Activations;
use crate::Op;
use crate::specialize::VRF;
use crate::tensor::form::BlockScaled;
use crate::tensor::form::fp8::{self, Blocks, GROUP, Group, Scales};
use core::borrow::Borrow;
use furiosa_opt_std::prelude::*;

/// A tile's four output rows, one a TRF lane.
type Elements<O> = m![O % ({ Block<O> }) = 4];
/// The four-row tiles of a slice's output rows.
type Tiles<O> = m![O % ({ Block<O> }) / 4];
/// The 128-input blocks of the input half a slice holds.
type Half<I> = m![I % (I / 2) / 128];
/// A half's blocks, padded as the weight scales store them.
type Padded<I> = m![{ Half<I> } # ({ Symbol<Blocks<I>> })];

/// Every row's half of the inputs, by group, in a slice.
type Grouped<R, I, O, D, Chips> =
    DmTensor<D, Chips, Cluster, Slices<I, O>, m![R, { Half<I> }, Group]>;

/// The quantization of every row's input half.
type Quantization<R, I> = Activations<
    m![R, { Half<I> }],
    m![R, { Half<I> }, Group / 16],
    m![R, { Half<I> }, Group / 8],
    m![R, { Half<I> }, Group / 4],
>;

/// A band of rows a linear projects at once, each as the one-row linear projects it.
pub(crate) struct Band<'a, T>(pub &'a T);

/// `output[r] = linear(input[r])` for each of the `R` rows, as the one-row linear gives it.
impl<I: AxisName, O: AxisName, R: AxisName, Chips: M, K, V>
    Op<Band<'_, HbmTensor<bf16, Chips, m![R, I]>>>
    for BlockScaled<I, O, f8e4m3, f32, Scales<I, O>, fp8::Block, Chips, K, V>
where
    K: Borrow<HbmTensor<f8e4m3, Chips, m![O, I]>>,
    V: Borrow<HbmTensor<f32, Chips, Scales<I, O>>>,
{
    type Output = HbmTensor<bf16, Chips, m![R, O]>;

    fn forward(
        &self,
        device: &mut Device,
        Band(input): Band<'_, HbmTensor<bf16, Chips, m![R, I]>>,
    ) -> Self::Output {
        const {
            assert!(
                R::SIZE > 1
                    && I::SIZE.is_multiple_of(2 * GROUP)
                    && O::SIZE.is_multiple_of(256)
                    // Every row's group scales and a tile's block scales share a slice's VRF.
                    && (R::SIZE + 1) * (I::SIZE / (2 * GROUP)) * 32 <= VRF
            );
        }
        let rows: DmTensor<bf16, Chips, Cluster, m![64, 1 # 2, I / (I / 2)], m![R, I % (I / 2)]> =
            input.view().to_dm(&mut device.tdma);
        // SAFETY: a slice's half of each row is its groups in order.
        let rows: Grouped<R, I, O, bf16, Chips> = unsafe { rows.reshape() };
        let scales: DmTensor<f32, Chips, Cluster, Slices<I, O>, m![R, { Half<I> }, 1 # 8]> =
            Quantization::<R, I>::scales(&mut device.main, rows.view())
                .commit_trim::<m![1 # 8]>()
                .commit();
        let activation: VrfTensor<f32, Chips, Cluster, Slices<I, O>, m![R, { Half<I> }, 1 # 8]> =
            device
                .sub
                .begin(scales.view())
                .fetch::<m![R, { Half<I> }], m![1 # 8]>()
                .collect::<m![R, { Half<I> }], m![1 # 8]>()
                .to_vrf();
        let quantized: Grouped<R, I, O, f8e4m3, Chips> =
            Quantization::<R, I>::quantized(&mut device.main, rows.view(), &activation).commit();
        // SAFETY: as above, a row's groups in order are its half of the inputs.
        let quantized: DmTensor<f8e4m3, Chips, Cluster, Slices<I, O>, m![R, I % (I / 2)]> =
            unsafe { quantized.reshape() };
        let weight_scale = unsafe {
            self.scales().view().reshape::<Chips, m![
                O / ({ Block<O> }), { Tiles<O> }, I / (I / 2), { Padded<I> }
            ]>()
        };
        let weight_scale: DmTensor<
            f32,
            Chips,
            Cluster,
            Slices<I, O>,
            m![{ Tiles<O> }, { Padded<I> }],
        > = weight_scale.to_dm(&mut device.tdma);
        let mut halves: DmTensor<f32, Chips, Cluster, Slices<I, O>, m![R, O % ({ Block<O> })]> =
            DmTensor::new();
        for tile in 0..(O::SIZE / 256) {
            let weight: DmTensor<f8e4m3, Chips, Cluster, Slices<I, O>, m![{ Elements<O> }, I % (I / 2)]> =
                self.codes()
                    .view()
                    .tile::<m![O % ({ Block<O> })], 4, m![O / ({ Block<O> }), { Elements<O> } # ({ Block<O> }), I]>(
                        4 * tile,
                    )
                    .to_dm(&mut device.tdma);
            let register: TrfTensor<
                f8e4m3,
                Chips,
                Cluster,
                Slices<I, O>,
                m![{ Elements<O> }],
                m![I % (I / 2)],
            > = device
                .sub
                .begin(weight.view())
                .fetch::<m![{ Elements<O> }, I % (I / 2) / 32], m![I % 32]>()
                .collect::<m![{ Elements<O> }, I % (I / 2) / 32], m![I % 32]>()
                .to_trf();
            let scale: VrfTensor<f32, Chips, Cluster, Slices<I, O>, m![{ Half<I> }, 1 # 8]> =
                device
                    .sub
                    .begin(weight_scale.view().tile::<Tiles<O>, 1, m![
                        { Tiles<O> } = 1 # ({ Tiles<O> }), { Padded<I> }
                    ]>(tile))
                    .fetch::<Half<I>, m![1 # 8]>()
                    .collect::<Half<I>, m![1 # 8]>()
                    .to_vrf();
            device
                .main
                .begin(quantized.view())
                .fetch::<m![R, I % (I / 2) / 32], m![I % 32]>()
                .collect::<m![R, I % (I / 2) / 32], m![I % 32]>()
                .contract_outer::<m![R, { Half<I> }, I % 128 / 64], m![I % 64], _, _, _>(&register)
                .contract_packet::<m![1]>()
                .contract_time::<m![R, { Half<I> }]>()
                .contract_lane::<m![R, { Half<I> }], m![{ Elements<O> } # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![{ Elements<O> }]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &activation)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
                .vector_intra_slice_reduce::<I, m![R], m![{ Elements<O> }]>(
                    IntraSliceReduceOpF32::Add,
                )
                .vector_widen_pad::<m![{ Elements<O> } # 8]>()
                .vector_final()
                .commit_trim::<m![{ Elements<O> }]>()
                .commit_view(halves.view_mut().tile::<m![O % ({ Block<O> })], 4, m![
                    R, { Elements<O> } #{!} ({ Block<O> })
                ]>(4 * tile));
        }
        let output: DmTensor<bf16, Chips, Cluster, Reduced<O>, m![R, O % ({ Block<O> })]> = device
            .main
            .begin(halves.view())
            .fetch::<m![R, O % ({ Block<O> }) / 4], m![O % 4]>()
            .collect::<m![R, O % ({ Block<O> }) / 4], m![O % 4 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<O>, m![R, O % ({ Block<O> }) / 4]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .cast::<bf16, m![O % 4 # 16]>()
            .commit_trim::<m![O % 4]>()
            .commit();
        let mut rows = HbmTensor::<bf16, Chips, m![R, O]>::new();
        output.view().to_hbm_view(&mut device.tdma, rows.view_mut());
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::super::Vector;
    use super::super::block::tests::{hash, integers, powers, reference};
    use super::*;
    use crate::tensor::form::fp8::layout;
    use crate::tensor::parallel::Chip;
    use crate::tests::DEVICE;
    use core::marker::PhantomData;
    use furiosa_opt_std::runtime::launch;

    axes![Wide = 2048, Narrow = 768, Packed = 1536, Rows16 = 16];

    type Linear<I, O> = BlockScaled<I, O, f8e4m3, f32, Scales<I, O>, fp8::Block, Chip>;

    #[device(chip = 1)]
    fn banded<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &Linear<I, O>,
        input: &HbmTensor<bf16, Chip, m![Rows16, I]>,
    ) -> HbmTensor<bf16, Chip, m![Rows16, O]> {
        linear.forward(device, super::Band(input))
    }

    #[device(chip = 1)]
    fn single<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &Linear<I, O>,
        input: &Vector<I>,
    ) -> Vector<O> {
        linear.forward(device, input)
    }

    async fn linear<I: AxisName, O: AxisName>(
        device: &mut Device,
        weight: &[f8e4m3],
        grid: &[f32],
    ) -> Linear<I, O> {
        BlockScaled {
            weight: HostTensor::<f8e4m3, m![O, I]>::from_vec(weight.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f32, Scales<I, O>>::from_vec(layout::<I, O>(grid))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            contract: fp8::Block,
            shape: PhantomData,
        }
    }

    /// The band's rows through `linear` at once, and each through the one-row linear.
    async fn both<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &Linear<I, O>,
        rows: &[bf16],
    ) -> (Vec<bf16>, Vec<Vec<bf16>>) {
        let input = HostTensor::<bf16, m![Rows16, I]>::from_vec(rows.to_vec())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let together = launch(banded, (&mut *device, linear, &input))
            .await
            .unwrap()
            .to_host::<m![Rows16, O]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let mut singles = Vec::new();
        for row in rows.chunks_exact(I::SIZE) {
            let input = HostTensor::<bf16, m![I]>::from_vec(row.to_vec())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(single, (&mut *device, linear, &input))
                .await
                .unwrap()
                .to_host::<m![O]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            singles.push(output);
        }
        (together, singles)
    }

    /// Rows of every magnitude: each group's values scaled by its own power of ten.
    fn rows(inputs: usize, seed: u32) -> Vec<bf16> {
        (0..Rows16::SIZE * inputs)
            .map(|at| {
                let group = at / GROUP;
                let value = ((hash(at, seed) % 2001) as f32 - 1000.0) / 500.0;
                bf16::from_f32(value * 10f32.powi((group % 5) as i32 - 2))
            })
            .collect()
    }

    /// Every row of a band gives the one-row linear's outputs bit for bit, at the widths of
    /// Qwen3-30B-A3B's experts: the packed gate and up projection and the down projection.
    #[tokio::test(flavor = "current_thread")]
    async fn each_row_matches_the_one_row_linear() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(banded.topology()).unwrap();
        let codes = |outputs: usize, inputs: usize, seed: u32| -> Vec<f8e4m3> {
            (0..outputs * inputs)
                .map(|at| f8e4m3::from_f32(((hash(at, seed) % 241) as f32 - 120.0) / 16.0))
                .collect()
        };
        let grid = |outputs: usize, inputs: usize, seed: u32| -> Vec<f32> {
            (0..(outputs / GROUP) * (inputs / GROUP))
                .map(|at| 0.002 + (hash(at, seed) % 13) as f32 * 0.0007)
                .collect()
        };
        let up = linear::<Wide, Packed>(
            &mut device,
            &codes(Packed::SIZE, Wide::SIZE, 3),
            &grid(Packed::SIZE, Wide::SIZE, 4),
        )
        .await;
        let (together, singles) = both(&mut device, &up, &rows(Wide::SIZE, 5)).await;
        for (row, alone) in singles.iter().enumerate() {
            let got = &together[row * Packed::SIZE..][..Packed::SIZE];
            let bits = |v: &[bf16]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(got), bits(alone), "gate and up, row {row}");
        }
        let down = linear::<Narrow, Wide>(
            &mut device,
            &codes(Wide::SIZE, Narrow::SIZE, 6),
            &grid(Wide::SIZE, Narrow::SIZE, 7),
        )
        .await;
        let (together, singles) = both(&mut device, &down, &rows(Narrow::SIZE, 8)).await;
        for (row, alone) in singles.iter().enumerate() {
            let got = &together[row * Wide::SIZE..][..Wide::SIZE];
            let bits = |v: &[bf16]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(got), bits(alone), "down, row {row}");
        }
    }

    /// Each block's integer sum takes its row's group scale and its weight scale before the blocks
    /// add up: every row of a band gives the reference exactly.
    #[tokio::test(flavor = "current_thread")]
    async fn scales_each_row_and_block_before_the_sum() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(banded.topology()).unwrap();
        let (weight, grid) = integers(Wide::SIZE, Narrow::SIZE, 1);
        let down = linear::<Narrow, Wide>(&mut device, &weight, &grid).await;
        // Each row its own powers of two: rows rotate the groups' exponents.
        let base = powers(Narrow::SIZE);
        let rows: Vec<bf16> = (0..Rows16::SIZE)
            .flat_map(|row| {
                let scale = 2f32.powi(row as i32 % 5 - 2);
                base.iter().map(move |v| bf16::from_f32(v.to_f32() * scale))
            })
            .collect();
        let (together, _) = both(&mut device, &down, &rows).await;
        for row in 0..Rows16::SIZE {
            let expected = reference(&rows[row * Narrow::SIZE..][..Narrow::SIZE], &weight, &grid);
            let got = &together[row * Wide::SIZE..][..Wide::SIZE];
            for (o, (got, expected)) in got.iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.to_f32(),
                    bf16::from_f32(*expected).to_f32(),
                    "row {row} output {o}"
                );
            }
        }
    }
}
