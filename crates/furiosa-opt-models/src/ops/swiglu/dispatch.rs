//! A routing group's selections ordered by expert on the device: each live selection's slot in
//! its expert's region of a dispatch buffer, and every expert's count of them.
//!
//! - A selection `p = r * K + k` of expert `e` keys as `e * P + p`, so its keys order selections by
//!   expert, then by row: every key is distinct, and a row past the live ones keys as the idle
//!   expert `X`, past every live key.
//! - A target's count is how many keys lie below it, each slice one target streaming every key: the
//!   keys whose distance to it is negative, summed. A selection's rank within its expert is its key's count less
//!   its expert's base `e * P`'s, and an expert's selections are the counts below `(e + 1) * P`
//!   less those below `e * P`.
//! - A selection's slot is row `rank` of its expert's region of `R` rows, which holds every row of
//!   the group that selects the expert; it is given as the byte offset of that row of `W` BF16
//!   features, the form a DMA scatter and gather read.
//! - A chain over one selection a packet commits it in the first word of a pair; a transpose of
//!   two such packets lays two selections side by side, so a row's keys and offsets are written
//!   dense, the form a key stream and a DMA's index list read.

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::ops::linear::experts::Indices;
use crate::tensor::live::Live;
use crate::tensor::mapping::BroadcastClusters as Cluster;
use crate::tensor::parallel::Chip;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

/// A value whose comparison with zero tagged it below.
const BELOW: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

/// One row a slice.
type PerRow<D, R, E> = DmTensor<D, Chip, Cluster, m![R # 256], E>;

/// The targets a pass of the count holds, one a slice.
#[derive(Clone, Debug)]
pub struct Lanes;

impl AxisName for Lanes {
    const NAME: Ident = Ident::new("DispatchLanes");
    const SIZE: usize = 256;
}

/// The passes over `T` targets.
#[derive(Clone, Debug)]
pub struct Passes<T>(PhantomData<T>);

impl<T: AxisName> AxisName for Passes<T> {
    const NAME: Ident = Ident::new("DispatchPasses");
    const SIZE: usize = {
        assert!(T::SIZE.is_multiple_of(Lanes::SIZE));
        T::SIZE / Lanes::SIZE
    };
}

/// The bases `e * P` of every expert and one past the last, padded to a pass.
pub type Thresholds = Lanes;

/// Every selection's value in the first word of its pair.
pub type Pairs<R, K> = HbmTensor<i32, Chip, m![R, K, 1 # 2]>;

/// Every selection's value, a row's in one packet.
pub type Dense<R, K> = HbmTensor<i32, Chip, m![R, K]>;

/// The constants a dispatch of `R` rows of `K` selections among `X` experts reads, built on host.
#[derive(DeviceSend)]
pub struct Dispatch<R: AxisName, K: AxisName, X: AxisName> {
    /// `r * K + k` of every selection, in the first word of its pair.
    pub positions: Pairs<R, K>,
    /// The key of every selection as a row past the live ones: `X * P + r * K + k`.
    pub idle: Dense<R, K>,
    /// `e * P` of every expert `e` to `X`, then past every key, in the first word of a pair.
    pub thresholds: HbmTensor<i32, Chip, m![Thresholds, 1 # 2]>,
    shape: PhantomData<fn() -> X>,
}

impl<R: AxisName, K: AxisName, X: AxisName> Dispatch<R, K, X> {
    /// Selections of the group.
    const P: usize = R::SIZE * K::SIZE;

    /// The host values: the positions, the idle keys and the thresholds.
    fn host() -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        const { assert!(X::SIZE < Thresholds::SIZE && (X::SIZE + 2) * Self::P < 1 << 30) };
        let positions: Vec<i32> = (0..Self::P as i32).flat_map(|p| [p, 0]).collect();
        let idle = (0..Self::P as i32)
            .map(|p| (X::SIZE * Self::P) as i32 + p)
            .collect();
        let thresholds = (0..Thresholds::SIZE)
            .flat_map(|e| [(e.min(X::SIZE + 1) * Self::P) as i32, 0])
            .collect();
        (positions, idle, thresholds)
    }
}

/// No checkpoint stores the constants: a load builds them on host.
impl<'a, R: AxisName, K: AxisName, X: AxisName> Load<&'a Checkpoint> for Dispatch<R, K, X> {
    fn load(context: &mut Context<'_>, _: &'a Checkpoint) -> Result<Self, LoadError> {
        let (positions, idle, thresholds) = Self::host();
        let positions = context.field("positions", "", |cx| {
            HbmTensor::load(cx, HostTensor::<i32, m![R, K, 1 # 2]>::from_vec(positions))
        })?;
        let idle = context.field("idle", "", |cx| {
            HbmTensor::load(cx, HostTensor::<i32, m![R, K]>::from_vec(idle))
        })?;
        let thresholds = context.field("thresholds", "", |cx| {
            HbmTensor::load(
                cx,
                HostTensor::<i32, m![Thresholds, 1 # 2]>::from_vec(thresholds),
            )
        })?;
        Ok(Self {
            positions,
            idle,
            thresholds,
            shape: PhantomData,
        })
    }
}

/// Every selection's slot, and every expert's selections.
pub struct Plan<R: AxisName, K: AxisName> {
    /// The byte offset of every selection's slot, a row's eight an index list of its own.
    pub offsets: Dense<R, K>,
    /// The bands of `B` rows each expert's selections fill, in the first word of a pair: the
    /// trip count of the expert's loop.
    pub bands: HbmTensor<i32, Chip, m![Thresholds, 1 # 2]>,
}

/// The bands of `B` rows the selections between consecutive `counts` fill:
/// `ceil((counts[e + 1] - counts[e]) / B)`, a slice an expert.
fn bands<const B: usize>(
    device: &mut Device,
    counts: &HbmTensor<i32, Chip, m![Thresholds, 1 # 2]>,
) -> HbmTensor<i32, Chip, m![Thresholds, 1 # 2]> {
    const { assert!(B.is_power_of_two()) };
    type T = Thresholds;
    let mut next = HbmTensor::<i32, Chip, m![T, 1 # 2]>::new();
    counts
        .view()
        .tile::<m![T], { T::SIZE - 1 }, m![T = { T::SIZE - 1 } # T, 1 # 2]>(1)
        .to_hbm_view(
            &mut device.tdma,
            next.view_mut()
                .tile::<m![T], { T::SIZE - 1 }, m![T = { T::SIZE - 1 } #{!} T, 1 # 2]>(0),
        );
    let current: DmTensor<i32, Chip, Cluster, m![T], m![1 # 2]> = counts.to_dm(&mut device.tdma);
    let current: VrfTensor<i32, Chip, Cluster, m![T], m![1 # 8]> = device
        .sub
        .begin(current.view())
        .fetch::<m![1], m![1 # 2]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let next: DmTensor<i32, Chip, Cluster, m![T], m![1 # 2]> = next.to_dm(&mut device.tdma);
    let rounded: DmTensor<i32, Chip, Cluster, m![T], m![1 # 2]> = device
        .main
        .begin(next.view())
        .fetch::<m![1], m![1 # 2]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::SubFxp, &current)
        .vector_clip(ClipBinaryOpI32::AddFxp, B as i32 - 1)
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit();
    let bands: DmTensor<i32, Chip, Cluster, m![T], m![1 # 2]> = device
        .main
        .begin(rounded.view())
        .fetch::<m![1], m![1 # 2]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(
            FxpBinaryOp::ArithRightShift,
            const { B.trailing_zeros() as i32 },
        )
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit();
    bands.to_hbm(&mut device.tdma)
}

/// A chain's packets of one selection each, two a row of the transpose: a row's eight dense.
macro_rules! dense {
    ($stream:expr, $K:ty) => {
        $stream
            .vector_final()
            .transpose::<m![$K / 2], m![$K % 2 # 8]>()
            .commit_trim::<m![$K % 2]>()
            .commit()
    };
}

impl<R: AxisName, K: AxisName, X: AxisName> Dispatch<R, K, X> {
    /// The slots of `live`'s selections of `indices` in a buffer of `X` regions of `R` rows of
    /// `row` bytes, and every expert's count of them.
    pub fn plan(
        &self,
        device: &mut Device,
        indices: &Indices<R, K>,
        live: Live<(), R>,
        row: usize,
    ) -> Plan<R, K> {
        type Selections<R, K> = crate::modules::feedforward::moe::Selections<R, K>;
        const { assert!(R::SIZE <= 256 && K::SIZE.is_multiple_of(2)) };
        let p = Self::P as i32;
        let selected: PerRow<i32, R, m![K, 1 # 2]> = indices.to_dm(&mut device.tdma);
        // A selection's value, its own packet's operand.
        let operand = |device: &mut Device, pairs: &PerRow<i32, R, m![K, 1 # 2]>| {
            let lanes: VrfTensor<i32, Chip, Cluster, m![R # 256], m![K, 1 # 8]> = device
                .sub
                .begin(pairs.view())
                .fetch::<m![K], m![1 # 2]>()
                .collect::<m![K], m![1 # 8]>()
                .to_vrf();
            lanes
        };
        let positions: PerRow<i32, R, m![K, 1 # 2]> = self.positions.to_dm(&mut device.tdma);
        let at = operand(&mut *device, &positions);
        // Each selection's expert, a packet a selection, times the selections.
        macro_rules! scaled {
            ($by:expr) => {
                device
                    .main
                    .begin(selected.view())
                    .fetch::<m![K], m![1 # 2]>()
                    .collect::<m![K], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_fxp(FxpBinaryOp::MulInt, $by)
            };
        }
        // The keys: dense for the stream, paired as the targets of their own counts.
        let dense: PerRow<i32, R, m![K]> =
            dense!(scaled!(p).vector_clip(ClipBinaryOpI32::AddFxp, &at), K);
        let own: PerRow<i32, R, m![K, 1 # 2]> = scaled!(p)
            .vector_clip(ClipBinaryOpI32::AddFxp, &at)
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        let base: PerRow<i32, R, m![K, 1 # 2]> = scaled!(p)
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        let mut keys: Dense<R, K> = dense.to_hbm(&mut device.tdma);
        let idle = live.start() + live.rows();
        for past in 0..(R::SIZE - idle).min(R::SIZE) {
            let r = idle + past;
            self.idle
                .view()
                .tile::<m![R], 1, m![R = 1 # R, K]>(r)
                .to_hbm_view(
                    &mut device.tdma,
                    keys.view_mut().tile::<m![R], 1, m![R = 1 #{!} R, K]>(r),
                );
        }
        // SAFETY: the rows' selections in order are the selections.
        let (keys, own, base): (
            HbmTensor<i32, Chip, m![{ Symbol<Selections<R, K>> }]>,
            HbmTensor<i32, Chip, m![{ Symbol<Selections<R, K>> }, 1 # 2]>,
            HbmTensor<i32, Chip, m![{ Symbol<Selections<R, K>> }, 1 # 2]>,
        ) = {
            let own: Pairs<R, K> = own.to_hbm(&mut device.tdma);
            let base: Pairs<R, K> = base.to_hbm(&mut device.tdma);
            unsafe { (keys.reshape(), own.reshape(), base.reshape()) }
        };
        let own = count(device, &own, &keys);
        let base = count(device, &base, &keys);
        let counts = count(device, &self.thresholds, &keys);
        // SAFETY: as above, back to the rows' selections.
        let (own, base): (Pairs<R, K>, Pairs<R, K>) = unsafe { (own.reshape(), base.reshape()) };
        let own: PerRow<i32, R, m![K, 1 # 2]> = own.to_dm(&mut device.tdma);
        let base: PerRow<i32, R, m![K, 1 # 2]> = base.to_dm(&mut device.tdma);
        let base = operand(&mut *device, &base);
        // The rank: the count below the key less the count below its expert's base.
        let rank: PerRow<i32, R, m![K, 1 # 2]> = device
            .main
            .begin(own.view())
            .fetch::<m![K], m![1 # 2]>()
            .collect::<m![K], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &base)
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        let rank = operand(&mut *device, &rank);
        // The slot's row: the expert's region, plus the rank.
        let slot: PerRow<i32, R, m![K, 1 # 2]> = scaled!(R::SIZE as i32)
            .vector_clip(ClipBinaryOpI32::AddFxp, &rank)
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        let offsets: PerRow<i32, R, m![K]> = dense!(
            device
                .main
                .begin(slot.view())
                .fetch::<m![K], m![1 # 2]>()
                .collect::<m![K], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::MulInt, row as i32),
            K
        );
        Plan {
            offsets: offsets.to_hbm(&mut device.tdma),
            bands: bands::<16>(device, &counts),
        }
    }
}

/// How many of `keys` lie below each of `targets`, a pass of a target a slice at a time,
/// each slice streaming every key.
fn count<T: AxisName, P: AxisName>(
    device: &mut Device,
    targets: &HbmTensor<i32, Chip, m![T, 1 # 2]>,
    keys: &HbmTensor<i32, Chip, m![P]>,
) -> HbmTensor<i32, Chip, m![T, 1 # 2]> {
    type Ps<T> = Passes<T>;
    const { assert!(P::SIZE.is_multiple_of(8)) };
    let keys: DmTensor<i32, Chip, Cluster, m![256], m![P]> = keys.to_dm(&mut device.tdma);
    // SAFETY: every slice holds every key; only the slice label changes.
    let keys: DmTensor<i32, Chip, Cluster, m![Lanes], m![P]> = unsafe { keys.reshape() };
    // SAFETY: the targets in order are their passes' targets in order.
    let targets = unsafe {
        targets
            .view()
            .reshape::<Chip, m![{ Symbol<Ps<T>> }, Lanes, 1 # 2]>()
    };
    let mut counts = HbmTensor::<i32, Chip, m![{ Symbol<Ps<T>> }, Lanes, 1 # 2]>::new();
    for pass in 0..<Ps<T> as AxisName>::SIZE {
        let target: DmTensor<i32, Chip, Cluster, m![Lanes], m![1 # 2]> = targets
            .tile::<m![{ Symbol<Ps<T>> }], 1, m![{ Symbol<Ps<T>> } = 1 # ({ Symbol<Ps<T>> }), Lanes, 1 # 2]>(pass)
            .to_dm(&mut device.tdma);
        let target: VrfTensor<i32, Chip, Cluster, m![Lanes], m![1 # 8]> = device
            .sub
            .begin(target.view())
            .fetch::<m![1], m![1 # 2]>()
            .collect::<m![1], m![1 # 8]>()
            .to_vrf();
        // A key's distance to the target, then one for a key below it: the sums are the counts.
        let distance: DmTensor<i32, Chip, Cluster, m![Lanes], m![P]> = device
            .main
            .begin(keys.view())
            .fetch::<m![P / 8], m![P % 8]>()
            .collect::<m![P / 8], m![P % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &target)
            .vector_final()
            .commit_trim::<m![P % 8]>()
            .commit();
        let below: DmTensor<i32, Chip, Cluster, m![Lanes], m![1 # 2]> = device
            .main
            .begin(distance.view())
            .fetch::<m![P / 8], m![P % 8]>()
            .collect::<m![P / 8], m![P % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(0),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_logic(LogicBinaryOpI32::BitAnd, 0)
            .vector_logic(LogicBinaryOpI32::BitOr, Branched::imm(BELOW, 1))
            .vector_narrow_split::<m![P / 4], m![P % 4]>()
            .vector_intra_slice_reduce::<P, m![1], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit();
        below.view().to_hbm_view(
            &mut device.tdma,
            counts.view_mut().tile::<m![{ Symbol<Ps<T>> }], 1, m![
                { Symbol<Ps<T>> } = 1 #{!} ({ Symbol<Ps<T>> }), Lanes, 1 # 2
            ]>(pass),
        );
    }
    // SAFETY: as above, the passes' targets in order are the targets.
    unsafe { counts.reshape() }
}

#[cfg(test)]
impl<R: AxisName, K: AxisName, X: AxisName> Dispatch<R, K, X> {
    /// The constants on `device`, as a load places them.
    pub(crate) async fn uploaded(device: &mut Device) -> Self {
        let (positions, idle, thresholds) = Self::host();
        Self {
            positions: HostTensor::<i32, m![R, K, 1 # 2]>::from_vec(positions)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            idle: HostTensor::<i32, m![R, K]>::from_vec(idle)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            thresholds: HostTensor::<i32, m![Thresholds, 1 # 2]>::from_vec(thresholds)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            shape: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Rows = 256, Picks = 8, Choices = 128];

    /// The bytes of a slot's row.
    const ROW: usize = 4096;

    #[device(chip = 1)]
    fn planned(
        device: &mut Device,
        dispatch: &Dispatch<Rows, Picks, Choices>,
        indices: &Indices<Rows, Picks>,
        count: &Value<Chip>,
    ) -> (
        Dense<Rows, Picks>,
        HbmTensor<i32, Chip, m![Thresholds, 1 # 2]>,
    ) {
        let live = Live::<(), Rows>::read(count);
        let plan = dispatch.plan(device, indices, live, ROW);
        (plan.offsets, plan.bands)
    }

    /// Each live selection's slot is its row's place among its expert's selections, in its
    /// expert's region; each expert's bands are those its live selections fill.
    #[tokio::test(flavor = "current_thread")]
    async fn orders_live_selections_by_expert() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(planned.topology()).unwrap();
        let dispatch = Dispatch::<Rows, Picks, Choices>::uploaded(&mut device).await;
        let (rows, k) = (Rows::SIZE, Picks::SIZE);
        // Each row's eight distinct experts.
        let chosen: Vec<i32> = (0..rows)
            .flat_map(|r| (0..k).map(move |j| ((r * 13 + j * 16 + r / 7) % 128) as i32))
            .collect();
        for row in chosen.chunks(k) {
            let mut unique = row.to_vec();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), k);
        }
        let pairs: Vec<i32> = chosen.iter().flat_map(|&e| [e, 0]).collect();
        let indices = HostTensor::<i32, m![Rows, Picks, 1 # 2]>::from_vec(pairs)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        for live in [40, 256] {
            let count = HostTensor::<i32, m![1]>::from_vec(vec![live as i32])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let (offsets, counts) = launch(planned, (&mut device, &dispatch, &indices, &count))
                .await
                .unwrap();
            let offsets = offsets
                .to_host::<m![Rows, Picks]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let counts = counts
                .to_host::<m![Thresholds, 1 # 2]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let mut seen = vec![0usize; 128];
            for p in 0..live * k {
                let e = chosen[p] as usize;
                let want = ((e * rows + seen[e]) * ROW) as i32;
                seen[e] += 1;
                assert_eq!(
                    offsets[p], want,
                    "{live} live rows: selection {p} of expert {e}"
                );
            }
            for e in 0..128 {
                assert_eq!(
                    counts[2 * e],
                    seen[e].div_ceil(16) as i32,
                    "{live} live rows: expert {e}'s bands"
                );
            }
        }
    }
}
