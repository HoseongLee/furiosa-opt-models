//! The cut of Qwen3 MoE's layer at its attention: the BF16 attention of Qwen3's layers
//! ([`affine`](super::affine)), then the sparse block on each tile's rows.
//!
//! - A tile's normalized rows leave data memory for HBM as rows of their features, where the
//!   sparse block routes them in groups of 256, the most a router's selection takes at once, and
//!   runs each live row's experts; the block's rows come back to data memory for the residual.
//! - The sparse block lands nothing ahead: its experts' weights are read where they are stored, the
//!   selected experts' only.

use super::affine::{self, Cut};
use super::{Launched, Output, Places, RowTile, Rows, Stored, Tokenwise};
use crate::Module;
use crate::models::decoder::attention::qwen3::launch;
use crate::models::decoder::attention::qwen3_moe::fp8::{Block, ForCausalLM, Hidden, Layer, Size};
use crate::modules::attention::Attention;
use crate::modules::generic::schedule::{Begin, Fill, Spill};
use crate::modules::language::{Backbone, Call};
use crate::modules::residual::Residual;
use crate::ops::attention::full::paged::Retained;
use crate::ops::embedding;
use crate::ops::normalization::span::{Owned, Span, Stage};
use crate::specialize::attention::Decode;
use crate::tensor::axis;
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use crate::{Sequential, Then};
use core::task::Poll;
use furiosa_opt_std::prelude::*;

type Capacity = launch::Capacity<Size>;
type Tile = Live<(), Capacity>;

/// The rows of one routing group, and a span's groups.
type Group = axis::Rows<256>;
axes![Groups = 2];

const _: () = assert!(Group::SIZE * Groups::SIZE == Span::SIZE);

/// A tile's normalized rows `rows` through the sparse block: each group of 256 rows with a live row
/// routed in HBM, the block's rows of the whole span returned to data memory.
fn sparse(
    device: &mut Device,
    block: &Block,
    rows: Live<&Owned<Hidden, Span>, Capacity>,
) -> Owned<Hidden, Span> {
    let mut features: HbmTensor<bf16, Chip, m![Span, Hidden]> = HbmTensor::new();
    rows.value()
        .view()
        .to_hbm_view(&mut device.tdma, features.view_mut());
    // SAFETY: a span's rows in order are its groups' rows in order.
    let features: HbmTensor<bf16, Chip, m![Groups, Group, Hidden]> = unsafe { features.reshape() };
    let mut routed: HbmTensor<bf16, Chip, m![Groups, Group, Hidden]> = HbmTensor::new();
    for group in 0..rows.count::<Group>() {
        let mut input: HbmTensor<bf16, Chip, m![Group, Hidden]> = HbmTensor::new();
        features
            .view()
            .tile::<m![Groups], 1, m![Groups = 1 # Groups, Group, Hidden]>(group)
            .to_hbm_view(&mut device.tdma, input.view_mut());
        let output = block.rows(&mut *device, &input, rows.group::<Group>(group));
        let into = routed
            .view_mut()
            .tile::<m![Groups], 1, m![Groups = 1 #{!} Groups, Group, Hidden]>(group);
        output.view().to_hbm_view(&mut device.tdma, into);
    }
    // SAFETY: the groups' rows in order are the span's.
    let routed: HbmTensor<bf16, Chip, m![Span, Hidden]> = unsafe { routed.reshape() };
    routed.view().to_dm(&mut device.tdma)
}

/// A tile through `layer` from its attention, given a tile `output` of its attention output and
/// `input` of its input rows: the output projection, the attention residual, the sparse block on
/// its norm and the block's residual; the next layer's input rows.
fn end(
    device: &mut Device,
    layer: &Layer,
    (attended, landed): Output<'_, Size, ()>,
    input: RowTile<Size>,
) -> RowTile<Size> {
    let Sequential(Residual(_, attention, sum), Residual(norm, block, add)) = layer;
    let Attention { output, .. } = &attention.0;
    let tile = input.with(());
    let gain = crate::Op::forward(&Stage, &mut *device, &norm.0);
    let projected = match output.forward(
        &mut *device,
        (),
        &mut (),
        (&mut *landed, &attended.value().0, tile),
    ) {
        Poll::Ready(rows) => tile.with(rows),
        Poll::Pending => tile.with(Owned::new()),
    };
    let attended = sum.forward(&mut *device, (), &mut (), (&input, &projected));
    let normalized = gain.forward(&mut *device, (), &mut (), &attended);
    let fed = tile.with(sparse(
        &mut *device,
        &block.0,
        tile.with(normalized.value()),
    ));
    add.forward(device, (), &mut (), (&attended, &fed))
}

/// A layer viewed as the BF16 cut: its attention residual, then its sparse block's.
fn cut(layer: &Layer) -> &Cut<Size, <Layer as Second>::Part> {
    layer
}

/// The second part of a sequence.
trait Second {
    type Part;
}

impl<A, B> Second for Sequential<A, B> {
    type Part = B;
}

/// A tile of the embedded rows through the first layer up to its attention's store.
impl<'l, 'c, 'a, 's> Module<Tile, Launched<'c, 'a, Size>, &'s mut Stored<Size>>
    for Tokenwise<&'l Rows<Size>, &'l Layer>
{
    type Output = RowTile<Size>;

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a, Size>,
        stored: &'s mut Stored<Size>,
        tile: Tile,
    ) -> Self::Output {
        let Self(embedded, layer) = *self;
        let front = affine::front(device, cut(layer), tile);
        let rows = tile.with(embedded.fill(&mut *device, tile));
        affine::start(device, call, cut(layer), front, rows, stored)
    }
}

/// A tile through a layer from its attention and the next layer up to its attention's store.
impl<'l, 'c, 'a, 'p, 's>
    Module<(Output<'p, Size, ()>, RowTile<Size>), Launched<'c, 'a, Size>, &'s mut Stored<Size>>
    for Tokenwise<&'l Layer, &'l Layer>
{
    type Output = RowTile<Size>;

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a, Size>,
        stored: &'s mut Stored<Size>,
        (output, input): (Output<'p, Size, ()>, RowTile<Size>),
    ) -> Self::Output {
        let Self(layer, next) = *self;
        let tile = input.with(());
        let rows = end(device, layer, output, input);
        let front = affine::front(device, cut(next), tile);
        affine::start(device, call, cut(next), front, rows, stored)
    }
}

/// A launch's rows through the model's leaves in the decoder family's order: embedded, the first
/// layer's stores over every tile, then each layer attended and, tile by tile, its work up to the
/// next layer's store or the final norm, from whose rows the selected rows' tokens are read.
impl<'i, 'a, 's, 'b, Env> Module<&'i mut launch::Input<'a, Size>, Env, launch::State<'s, 'b, Size>>
    for ForCausalLM
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        state: launch::State<'s, 'b, Size>,
        input: &'i mut launch::Input<'a, Size>,
    ) {
        const LAYERS: usize = <Size as crate::models::decoder::attention::Geometry>::LAYERS;
        let causal = &self.model;
        let live = Live::<(), Capacity>::read(&input.count);
        let context =
            causal
                .model
                .rotary_emb
                .forward(&mut *device, (), &mut (), (live, &mut input.cache));
        let call = Call {
            context: &context,
            tokens: &input.tokens,
            live,
        };
        let (((embedded, slots), ()), tokens) = state;
        let Then(Backbone { embedding, body }, norm) = &causal.model.body;
        let launch = embedding::Launch {
            ids: call.tokens,
            count: live,
            output: &mut *embedded,
        };
        crate::Op::forward(&embedding.0, &mut *device, launch);
        let (layers, spans) = (&body.layers.0, live.count::<Span>());
        // A layer's attention once every tile stored its rows: the rows attended as items written
        // to the cache before, the deferred rows after.
        let attend = |device: &mut Device,
                      layer: &Layer,
                      slot: &mut launch::Layer<'_>,
                      places: &mut Places<Size>| {
            let Sequential(Residual(_, attention, _), _) = layer;
            let Attention {
                cache: stage,
                attend,
                output,
                ..
            } = &attention.0;
            let (context, pool) = (&call.context.0, &mut slot.0.1);
            let operands = (&context.write, &context.read);
            let deferred = stage.forward(
                &mut *device,
                (),
                &mut *pool,
                (&places.stored.retained, operands, live),
            );
            let history = Retained {
                pool: &**pool,
                key: &places.stored.retained.0,
                value: &places.stored.retained.1,
                read: &stage.read,
            };
            let rows = (&places.stored.queries, &history, &context.read);
            // The dense output projection stages nothing at the boundary.
            let mut staged = ();
            let boundary = (&output.0, &mut staged);
            let rows = (rows.0, rows.1, rows.2, boundary, &mut places.attended.0);
            attend.forward(&mut *device, (), &mut (), rows);
            let rows = (deferred, &places.stored.retained, &context.write);
            stage.forward(&mut *device, (), &mut *pool, rows);
            output.forward(device, (), &mut (), Begin(staged))
        };
        let mut places = Places::new();
        for span in 0..spans {
            let tile = live.tile::<Span>(span);
            let tokenwise = Tokenwise(&*embedded, &layers[0]);
            let rows = tokenwise.forward(&mut *device, call, &mut places.stored, tile);
            places.input.spill(&mut *device, rows.value(), tile);
        }
        for index in 0..LAYERS - 1 {
            let (layer, next) = (&layers[index], &layers[index + 1]);
            let mut projecting = attend(&mut *device, layer, &mut slots[index], &mut places);
            let mut ahead = Places::new();
            for span in 0..spans {
                let tile = live.tile::<Span>(span);
                let input = tile.with(places.input.fill(&mut *device, tile));
                let output = (tile.with(&places.attended), &mut projecting);
                let rows = Tokenwise(layer, next).forward(
                    &mut *device,
                    call,
                    &mut ahead.stored,
                    (output, input),
                );
                ahead.input.spill(&mut *device, rows.value(), tile);
            }
            places = ahead;
        }
        let last = &layers[LAYERS - 1];
        let mut projecting = attend(&mut *device, last, &mut slots[LAYERS - 1], &mut places);
        let mut head: Rows<Size> = HbmTensor::new();
        for span in 0..spans {
            let tile = live.tile::<Span>(span);
            let input = tile.with(places.input.fill(&mut *device, tile));
            let output = (tile.with(&places.attended), &mut projecting);
            let rows = end(&mut *device, last, output, input);
            let rows = norm.forward(&mut *device, (), &mut (), &rows);
            head.spill(&mut *device, rows.value(), tile);
        }
        causal.lm_head.read(device, &head, call.context.1, tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3-30B-A3B's launch runs by the decoder family's loops on its launch's input and state.
    #[test]
    fn runs_by_the_loops() {
        fn runs<T>()
        where
            for<'i, 'a, 's, 'b> T:
                Module<&'i mut launch::Input<'a, Size>, (), launch::State<'s, 'b, Size>>,
        {
        }
        runs::<ForCausalLM>();
        let _ = <Size as Decode>::Rows::SIZE;
    }
}
