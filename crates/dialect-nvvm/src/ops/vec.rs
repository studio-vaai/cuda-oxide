// LOCAL EXPERIMENT for studio-vaai cloth-solver-cuda — DO NOT UPSTREAM.
//
// Vector inline-PTX escape-hatch dialect op. Only the global vector
// atomic-add hatch lives here — plain v4 load/store no longer need
// inline-PTX once the Gap 1 export-alignment fix landed (`dialect-llvm/src/
// export.rs` now emits `align N` on plain Load/Store, so libNVVM fuses
// `*ptr_to_[f32; 4]` / `*ptr_to_CuSimd<f32, 4>` patterns into LDG/STG.E.128
// natively).

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Native vector global atomic-add: `red.global.add.v4.f32 [p], {a, b, c, d};`.
///
/// SASS: `REDG.E.ADD.F32x4`. Single instruction, no CAS loop. Available on
/// sm_9.0+ (Hopper) and Blackwell. Result is discarded (no per-element
/// previous values returned).
#[pliron_op(
    name = "nvvm.atomic_add_global_v4_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct AtomicAddGlobalV4F32Op;

impl AtomicAddGlobalV4F32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        AtomicAddGlobalV4F32Op { op }
    }
}

pub fn register(ctx: &mut Context) {
    AtomicAddGlobalV4F32Op::register(ctx);
}
