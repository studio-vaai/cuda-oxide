// LOCAL EXPERIMENT for studio-vaai cloth-solver-cuda — DO NOT UPSTREAM.
//
// Vector inline-PTX escape-hatch dialect ops.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Coalesced 16-byte vector store: `st.global.v4.f32 [p], {a, b, c, d};`.
#[pliron_op(
    name = "nvvm.st_global_v4_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct StGlobalV4F32Op;

impl StGlobalV4F32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        StGlobalV4F32Op { op }
    }
}

/// Coalesced 16-byte vector load: `ld.global.v4.f32 {a, b, c, d}, [p];`.
///
/// Operand: `p` (ptr). Result: 4 f32 values the importer wires into the
/// destructured `[f32; 4]` slots.
#[pliron_op(
    name = "nvvm.ld_global_v4_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<4>],
)]
pub struct LdGlobalV4F32Op;

impl LdGlobalV4F32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        LdGlobalV4F32Op { op }
    }
}

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
    StGlobalV4F32Op::register(ctx);
    LdGlobalV4F32Op::register(ctx);
    AtomicAddGlobalV4F32Op::register(ctx);
}
