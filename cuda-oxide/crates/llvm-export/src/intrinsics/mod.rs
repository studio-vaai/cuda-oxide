/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Intrinsic semantics proven by lowering and carried on LLVM declarations.
//! Missing facts remain unknown; explicit LLVM attributes may add protection.

use crate::ops::FuncOp;
use pliron::{
    context::{Context, Ptr},
    identifier::Identifier,
    operation::Operation,
};
use pliron_derive::pliron_attr;

/// Convergence of the exact intrinsic selected by a lowering recipe.
/// Unlike absence of LLVM's `convergent`, false records a proven relaxation.
#[pliron_attr(name = "llvm.intrinsic_convergence", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntrinsicConvergenceAttr {
    /// Whether the primitive requires coordinated thread participation.
    /// Purity and memory effects are independent and supply no such proof.
    pub convergent: bool,
}

/// Internal transport key, separate from LLVM's function attribute dictionary.
/// The textual exporter reads the fact but never emits this key.
const CONVERGENCE_KEY: &str = "cuda_oxide_intrinsic_convergence";

/// Require a declaration before accepting a primitive semantic fact.
/// Function bodies must instead be analyzed over their complete call graph.
fn require_declaration(ctx: &Context, op: Ptr<Operation>) -> Result<(), String> {
    if Operation::get_op::<FuncOp>(op, ctx).is_none() || op.deref(ctx).regions().count() != 0 {
        return Err("intrinsic convergence facts require an LLVM function declaration".into());
    }
    Ok(())
}

/// Read and validate a lowering fact, preserving unknown when it is absent.
/// Reject mistyped transport instead of silently relaxing convergence.
pub fn intrinsic_convergence(
    ctx: &Context,
    op: Ptr<Operation>,
) -> Result<Option<IntrinsicConvergenceAttr>, String> {
    let key = Identifier::try_new(CONVERGENCE_KEY.into()).unwrap();
    let operation = op.deref(ctx);
    if !operation.attributes.0.contains_key(&key) {
        return Ok(None);
    }
    require_declaration(ctx, op)?;
    operation
        .attributes
        .get::<IntrinsicConvergenceAttr>(&key)
        .copied()
        .map(Some)
        .ok_or_else(|| "intrinsic convergence fact has the wrong attribute type".into())
}

/// Attach an exact intrinsic's proven requirement, allowing identical reuse.
/// Conflicting recipes for a shared declaration are rejected in either order.
pub fn set_intrinsic_convergence(
    ctx: &mut Context,
    op: Ptr<Operation>,
    fact: IntrinsicConvergenceAttr,
) -> Result<(), String> {
    require_declaration(ctx, op)?;
    if let Some(previous) = intrinsic_convergence(ctx, op)?
        && previous != fact
    {
        return Err("conflicting convergence facts for an intrinsic declaration".into());
    }
    let key = Identifier::try_new(CONVERGENCE_KEY.into()).unwrap();
    op.deref_mut(ctx).attributes.set(key, fact);
    Ok(())
}
