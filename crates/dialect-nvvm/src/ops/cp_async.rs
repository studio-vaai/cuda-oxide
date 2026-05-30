/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Non-bulk asynchronous copy (`cp.async.*`) operations for Ampere+ GPUs.
//!
//! These are the per-element LDGSTS copies (global→shared), distinct from the
//! descriptor-based TMA bulk copies in [`super::tma`]. They are used for
//! software-pipelined prefetch of scattered gathers.
//!
//! ```text
//! ┌──────────────────────────────┬───────────────────────────┬───────────────┐
//! │ Op                           │ PTX                       │ Min SM        │
//! ├──────────────────────────────┼───────────────────────────┼───────────────┤
//! │ CpAsyncCaShared16Op          │ cp.async.ca.shared.global │ sm_80         │
//! │ CpAsyncCommitGroupOp         │ cp.async.commit_group     │ sm_80         │
//! │ CpAsyncWaitGroupOp           │ cp.async.wait_group N     │ sm_80         │
//! └──────────────────────────────┴───────────────────────────┴───────────────┘
//! ```
//!
//! # Requirements
//!
//! - **PTX ISA**: 7.0+
//! - **Architecture**: sm_80+ (Ampere and newer)

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Async 16-byte copy from global to shared memory (`.ca` cache hint).
///
/// Corresponds to `llvm.nvvm.cp.async.ca.shared.global.16`.
///
/// # Operands
///
/// - `dst` (ptr addrspace(3)): destination in shared memory
/// - `src` (ptr addrspace(1)): source in global memory
///
/// # Results
///
/// - None (completion tracked via commit/wait groups)
#[pliron_op(
    name = "nvvm.cp_async_ca_shared_global_16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct CpAsyncCaShared16Op;

impl CpAsyncCaShared16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCaShared16Op { op }
    }
}

/// Commit all preceding `cp.async` copies into a completion group.
///
/// Lowered to inline PTX `cp.async.commit_group;`.
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_commit_group",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct CpAsyncCommitGroupOp;

impl CpAsyncCommitGroupOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCommitGroupOp { op }
    }
}

/// Wait until at most `n` committed `cp.async` groups are still pending.
///
/// Lowered to inline PTX `cp.async.wait_group N;`. `n` must be an immediate.
///
/// # Operands
///
/// - `n` (i32): maximum number of pending groups (0 = wait for all)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_wait_group",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct CpAsyncWaitGroupOp;

impl CpAsyncWaitGroupOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncWaitGroupOp { op }
    }
}

/// Register non-bulk cp.async operations with the context.
pub(super) fn register(ctx: &mut Context) {
    CpAsyncCaShared16Op::register(ctx);
    CpAsyncCommitGroupOp::register(ctx);
    CpAsyncWaitGroupOp::register(ctx);
}
