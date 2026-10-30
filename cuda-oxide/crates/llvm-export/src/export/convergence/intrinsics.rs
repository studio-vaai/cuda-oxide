/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Intrinsic convergence policy shared by analysis and emission queries.
//!
//! Core LLVM intrinsic families are classified here. NVVM semantics arrive
//! as typed lowering facts; unrecognized declarations remain unknown.

/// Return the known convergence requirement, leaving unrecognized names unknown.
/// An unknown external target must retain convergence protection at its callers.
pub(super) fn intrinsic_convergence(name: &str) -> Option<bool> {
    if known_nonconvergent_core_intrinsic(name) {
        Some(false)
    } else {
        None
    }
}

/// These core intrinsic families do not depend on inter-thread participation.
/// Memory effects are deliberately not used as a proof of non-convergence.
fn known_nonconvergent_core_intrinsic(name: &str) -> bool {
    matches!(name, "llvm.trap" | "llvm.debugtrap" | "llvm.assume")
        || [
            "llvm.abs.",
            "llvm.bswap.",
            "llvm.bitreverse.",
            "llvm.ctlz.",
            "llvm.cttz.",
            "llvm.ctpop.",
            "llvm.fshl.",
            "llvm.fshr.",
            "llvm.fabs.",
            "llvm.copysign.",
            "llvm.sqrt.",
            "llvm.pow.",
            "llvm.powi.",
            "llvm.sin.",
            "llvm.cos.",
            "llvm.exp.",
            "llvm.exp2.",
            "llvm.log.",
            "llvm.log2.",
            "llvm.log10.",
            "llvm.floor.",
            "llvm.ceil.",
            "llvm.trunc.",
            "llvm.rint.",
            "llvm.nearbyint.",
            "llvm.round.",
            "llvm.roundeven.",
            "llvm.fma.",
            "llvm.fmuladd.",
            "llvm.minnum.",
            "llvm.maxnum.",
            "llvm.minimum.",
            "llvm.maximum.",
            "llvm.sadd.with.overflow.",
            "llvm.uadd.with.overflow.",
            "llvm.ssub.with.overflow.",
            "llvm.usub.with.overflow.",
            "llvm.smul.with.overflow.",
            "llvm.umul.with.overflow.",
            "llvm.sadd.sat.",
            "llvm.uadd.sat.",
            "llvm.ssub.sat.",
            "llvm.usub.sat.",
            "llvm.memcpy.",
            "llvm.memmove.",
            "llvm.memset.",
            "llvm.lifetime.start.",
            "llvm.lifetime.end.",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}
