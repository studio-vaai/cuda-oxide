/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::intrinsics::intrinsic_convergence;

/// An NVVM spelling supplies no proof without its typed lowering fact.
/// Recognizable names and future variants share the conservative unknown policy.
#[test]
fn nvvm_names_alone_do_not_prove_nonconvergence() {
    for name in [
        "llvm.nvvm.read.ptx.sreg.tid.x",
        "llvm.nvvm.activemask",
        "llvm.nvvm.read.ptx.sreg.future",
        "llvm.nvvm.match.all.sync.i32p.extra",
    ] {
        assert_eq!(intrinsic_convergence(name), None, "{name}");
    }
}

/// Standard scalar LLVM policy remains independent of catalog-owned NVVM semantics.
/// Unknown core intrinsic families retain the conservative fallback.
#[test]
fn core_intrinsic_policy_is_preserved() {
    assert_eq!(intrinsic_convergence("llvm.sqrt.f32"), Some(false));
    assert_eq!(intrinsic_convergence("llvm.trap"), Some(false));
    assert_eq!(intrinsic_convergence("llvm.experimental.unknown"), None);
}
