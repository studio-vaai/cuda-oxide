/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::render::convergence::convergence_facts;

/// Load the checked-in catalog used by generation and its generated lowering recipes.
/// This exercises real aliases and overload names rather than an invented subset.
fn catalog() -> CatalogFile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../intrinsics/catalog.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// Reviewed collectives and special registers retain their distinct semantics.
/// Non-convergent entries must not classify an unreviewed spelling by prefix.
#[test]
fn resolved_names_are_exact_and_catalog_semantics_are_preserved() {
    let catalog = catalog();
    let facts = convergence_facts(&catalog).unwrap();
    assert_eq!(facts.get("llvm.nvvm.activemask"), Some(&true));
    assert_eq!(facts.get("llvm.nvvm.match.all.sync.i32p"), Some(&true));
    assert_eq!(facts.get("llvm.nvvm.read.ptx.sreg.tid.x"), Some(&false));
    assert!(!facts.contains_key("llvm.nvvm.read.ptx.sreg.future"));
    assert!(!facts.contains_key("llvm.nvvm.match.all.sync.i32p.extra"));
}

/// A second catalog route cannot silently weaken an intrinsic's requirement.
/// Conflicting facts must fail before lowering recipes are generated.
#[test]
fn conflicting_duplicate_semantics_are_rejected() {
    let mut catalog = catalog();
    let mut conflict = catalog
        .intrinsics
        .iter()
        .find(|record| {
            record
                .llvm
                .as_ref()
                .is_some_and(|llvm| llvm.symbol == "llvm.nvvm.activemask")
        })
        .unwrap()
        .clone();
    conflict.semantics.convergent = false;
    catalog.intrinsics.push(conflict);
    assert!(
        convergence_facts(&catalog)
            .unwrap_err()
            .to_string()
            .contains("llvm.nvvm.activemask")
    );
}

/// Catalog semantics become constants at the recipe that selects the intrinsic.
/// The exporter receives typed facts rather than another symbol classification table.
#[test]
fn lowering_carries_catalog_convergence_with_intrinsic_names() {
    let catalog = catalog();
    let lowering = render_lowering(&catalog, "test-hash");
    assert!(lowering.contains("Intrinsic::new(\"llvm_nvvm_read_ptx_sreg_tid_x\", false)"));
    assert!(lowering.contains("Intrinsic::new(\"llvm_nvvm_barrier_cluster_wait\", true)"));
    let outputs = all_outputs(&catalog, "{}".into(), "test-hash").unwrap();
    assert!(
        !outputs
            .keys()
            .any(|path| path.starts_with("crates/llvm-export"))
    );
}
