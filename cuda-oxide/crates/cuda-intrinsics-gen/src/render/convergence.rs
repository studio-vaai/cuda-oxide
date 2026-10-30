/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Validate catalog convergence facts before emitting intrinsic lowering recipes.
//! No symbol classifier or catalog reader is generated for the LLVM exporter.

use std::collections::BTreeMap;

use anyhow::{Result, ensure};

use crate::model::CatalogFile;

/// Collect exact emitted LLVM names, resolving overloads before classification.
/// Duplicate catalog routes must agree on convergence; ambiguity fails generation.
pub(super) fn convergence_facts(catalog: &CatalogFile) -> Result<BTreeMap<&str, bool>> {
    let mut facts = BTreeMap::new();
    for record in &catalog.intrinsics {
        let Some(llvm) = &record.llvm else { continue };
        let name = llvm.resolved_symbol.as_deref().unwrap_or(&llvm.symbol);
        if !name.starts_with("llvm.nvvm.") {
            continue;
        }
        if let Some(previous) = facts.insert(name, record.semantics.convergent) {
            ensure!(
                previous == record.semantics.convergent,
                "conflicting convergence semantics for LLVM intrinsic `{name}`"
            );
        }
    }
    Ok(facts)
}
