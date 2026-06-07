/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM function-parameter attributes: the derivation-neutral description and
//! its rendering to textual LLVM.
//!
//! This type lives in the exporter crate on purpose: it is the *emission*
//! vocabulary for LLVM parameter attributes, and this crate owns `.ll` textual
//! emission. The backend (`rustc-codegen-cuda`) decodes rustc's
//! `FnAbi`/`ArgAttributes` into [`ArgAttrs`] (the *policy* step); the pipeline
//! transports it; `mir-lower` remaps it onto the flattened LLVM parameters and
//! renders it here via [`ArgAttrs::to_fragment`] (the *emission* step). Keeping
//! policy and emission apart mirrors rustc's own `rustc_ty_utils::abi` (policy)
//! vs `rustc_codegen_llvm::abi` (emission) split, and lets a future stable-only
//! derivation feed the very same renderer.
//!
//! # Per-attribute applicability
//!
//! Not every attribute is valid on every parameter: `noalias`/`readonly`/
//! `nonnull`/`dereferenceable`/`align` are pointer-only, `signext`/`zeroext`
//! are integer-only, and `noundef` applies to any value. [`ArgAttrs`] carries
//! the *decision* for a source parameter; [`to_fragment`](ArgAttrs::to_fragment)
//! takes the [`LlvmParamClass`] of the lowered parameter the attribute will
//! actually land on and emits only the tokens valid there. This is what lets a
//! `noundef` rustc attached to a scalar survive while the pointer-only tokens
//! that would be illegal on that scalar are dropped.

/// Integer sign-/zero-extension for a narrow-integer parameter at a (non-
/// inlined) call boundary — LLVM `signext` / `zeroext`. Mirrors rustc's
/// `rustc_target::callconv::ArgExtension`. Only meaningful on integer
/// parameters; ignored elsewhere by [`ArgAttrs::to_fragment`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ArgExt {
    /// No extension.
    #[default]
    None,
    /// `signext` — extend with the sign bit.
    Sign,
    /// `zeroext` — extend with zero.
    Zero,
}

/// The kind of lowered LLVM parameter an [`ArgAttrs`] is being placed on, used
/// to gate each attribute by applicability. Determined from the flattened LLVM
/// parameter type by `mir-lower`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlvmParamClass {
    /// An LLVM `ptr`. Carries the pointer attributes (`noalias`, `readonly`,
    /// `nonnull`, `dereferenceable`, `align`) plus `noundef`.
    Pointer,
    /// An LLVM integer (`iN`). Carries `signext`/`zeroext` plus `noundef`.
    Integer,
    /// Any other lowered type (float, etc.). Carries `noundef` only.
    Other,
}

/// Derivation-neutral description of the LLVM attributes for a single function
/// parameter.
///
/// This deliberately carries no rustc types: the backend decodes rustc's
/// `FnAbi`/`ArgAttributes` into this struct (the *policy* step), and this crate
/// renders it onto the IR (the *emission* step).
///
/// Soundness note: for ordinary parameters these values are a faithful
/// pass-through of `FnAbi`, so the aliasing guarantees match Rust's reference
/// rules (e.g. an immutable shared `&T` to `Freeze` data gets *both* `noalias`
/// and `readonly`; raw pointers get nothing). The one type that goes beyond
/// `FnAbi` is `cuda_device::DisjointSlice<T>`: it wraps a `*mut T` (no `FnAbi`
/// attributes) but its `unsafe` contract asserts exclusive, aligned access, so
/// the backend synthesizes `noalias`/`align`/`nonnull`/`noundef` from that
/// contract. Either way this struct only *carries* the decision; it does not
/// make it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArgAttrs {
    /// `noalias` — only for unique borrows (`&mut T`, `Box<T>`), never shared
    /// refs or raw pointers. Pointer-only.
    pub noalias: bool,
    /// `readonly` — the callee does not write through this pointer. Pointer-only.
    pub readonly: bool,
    /// `nonnull` — the pointer is guaranteed non-null. Pointer-only.
    pub nonnull: bool,
    /// `noundef` — the value is guaranteed to be well-defined. Valid on any
    /// parameter (pointer, integer, or other).
    pub noundef: bool,
    /// `dereferenceable(N)` — at least `N` bytes are dereferenceable. Absent for
    /// unsized pointees (e.g. slice data pointers), where the size is unknown.
    /// Pointer-only.
    pub dereferenceable: Option<u64>,
    /// `align(N)` — the pointee's minimum alignment in bytes. Pointer-only.
    pub align: Option<u64>,
    /// `signext` / `zeroext` for a narrow integer at a call boundary.
    /// Integer-only.
    pub ext: ArgExt,
}

impl ArgAttrs {
    /// True when nothing would ever be emitted, on any parameter class. Used by
    /// the backend to collapse "no attributes" to `None` before transport.
    pub fn is_empty(&self) -> bool {
        !self.noalias
            && !self.readonly
            && !self.nonnull
            && !self.noundef
            && self.dereferenceable.is_none()
            && self.align.is_none()
            && self.ext == ArgExt::None
    }

    /// Render to the textual LLVM parameter-attribute fragment that sits between
    /// a parameter's type and its value name, e.g. `noalias nonnull align 4` or
    /// `zeroext noundef`.
    ///
    /// `class` is the kind of lowered parameter the attribute will land on; only
    /// the tokens valid for that class are emitted (see the module docs). Returns
    /// `None` when nothing applicable is set, so callers can skip the parameter.
    /// The token set is valid both on the textual `.ll` path (modern `llc`) and
    /// on the NVVM-IR path (LLVM 20 dialect); only LLVM-21-era attributes such as
    /// `captures(...)` are intentionally omitted.
    pub fn to_fragment(&self, class: LlvmParamClass) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();

        // Integer extension comes first, matching clang's `iN zeroext` ordering.
        if class == LlvmParamClass::Integer {
            match self.ext {
                ArgExt::Sign => parts.push("signext".to_string()),
                ArgExt::Zero => parts.push("zeroext".to_string()),
                ArgExt::None => {}
            }
        }

        let is_ptr = class == LlvmParamClass::Pointer;
        if is_ptr && self.noalias {
            parts.push("noalias".to_string());
        }
        if is_ptr && self.readonly {
            parts.push("readonly".to_string());
        }
        if is_ptr && self.nonnull {
            parts.push("nonnull".to_string());
        }
        // `noundef` is valid on any value type.
        if self.noundef {
            parts.push("noundef".to_string());
        }
        if is_ptr && let Some(n) = self.dereferenceable {
            parts.push(format!("dereferenceable({n})"));
        }
        if is_ptr && let Some(a) = self.align {
            parts.push(format!("align {a}"));
        }

        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    }
}
