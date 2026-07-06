/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM function-parameter attributes: the derivation-neutral description, the
//! first-class IR carrier, and the textual rendering.
//!
//! These live in the exporter crate on purpose: they are the *emission*
//! vocabulary for LLVM parameter attributes, and this crate owns `.ll` textual
//! emission. Three representations, one per pipeline stage:
//!
//! 1. [`ArgAttrs`] — the **policy** representation. The backend
//!    (`rustc-codegen-cuda`) decodes rustc's `FnAbi`/`ArgAttributes` into this
//!    plain struct (see `rustc_codegen_cuda::abi_attrs`). It carries no rustc
//!    and no pliron types, so the backend can produce it without building IR.
//! 2. [`LlvmParamAttrsAttr`] — the **carrier**. A first-class pliron attribute
//!    that rides on the function op. `mir-importer` builds one per *source*
//!    parameter from the policy [`ArgAttrs`] and stamps a
//!    [`VecAttr`](pliron::builtin::attributes::VecAttr) of them onto the
//!    `dialect-mir` func op; `mir-lower` reads that vector, remaps it onto the
//!    flattened LLVM parameters, and re-stamps it onto the LLVM func op. No
//!    side-channel map, no pre-rendered strings — the attribute *is* the IR.
//! 3. Textual `.ll` — the **emission**. The exporter reads the flattened
//!    carrier off the LLVM func op and renders each parameter via
//!    [`ArgAttrs::to_fragment`], gated by the lowered parameter's
//!    [`LlvmParamClass`].
//!
//! Keeping policy, carrier, and emission apart mirrors rustc's own
//! `rustc_ty_utils::abi` (policy) vs `rustc_codegen_llvm::abi` (emission) split.
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
//!
//! Soundness note: for ordinary parameters these values are a faithful
//! pass-through of `FnAbi`, so the aliasing guarantees match Rust's reference
//! rules (an immutable shared `&T` to `Freeze` data gets *both* `noalias` and
//! `readonly`; raw pointers get nothing). The one type that goes beyond
//! `FnAbi` is `cuda_device::DisjointSlice<T>`: it wraps a `*mut T` (no `FnAbi`
//! attributes) but its `unsafe` contract asserts exclusive, aligned access, so
//! the backend synthesizes `noalias`/`align`/`nonnull`/`noundef` from that
//! contract. Either way these types only *carry* the decision; they do not
//! make it.

use pliron::derive::pliron_attr;

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
/// parameter type by `mir-lower` / the exporter.
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
/// parameter — the *policy* representation.
///
/// This deliberately carries no rustc and no pliron types: the backend decodes
/// rustc's `FnAbi`/`ArgAttributes` into this struct, and downstream stages turn
/// it into the IR carrier ([`LlvmParamAttrsAttr`]) and, finally, text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArgAttrs {
    /// `noalias` — only for unique borrows (`&mut T`, `Box<T>`) or a
    /// contract-backed `DisjointSlice`, never shared refs or raw pointers.
    /// Pointer-only.
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
    /// True when nothing would ever be emitted, on any parameter class. Used to
    /// collapse "no attributes" to `None` before transport.
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

/// First-class pliron attribute carrying one parameter's LLVM attributes — the
/// *carrier* representation.
///
/// This is what rides on the IR. `mir-importer` builds one per source parameter
/// from the policy [`ArgAttrs`] and stamps a
/// [`VecAttr`](pliron::builtin::attributes::VecAttr) of them onto the func op;
/// `mir-lower` remaps that vector onto the flattened LLVM parameters and
/// re-stamps it onto the LLVM func op; the exporter reads it back and renders.
///
/// The optional `dereferenceable(N)` / `align(N)` of [`ArgAttrs`] are encoded
/// with `0` meaning *absent* — neither a zero dereferenceable span nor a zero
/// alignment is meaningful in LLVM, so the sentinel is unambiguous. Integer
/// extension is split into the two mutually-exclusive `signext`/`zeroext`
/// booleans that map directly onto the emitted tokens, so the carrier needs no
/// nested enum attribute.
#[pliron_attr(name = "llvm.param_attrs", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash, Default)]
pub struct LlvmParamAttrsAttr {
    /// `noalias`. Pointer-only.
    pub noalias: bool,
    /// `readonly`. Pointer-only.
    pub readonly: bool,
    /// `nonnull`. Pointer-only.
    pub nonnull: bool,
    /// `noundef`. Valid on any value.
    pub noundef: bool,
    /// `signext`. Integer-only. Mutually exclusive with [`Self::zeroext`].
    pub signext: bool,
    /// `zeroext`. Integer-only. Mutually exclusive with [`Self::signext`].
    pub zeroext: bool,
    /// `dereferenceable(N)`; `0` means absent. Pointer-only.
    pub dereferenceable: u64,
    /// `align(N)`; `0` means absent. Pointer-only.
    pub align: u64,
}

impl LlvmParamAttrsAttr {
    /// True when the carrier holds no attribute at all, so callers can skip a
    /// parameter without materializing a fragment.
    pub fn is_empty(&self) -> bool {
        self.to_arg_attrs().is_empty()
    }

    /// Rebuild the policy [`ArgAttrs`] this carrier was encoded from, so the
    /// exporter can render it via [`ArgAttrs::to_fragment`].
    pub fn to_arg_attrs(&self) -> ArgAttrs {
        ArgAttrs {
            noalias: self.noalias,
            readonly: self.readonly,
            nonnull: self.nonnull,
            noundef: self.noundef,
            dereferenceable: (self.dereferenceable != 0).then_some(self.dereferenceable),
            align: (self.align != 0).then_some(self.align),
            ext: match (self.signext, self.zeroext) {
                (true, _) => ArgExt::Sign,
                (_, true) => ArgExt::Zero,
                _ => ArgExt::None,
            },
        }
    }

    /// Render this parameter's attributes for the lowered-parameter `class`,
    /// deferring the applicability gate to [`ArgAttrs::to_fragment`].
    pub fn to_fragment(&self, class: LlvmParamClass) -> Option<String> {
        self.to_arg_attrs().to_fragment(class)
    }
}

impl From<&ArgAttrs> for LlvmParamAttrsAttr {
    fn from(attrs: &ArgAttrs) -> Self {
        LlvmParamAttrsAttr {
            noalias: attrs.noalias,
            readonly: attrs.readonly,
            nonnull: attrs.nonnull,
            noundef: attrs.noundef,
            signext: attrs.ext == ArgExt::Sign,
            zeroext: attrs.ext == ArgExt::Zero,
            dereferenceable: attrs.dereferenceable.unwrap_or(0),
            align: attrs.align.unwrap_or(0),
        }
    }
}

/// The op-attribute dictionary key under which the per-parameter
/// [`VecAttr`](pliron::builtin::attributes::VecAttr) of [`LlvmParamAttrsAttr`]
/// is stamped — first on the `dialect-mir` func op (source-parameter indexed),
/// then, after remapping, on the LLVM func op (flattened-parameter indexed).
/// Shared by the producer (`mir-importer`), the remapper (`mir-lower`), and the
/// consumer (this crate's exporter) so the three never drift.
pub const ARG_ATTRS_KEY: &str = "llvm_arg_attrs";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_family_renders_in_canonical_order() {
        let attrs = ArgAttrs {
            noalias: true,
            readonly: true,
            nonnull: true,
            noundef: true,
            dereferenceable: Some(16),
            align: Some(4),
            ext: ArgExt::None,
        };
        assert_eq!(
            attrs.to_fragment(LlvmParamClass::Pointer).as_deref(),
            Some("noalias readonly nonnull noundef dereferenceable(16) align 4")
        );
    }

    #[test]
    fn pointer_tokens_are_dropped_on_non_pointer_classes() {
        // A `noundef` rustc attached to a scalar survives; pointer-only tokens
        // that would be illegal on that scalar are dropped.
        let attrs = ArgAttrs {
            noalias: true,
            nonnull: true,
            noundef: true,
            align: Some(4),
            ..Default::default()
        };
        assert_eq!(
            attrs.to_fragment(LlvmParamClass::Integer).as_deref(),
            Some("noundef")
        );
        assert_eq!(
            attrs.to_fragment(LlvmParamClass::Other).as_deref(),
            Some("noundef")
        );
    }

    #[test]
    fn integer_extension_only_on_integers() {
        let sext = ArgAttrs {
            ext: ArgExt::Sign,
            noundef: true,
            ..Default::default()
        };
        assert_eq!(
            sext.to_fragment(LlvmParamClass::Integer).as_deref(),
            Some("signext noundef")
        );
        // On a pointer the extension is meaningless and dropped.
        assert_eq!(
            sext.to_fragment(LlvmParamClass::Pointer).as_deref(),
            Some("noundef")
        );
    }

    #[test]
    fn empty_renders_to_nothing() {
        assert!(ArgAttrs::default().is_empty());
        assert_eq!(
            ArgAttrs::default().to_fragment(LlvmParamClass::Pointer),
            None
        );
    }

    #[test]
    fn carrier_round_trips_through_policy_struct() {
        for attrs in [
            ArgAttrs {
                noalias: true,
                nonnull: true,
                noundef: true,
                dereferenceable: Some(8),
                align: Some(8),
                ..Default::default()
            },
            ArgAttrs {
                readonly: true,
                ext: ArgExt::Zero,
                ..Default::default()
            },
            ArgAttrs::default(),
        ] {
            let carrier = LlvmParamAttrsAttr::from(&attrs);
            assert_eq!(carrier.to_arg_attrs(), attrs);
            assert_eq!(carrier.is_empty(), attrs.is_empty());
        }
    }
}
