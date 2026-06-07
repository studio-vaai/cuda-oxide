/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! # Parameter attributes from `FnAbi` (Route A, *derive* step)
//!
//! This module reads the LLVM parameter attributes rustc already computed for a
//! function and decodes them into the derivation-neutral
//! [`mir_importer::ArgAttrs`]. It deliberately does **not** re-derive the
//! attribute policy by hand. The bulk of the policy worth understanding is on
//! the pointer family (`noalias`/`readonly` — see below); `noundef` (any value)
//! and `signext`/`zeroext` (narrow integers) are decoded the same faithful way.
//!
//! ## Why reuse `FnAbi` rather than re-derive
//!
//! rustc's attribute policy encodes subtle soundness decisions that are easy to
//! get wrong from memory. On the pinned toolchain (see
//! `rustc_ty_utils::abi::ptr_attrs`-style logic):
//!
//! - `&mut T` (when `T: Unpin`) / `Box<T>` (when `Unpin` + global) → `noalias`
//! - `&T` where `T: Freeze` (no `UnsafeCell`) → **`noalias` *and* `readonly`**.
//!   This surprises people — rustc's `noalias` is defined by memory
//!   dependencies, not pointer equality, so an immutable shared reference
//!   qualifies. (A common, outdated belief is that shared refs never get
//!   `noalias`; that was true years ago, but not on this toolchain.)
//! - `&T` containing an `UnsafeCell` (not `Freeze`) → neither
//! - raw pointers (`*const T` / `*mut T`) → no aliasing guarantees at all
//!
//! Hand-rolling this (especially the `Freeze`/`Unpin` checks) risks either
//! reintroducing a known unsoundness class — a reference wrongly marked
//! `noalias`, miscompiled when another argument writes the same memory — or
//! silently *dropping* a sound attribute rustc intends (e.g. `noalias` on a
//! shared `&[f32]`). So we read the bits straight out of `FnAbi`; faithful
//! pass-through is the whole point, and it stays correct as rustc's policy
//! evolves.
//!
//! ## The one exception: `DisjointSlice`
//!
//! `cuda_device::DisjointSlice<T>` wraps a `*mut T`, so rustc gives its `FnAbi`
//! no attributes at all. But the type *exists* to assert, via its `unsafe`
//! `from_raw_parts` contract, exactly what these attributes encode: exclusive,
//! aligned, valid access over a disjoint element range. So for this one type we
//! synthesize the attributes from the contract (`noalias` + `align` + `nonnull`
//! + `noundef`) rather than from `FnAbi` — see [`disjoint_slice_attrs`]. This is
//! the same kind of trust rustc places in `&mut T → noalias`: a contract the
//! caller must uphold, whose violation is UB. Choosing `DisjointSlice` for a
//! parameter *is* the opt-in.
//!
//! ## Module paths (pinned nightly: `nightly-2026-04-03`)
//!
//! The ABI types live in `rustc_target::callconv` (this was
//! `rustc_target::abi::call` a few releases ago — check here first on a version
//! bump):
//!
//! - `rustc_target::callconv::{FnAbi, ArgAbi, PassMode, ArgAttributes, ArgAttribute, ArgExtension}`
//! - `ArgAttribute` bitflags: `NoAlias` (`1 << 3`), `NonNull` (`1 << 4`),
//!   `ReadOnly` (`1 << 5`), `NoUndef` (`1 << 7`).
//! - `ArgExtension` enum (the `arg_ext` field): `None` / `Zext` / `Sext` —
//!   narrow integer sign/zero extension at a call boundary (`signext` /
//!   `zeroext`).
//! - Query entry point: `TyCtxt::fn_abi_of_instance(PseudoCanonicalInput<(Instance, &List<Ty>)>)`
//!   (`rustc_middle::ty::mod.rs`), built via `TypingEnv::as_query_input`.
//!   `Size`/`Align` (`rustc_abi`, re-exported through `rustc_target`) expose
//!   `.bytes()`.

use mir_importer::{ArgAttrs, ArgExt};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, TyKind, TypingEnv};
use rustc_target::callconv::{ArgAttribute, ArgAttributes, ArgExtension, PassMode};

/// Derive faithful per-parameter LLVM attributes for `instance` from the
/// `FnAbi` rustc already computed.
///
/// The returned vector has one entry per **formal parameter**, in source order
/// (matching `fn_sig().inputs()` and the `mir.func` argument list). `None`
/// means that parameter carries no attributes at all (e.g. raw pointers,
/// aggregates, or `Cast`/`Ignore` modes). Each `Some` is a faithful decode of
/// the pointer family (`noalias`/`readonly`/`nonnull`/`dereferenceable`/`align`),
/// plus `noundef` (any value) and `signext`/`zeroext` (narrow integers); the
/// per-attribute applicability is enforced downstream in `mir-lower`.
///
/// Kernels and device functions are monomorphic at codegen, so we query with
/// [`TypingEnv::fully_monomorphized`] and no extra (variadic) arguments. If the
/// ABI query fails for any reason we return an empty vector — emitting *no*
/// attributes is always sound; the only failure mode worth avoiding is emitting
/// *wrong* ones.
pub fn derive_arg_attrs<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> Vec<Option<ArgAttrs>> {
    // `codegen_llvm` reaches the same policy through this exact query; we use the
    // raw `TyCtxt` query rather than the `FnAbiOf` trait because no codegen
    // context implementing that trait exists on the device path.
    let extra_args: &'tcx ty::List<Ty<'tcx>> = ty::List::empty();
    let input = TypingEnv::fully_monomorphized().as_query_input((instance, extra_args));

    let fn_abi = match tcx.fn_abi_of_instance(input) {
        Ok(fn_abi) => fn_abi,
        Err(_) => return Vec::new(),
    };

    fn_abi
        .args
        .iter()
        .map(|arg| {
            // `DisjointSlice<T>` wraps a raw `*mut T`, so its `FnAbi` carries no
            // attributes — but the type asserts an unsafe contract (exclusive,
            // aligned access) that we *can* encode. Synthesize from the contract
            // when we recognize the type; this is the one place we go beyond
            // faithful `FnAbi` pass-through, and only for a type whose whole
            // purpose is to make these guarantees. See `disjoint_slice_attrs`.
            if let Some(attrs) = disjoint_slice_attrs(tcx, arg.layout.ty) {
                return Some(attrs);
            }

            match &arg.mode {
                // Directly-passed (a scalar, or a thin pointer like `&T` /
                // `&mut T` / `Box<T>` / `*mut T`) or indirectly-passed: the
                // attributes are the arg's own `attrs`.
                PassMode::Direct(attrs) | PassMode::Indirect { attrs, .. } => decode(attrs),
                // Fat pointer (`&[T]`, `&dyn Trait`): the data-pointer
                // attributes are in element 0; element 1 is the length/vtable,
                // whose attributes we don't track.
                PassMode::Pair(data_attrs, _meta_attrs) => decode(data_attrs),
                // ZST / cast-through: nothing.
                PassMode::Ignore | PassMode::Cast { .. } => None,
            }
        })
        .collect()
}

/// Synthesize pointer attributes for a `cuda_device::DisjointSlice<T>` argument
/// from the guarantees its `from_raw_parts` contract establishes.
///
/// `DisjointSlice` exists precisely to assert what rustc cannot see through the
/// raw `*mut T` it wraps: each thread holds an exclusive, aligned, valid view of
/// a disjoint element range. Its safety contract requires that "no other live
/// `DisjointSlice` (or `&mut [T]` / `&[T]` / raw access) covers any byte of the
/// range", which is exactly LLVM `noalias` — the same guarantee rustc emits for
/// `&mut T`. Trusting that contract is the *opt-in* a user makes by choosing
/// `DisjointSlice` for a kernel parameter.
///
/// Returns `None` for any other type. When `T`'s layout is available we also
/// emit `align(align_of::<T>())`, which is what lets whole-element loads/stores
/// on the buffer fuse into vectorized `ld/st.global.v*`.
///
/// Detection matches `mir-importer`'s own (the type named `DisjointSlice` from
/// the `cuda_device` crate); see `mir_importer::translator::types`.
fn disjoint_slice_attrs<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<ArgAttrs> {
    let TyKind::Adt(adt_def, args) = ty.kind() else {
        return None;
    };
    let did = adt_def.did();
    if tcx.item_name(did).as_str() != "DisjointSlice"
        || tcx.crate_name(did.krate).as_str() != "cuda_device"
    {
        return None;
    }

    // Element type `T` (first type-kind generic arg of `DisjointSlice<'a, T, IS>`).
    // Its alignment drives whether whole-element stores can vectorize.
    let align = args.types().next().and_then(|elem_ty| {
        tcx.layout_of(TypingEnv::fully_monomorphized().as_query_input(elem_ty))
            .ok()
            .map(|layout| layout.align.abi.bytes())
    });

    Some(ArgAttrs {
        // Exclusive access over the whole range (the `from_raw_parts` contract),
        // encoded like rustc does for a unique `&mut` borrow.
        noalias: true,
        // It is the write channel.
        readonly: false,
        // The contract requires a valid, non-null, well-defined pointer.
        nonnull: true,
        noundef: true,
        // `len` is a runtime value, so there is no static `dereferenceable` size.
        dereferenceable: None,
        align,
        // A pointer carries no integer sign/zero extension.
        ext: ArgExt::None,
    })
}

/// Decode a single argument's [`ArgAttributes`] into the neutral struct.
///
/// This is a faithful pass-through of every attribute rustc computed, including
/// ones that aren't pointer-specific: `noundef` on a scalar, and `signext`/
/// `zeroext` on a narrow integer. Their applicability is enforced later, where
/// the lowered parameter type is known: `mir-lower`'s per-attribute gate emits
/// each token only on a parameter class it is valid on (e.g. `signext` only on
/// integers, the `noalias`/`align` family only on pointers).
///
/// Returns `None` only when *nothing at all* is set, so the transport layer can
/// skip the parameter.
fn decode(attrs: &ArgAttributes) -> Option<ArgAttrs> {
    let regular = attrs.regular;
    // `pointee_size` is `dereferenceable_or_null` semantics; zero means the
    // pointee size is unknown (e.g. the data pointer of a slice — unsized), so
    // we omit `dereferenceable` there and keep only `align`.
    let pointee_bytes = attrs.pointee_size.bytes();

    let decoded = ArgAttrs {
        noalias: regular.contains(ArgAttribute::NoAlias),
        readonly: regular.contains(ArgAttribute::ReadOnly),
        nonnull: regular.contains(ArgAttribute::NonNull),
        noundef: regular.contains(ArgAttribute::NoUndef),
        dereferenceable: (pointee_bytes != 0).then_some(pointee_bytes),
        align: attrs.pointee_align.map(|align| align.bytes()),
        // Integer sign/zero extension for narrow ints at a call boundary.
        ext: match attrs.arg_ext {
            ArgExtension::None => ArgExt::None,
            ArgExtension::Sext => ArgExt::Sign,
            ArgExtension::Zext => ArgExt::Zero,
        },
    };

    // Collapse "nothing to emit" to `None` so the transport layer can skip it.
    (!decoded.is_empty()).then_some(decoded)
}
