/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end coverage for the `llvm-op-attributes` path: a frontend stamps
//! source-parameter [`LlvmParamAttrsAttr`] carriers on a `dialect-mir` func,
//! `mir-lower` remaps them onto the flattened LLVM parameters, and the exporter
//! renders them into the textual `.ll` signature.
//!
//! This mirrors what `mir-importer::run_pipeline` does at translation time,
//! using only public API so it exercises the real carrier round-trip rather
//! than the internal helpers.

use dialect_mir::ops as mir;
use dialect_mir::types::MirPtrType;
use llvm_export::export::export_module_to_string;
use llvm_export::{ARG_ATTRS_KEY, ArgAttrs, ArgExt, LlvmParamAttrsAttr};
use pliron::attribute::AttrObj;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::{TypeAttr, VecAttr};
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::identifier::Identifier;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;

/// Fresh context with every dialect the lowering + export need registered.
fn make_ctx() -> Context {
    let mut ctx = Context::new();
    // The LLVM dialect (and our `LlvmParamAttrsAttr`) auto-register on context
    // creation; the local dialects need an explicit register call.
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);
    ctx
}

/// Build a single-function module `attr_probe(ptr, i32)` with a void body and
/// stamp `src_attrs` (one per source parameter) as the frontend would, then
/// return the module. The caller lowers and exports it.
fn build_module(ctx: &mut Context, src_attrs: Vec<Option<ArgAttrs>>) -> ModuleOp {
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let ptr_ty: TypeHandle = MirPtrType::get_generic(ctx, i32_ty, true).into();
    let arg_tys = vec![ptr_ty, i32_ty];

    let module = ModuleOp::new(ctx, "attr_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_ty = FunctionType::get(ctx, arg_tys.clone(), vec![]);
    let func_op_ptr = Operation::new(
        ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(ctx, func_op_ptr, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(ctx, "attr_probe".try_into().unwrap());

    let region = func.get_operation().deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, arg_tys);
    entry.insert_at_back(region, ctx);

    // Void return so the body verifies.
    let ret = Operation::new(
        ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(entry, ctx);

    let module_block = module_ptr
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .unwrap();
    func.get_operation().insert_at_back(module_block, ctx);

    // Stamp source-parameter carriers, exactly as `mir-importer` does.
    stamp_source_arg_attrs(ctx, func_op_ptr, &src_attrs);

    module
}

/// Same shape as `mir_importer::pipeline::stamp_source_arg_attrs`, replicated
/// here so the test drives the public carrier type end to end.
fn stamp_source_arg_attrs(ctx: &mut Context, func_op: Ptr<Operation>, src: &[Option<ArgAttrs>]) {
    let key = Identifier::try_from(ARG_ATTRS_KEY).unwrap();
    let elems: Vec<AttrObj> = src
        .iter()
        .map(|slot| {
            slot.as_ref()
                .map(LlvmParamAttrsAttr::from)
                .unwrap_or_default()
                .into()
        })
        .collect();
    func_op
        .deref_mut(ctx)
        .attributes
        .set(key, VecAttr::new(elems));
}

/// Lower and export, returning the `define` line for `@attr_probe`.
fn lower_and_get_signature(ctx: &mut Context, module: &ModuleOp) -> String {
    mir_lower::lower_mir_to_llvm(ctx, module.get_operation()).expect("lowering succeeds");
    let ir = export_module_to_string(ctx, module).expect("export succeeds");
    ir.lines()
        .find(|line| line.contains("@attr_probe("))
        .unwrap_or_else(|| panic!("no @attr_probe signature in:\n{ir}"))
        .to_string()
}

#[test]
fn faithful_pointer_and_integer_attrs_reach_the_ll_signature() {
    let mut ctx = make_ctx();
    let module = build_module(
        &mut ctx,
        vec![
            // `&mut i32`-shaped pointer.
            Some(ArgAttrs {
                noalias: true,
                nonnull: true,
                noundef: true,
                dereferenceable: Some(16),
                align: Some(4),
                ..Default::default()
            }),
            // Narrow signed integer at the ABI boundary.
            Some(ArgAttrs {
                noundef: true,
                ext: ArgExt::Sign,
                ..Default::default()
            }),
        ],
    );
    let sig = lower_and_get_signature(&mut ctx, &module);

    // Pointer family lands on the `ptr` parameter, in canonical order.
    assert!(
        sig.contains("noalias nonnull noundef dereferenceable(16) align 4"),
        "pointer attrs missing from signature: {sig}"
    );
    // Integer extension + `noundef` land on the `i32` parameter; pointer-only
    // tokens do not leak onto it.
    assert!(
        sig.contains("signext noundef"),
        "integer attrs missing from signature: {sig}"
    );
    assert!(
        !sig.contains("signext noalias"),
        "pointer-only tokens leaked onto the integer parameter: {sig}"
    );
}

#[test]
fn a_function_without_attrs_stays_bare() {
    let mut ctx = make_ctx();
    // No source attributes at all: the func carries no carrier, and the
    // exporter must emit bare parameters (the pre-attributes behaviour).
    let module = build_module(&mut ctx, vec![None, None]);
    let sig = lower_and_get_signature(&mut ctx, &module);

    for token in ["noalias", "noundef", "signext", "align ", "dereferenceable"] {
        assert!(
            !sig.contains(token),
            "unexpected `{token}` on a function with no attributes: {sig}"
        );
    }
}

#[test]
fn empty_carriers_render_nothing() {
    let mut ctx = make_ctx();
    // Present-but-empty carriers (all fields false/zero) must render bare, so a
    // frontend that supplies a vector of empties does not change the output.
    let module = build_module(
        &mut ctx,
        vec![Some(ArgAttrs::default()), Some(ArgAttrs::default())],
    );
    let sig = lower_and_get_signature(&mut ctx, &module);
    assert!(
        !sig.contains("noalias") && !sig.contains("noundef") && !sig.contains("signext"),
        "empty carriers should render nothing: {sig}"
    );
}
