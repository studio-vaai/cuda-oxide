/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::export_module_to_string,
    ops::{AsmKind, CallOp, FuncOp, InlineAsmOp, InlineAsmOpExt, ReturnOp},
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    builtin::{op_interfaces::CallOpCallable, ops::ModuleOp},
    context::Context,
    op::Op,
};

use crate::common::module_top_block;

fn function(ctx: &mut Context, module: &ModuleOp, name: &str, callees: &[&str], defined: bool) {
    let ty = FuncType::get(ctx, VoidType::get(ctx).into(), vec![], false);
    let func = FuncOp::new(ctx, name.try_into().unwrap(), ty);
    if defined {
        let entry = func.get_or_create_entry_block(ctx);
        for callee in callees {
            CallOp::new(
                ctx,
                CallOpCallable::Direct((*callee).try_into().unwrap()),
                ty,
                vec![],
            )
            .get_operation()
            .insert_at_back(entry, ctx);
        }
        ReturnOp::new(ctx, None)
            .get_operation()
            .insert_at_back(entry, ctx);
    }
    let block = module_top_block(ctx, module);
    func.get_operation().insert_at_back(block, ctx);
}

fn definition_is_convergent(ir: &str, name: &str) -> bool {
    ir.lines()
        .find(|line| line.starts_with(&format!("define void @{name}(")))
        .unwrap_or_else(|| panic!("missing definition {name}:\n{ir}"))
        .contains("#0")
}

#[test]
fn intrinsic_seeds_containing_function_and_entire_wrapper_chain() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "convergence_chain".try_into().unwrap());
    let inner = reserved_oxide_symbols::device_symbol("inner");
    // Emit callers before their callees to exercise the prepass rather than
    // relying on textual order. Device prefix normalization must also agree.
    function(&mut ctx, &module, "outer", &[&inner], true);
    function(&mut ctx, &module, &inner, &["llvm_nvvm_barrier0"], true);
    function(&mut ctx, &module, "arithmetic", &[], true);
    function(&mut ctx, &module, "mixed", &[&inner, "arithmetic"], true);
    function(
        &mut ctx,
        &module,
        "arithmetic_wrapper",
        &["arithmetic"],
        true,
    );
    function(&mut ctx, &module, "llvm_nvvm_barrier0", &[], false);
    let ir = export_module_to_string(&ctx, &module).unwrap();
    for name in ["outer", "inner", "mixed"] {
        assert!(definition_is_convergent(&ir, name), "{ir}");
    }
    for name in ["arithmetic", "arithmetic_wrapper"] {
        assert!(!definition_is_convergent(&ir, name), "{ir}");
    }
    assert!(ir.contains("declare void @llvm.nvvm.barrier0() #0"), "{ir}");
    assert!(ir.contains("call void @llvm.nvvm.barrier0() #0"), "{ir}");
    assert!(ir.contains("call void @inner() #0"), "{ir}");
    assert!(ir.contains("call void @arithmetic()\n"), "{ir}");
}

#[test]
fn convergence_propagates_through_recursive_components_but_not_pure_cycles() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "recursive_convergence".try_into().unwrap());
    function(&mut ctx, &module, "a", &["b"], true);
    function(&mut ctx, &module, "b", &["a", "llvm_nvvm_barrier0"], true);
    function(&mut ctx, &module, "pure_a", &["pure_b"], true);
    function(&mut ctx, &module, "pure_b", &["pure_a"], true);
    function(&mut ctx, &module, "llvm_nvvm_barrier0", &[], false);
    let ir = export_module_to_string(&ctx, &module).unwrap();
    for name in ["a", "b"] {
        assert!(definition_is_convergent(&ir, name), "{ir}");
    }
    for name in ["pure_a", "pure_b"] {
        assert!(!definition_is_convergent(&ir, name), "{ir}");
    }
}

#[test]
fn inline_asm_convergence_is_independent_of_memory_effects() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "asm_convergence".try_into().unwrap());
    let ty = FuncType::get(&ctx, VoidType::get(&ctx).into(), vec![], false);
    for (name, kind, convergent) in [
        ("collective", AsmKind::Convergent, true),
        ("collective_pure", AsmKind::ConvergentPure, true),
        ("effect", AsmKind::SideEffect, false),
        ("register", AsmKind::Pure, false),
    ] {
        let func = FuncOp::new(&mut ctx, name.try_into().unwrap(), ty);
        let entry = func.get_or_create_entry_block(&mut ctx);
        let void = VoidType::get(&ctx);
        InlineAsmOp::build(&mut ctx, void.into(), vec![], "nop;", "", kind)
            .get_operation()
            .insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let block = module_top_block(&mut ctx, &module);
        func.get_operation().insert_at_back(block, &ctx);
        function(&mut ctx, &module, &format!("{name}_wrapper"), &[name], true);
        let ir = export_module_to_string(&ctx, &module).unwrap();
        assert_eq!(definition_is_convergent(&ir, name), convergent, "{ir}");
        assert_eq!(
            definition_is_convergent(&ir, &format!("{name}_wrapper")),
            convergent,
            "{ir}"
        );
    }
}

#[test]
fn user_inline_asm_flag_propagates_to_wrappers() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "user_asm_convergence".try_into().unwrap());
    let ty = FuncType::get(&ctx, VoidType::get(&ctx).into(), vec![], false);
    for (name, convergent) in [("collective", true), ("ordinary", false)] {
        let func = FuncOp::new(&mut ctx, name.try_into().unwrap(), ty);
        let entry = func.get_or_create_entry_block(&mut ctx);
        let void = VoidType::get(&ctx);
        let asm = InlineAsmOp::new(&mut ctx, void.into(), vec![], "nop;", "", convergent);
        llvm_export::ops::set_inline_asm_sideeffect(&mut ctx, asm.get_operation(), false);
        asm.get_operation().insert_at_back(entry, &ctx);
        ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(entry, &ctx);
        let block = module_top_block(&mut ctx, &module);
        func.get_operation().insert_at_back(block, &ctx);
        function(&mut ctx, &module, &format!("{name}_wrapper"), &[name], true);
    }
    let ir = export_module_to_string(&ctx, &module).unwrap();
    for (name, convergent) in [("collective", true), ("ordinary", false)] {
        assert_eq!(definition_is_convergent(&ir, name), convergent, "{ir}");
        assert_eq!(
            definition_is_convergent(&ir, &format!("{name}_wrapper")),
            convergent,
            "{ir}"
        );
    }
}

#[test]
fn opaque_external_and_indirect_calls_remain_conservative() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "opaque_convergence".try_into().unwrap());
    function(&mut ctx, &module, "external_wrapper", &["opaque"], true);
    function(&mut ctx, &module, "opaque", &[], false);
    function(
        &mut ctx,
        &module,
        "intrinsic_wrapper",
        &["llvm_nvvm_future_collective"],
        true,
    );
    function(&mut ctx, &module, "llvm_nvvm_future_collective", &[], false);
    let ptr = PointerType::get(&ctx, 0);
    let callee_ty = FuncType::get(&ctx, VoidType::get(&ctx).into(), vec![], false);
    let ty = FuncType::get(&ctx, VoidType::get(&ctx).into(), vec![ptr.into()], false);
    let func = FuncOp::new(&mut ctx, "indirect_wrapper".try_into().unwrap(), ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let callee = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Indirect(callee),
        callee_ty,
        vec![],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    let block = module_top_block(&mut ctx, &module);
    func.get_operation().insert_at_back(block, &ctx);
    let ir = export_module_to_string(&ctx, &module).unwrap();
    for name in ["external_wrapper", "intrinsic_wrapper", "indirect_wrapper"] {
        assert!(definition_is_convergent(&ir, name), "{ir}");
    }
    assert!(ir.contains("call void @opaque() #0"), "{ir}");
    assert!(ir.contains("call void %v0() #0"), "{ir}");
}

#[test]
fn ordinary_intrinsic_calls_and_wrappers_need_no_convergence() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "ordinary_intrinsic".try_into().unwrap());
    function(&mut ctx, &module, "trap_wrapper", &["llvm_trap"], true);
    function(&mut ctx, &module, "llvm_trap", &[], false);
    let ir = export_module_to_string(&ctx, &module).unwrap();
    assert!(!definition_is_convergent(&ir, "trap_wrapper"), "{ir}");
    assert!(ir.contains("call void @llvm.trap()\n"), "{ir}");
    assert!(!ir.contains("attributes #0"), "{ir}");
}

#[test]
fn math_intrinsic_wrappers_are_nonconvergent() {
    use pliron::builtin::types::FP32Type;
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "math_convergence".try_into().unwrap());
    let float = FP32Type::get(&ctx);
    let ty = FuncType::get(&ctx, float.into(), vec![float.into()], false);
    let decl = FuncOp::new(&mut ctx, "llvm_sqrt_f32".try_into().unwrap(), ty);
    let func = FuncOp::new(&mut ctx, "sqrt_wrapper".try_into().unwrap(), ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let input = entry.deref(&ctx).get_argument(0);
    let call = CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("llvm_sqrt_f32".try_into().unwrap()),
        ty,
        vec![input],
    );
    let result = call.get_operation().deref(&ctx).get_result(0);
    call.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, Some(result))
        .get_operation()
        .insert_at_back(entry, &ctx);
    let block = module_top_block(&mut ctx, &module);
    func.get_operation().insert_at_back(block, &ctx);
    decl.get_operation().insert_at_back(block, &ctx);
    let ir = export_module_to_string(&ctx, &module).unwrap();
    assert!(
        ir.contains("define float @sqrt_wrapper(float %v0) {"),
        "{ir}"
    );
    assert!(!ir.contains("#0"), "{ir}");
}

#[test]
fn external_purity_does_not_discard_convergence_protection() {
    use llvm_export::export::{
        DeviceExternAttrs, DeviceExternDecl, DeviceExternType, NvvmExportConfig, NvvmIrDialect,
        export_module_with_externs,
    };
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "external_convergence".try_into().unwrap());
    function(&mut ctx, &module, "external_wrapper", &["collective"], true);
    function(&mut ctx, &module, "collective", &[], false);
    let externs = [DeviceExternDecl {
        export_name: "collective".to_string(),
        param_types: vec![],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs {
            is_convergent: true,
            is_pure: true,
            ..Default::default()
        },
    }];
    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .unwrap();
    assert!(definition_is_convergent(&ir, "external_wrapper"), "{ir}");
    assert!(ir.contains("call void @collective() #0"), "{ir}");
}
