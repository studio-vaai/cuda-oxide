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

/// Build a named declaration or a body with the requested direct calls.
/// The configurable insertion order exercises classification before textual emission.
fn function(
    ctx: &mut Context,
    module: &ModuleOp,
    name: &str,
    callees: &[&str],
    defined: bool,
) -> FuncOp {
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
    func
}

/// Read convergence on a definition emitted by this exporter.
/// Requiring the definition prevents missing output from passing an absence assertion.
fn definition_is_convergent(ir: &str, name: &str) -> bool {
    ir.lines()
        .find(|line| line.starts_with(&format!("define void @{name}(")))
        .unwrap_or_else(|| panic!("missing definition {name}:\n{ir}"))
        .split_whitespace()
        .any(|word| word == "#0" || word == "convergent")
}

/// A barrier seeds every wrapper even when callers precede their callees.
/// The same module keeps arithmetic-only helpers and their calls unrestricted.
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

/// A collective propagates throughout its recursive strongly connected component.
/// A separate cycle with no collective stays free of convergence restrictions.
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

/// Exercise all four combinations of side effects and inline-asm convergence.
/// Only convergence seeds protection in the containing function and its caller.
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

/// User-supplied convergence reaches the containing function and outer wrapper.
/// The ordinary-asm control stays unrestricted despite sharing the same module.
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
        llvm_export::ops::set_inline_asm_convergent(&mut ctx, asm.get_operation(), convergent);
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

/// Unknown external declarations and indirect targets cannot prove independence.
/// Their callers retain protection, including unrecognized future intrinsic names.
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

/// An ordinary trap intrinsic does not seed convergence in its callers.
/// A module without collective operations needs no convergence attribute group.
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

/// A scalar square-root intrinsic leaves its arithmetic wrapper unrestricted.
/// Check the complete signature so redundant convergence cannot pass unnoticed.
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

/// A pure external operation can still carry collective semantics.
/// Its marked declaration seeds protection through the containing wrapper.
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
    for dialect in [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7] {
        let ir =
            export_module_with_externs(&ctx, &module, &externs, &NvvmExportConfig::new(dialect))
                .unwrap();
        assert!(definition_is_convergent(&ir, "external_wrapper"), "{ir}");
        assert!(ir.contains("declare void @collective() #0"), "{ir}");
        assert!(ir.contains("call void @collective() #0"), "{ir}");
    }
}

/// Render each native entry point in both modern and legacy NVVM dialects.
/// Reusing the same IR also exercises the analysis's read-only contract.
fn export_variants(ctx: &Context, module: &ModuleOp) -> Vec<String> {
    use llvm_export::export::{
        DeviceExternDecl, NvvmExportConfig, NvvmIrDialect, export_module_to_string_with_config,
        export_module_with_externs,
    };
    [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7]
        .into_iter()
        .flat_map(|dialect| {
            let config = NvvmExportConfig::new(dialect);
            [
                export_module_to_string_with_config(ctx, module, &config).unwrap(),
                export_module_with_externs(ctx, module, &[] as &[DeviceExternDecl], &config)
                    .unwrap(),
            ]
        })
        .collect()
}

/// Annotate a known call without modifying the target function's attributes.
/// The distinction is important: call-site requirements belong to that caller only.
fn annotate_first_call(ctx: &mut Context, func: FuncOp, value: llvm_export::ops::LlvmAttrValue) {
    use pliron::{linked_list::ContainsLinkedList, operation::Operation};
    let entry = func.get_or_create_entry_block(ctx);
    let call = entry
        .deref(ctx)
        .iter(ctx)
        .find_map(|op| Operation::get_op::<CallOp>(op, ctx))
        .unwrap();
    let mut attrs = llvm_export::ops::LlvmAttributesAttr::new();
    attrs.set("convergent", value);
    call.set_attr_llvm_call_attrs(ctx, attrs);
}

/// Explicit function attributes seed transitive callers in either export path.
/// A declaration of a normally non-convergent intrinsic must also respect an explicit mark.
#[test]
fn explicit_function_attributes_propagate_without_changing_input_ir() {
    use llvm_export::ops::{LlvmAttrValue, LlvmAttributesAttr};
    for defined in [true, false] {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "explicit_function".try_into().unwrap());
        let name = if defined { "leaf" } else { "llvm_trap" };
        function(&mut ctx, &module, "outer", &["middle"], true);
        let middle = function(&mut ctx, &module, "middle", &[name], true);
        let leaf = function(&mut ctx, &module, name, &[], defined);
        function(&mut ctx, &module, "control", &[], true);
        let mut attrs = LlvmAttributesAttr::new();
        attrs.set("convergent", LlvmAttrValue::Unit);
        attrs.set("noinline", LlvmAttrValue::Unit);
        leaf.set_attr_llvm_func_attrs(&ctx, attrs);
        for ir in export_variants(&ctx, &module) {
            for caller in ["outer", "middle"] {
                assert!(definition_is_convergent(&ir, caller), "{ir}");
            }
            assert!(!definition_is_convergent(&ir, "control"), "{ir}");
            let symbol = if defined { "leaf" } else { "llvm.trap" };
            let signature = ir
                .lines()
                .find(|line| {
                    (line.starts_with("define ") || line.starts_with("declare "))
                        && line.contains(&format!("@{symbol}("))
                })
                .unwrap();
            assert!(signature.contains("convergent"), "{ir}");
            assert!(signature.contains("noinline"), "{ir}");
            assert!(
                !signature.contains("#0"),
                "explicit convergence should not be duplicated: {ir}"
            );
            assert!(ir.contains(&format!("call void @{symbol}() #0")), "{ir}");
        }
        assert!(
            leaf.get_attr_llvm_func_attrs(&ctx)
                .unwrap()
                .has("convergent")
        );
        assert!(middle.get_attr_llvm_func_attrs(&ctx).is_none());
    }
}

/// A convergent call seeds its enclosing function and all transitive callers.
/// The callee and its unannotated sibling caller must remain non-convergent.
#[test]
fn explicit_call_attributes_propagate_without_tainting_other_calls() {
    use llvm_export::ops::LlvmAttrValue;
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "explicit_call".try_into().unwrap());
    function(&mut ctx, &module, "outer", &["middle"], true);
    let middle = function(&mut ctx, &module, "middle", &["leaf"], true);
    let leaf = function(&mut ctx, &module, "leaf", &[], true);
    function(&mut ctx, &module, "sibling", &["leaf"], true);
    annotate_first_call(&mut ctx, middle, LlvmAttrValue::Unit);
    for ir in export_variants(&ctx, &module) {
        for name in ["outer", "middle"] {
            assert!(definition_is_convergent(&ir, name), "{ir}");
        }
        for name in ["leaf", "sibling"] {
            assert!(!definition_is_convergent(&ir, name), "{ir}");
        }
        assert!(ir.contains("call void @leaf() convergent\n"), "{ir}");
        assert!(ir.contains("call void @leaf()\n"), "{ir}");
        assert!(!ir.contains("call void @leaf() convergent #0"), "{ir}");
    }
    assert!(leaf.get_attr_llvm_func_attrs(&ctx).is_none());
    assert!(middle.get_attr_llvm_func_attrs(&ctx).is_none());
}

/// Attribute composition preserves unrelated attributes for every convergence combination.
/// An explicit call annotation replaces the inferred group even for a collective target.
#[test]
fn call_attributes_merge_explicit_and_inferred_convergence() {
    use llvm_export::ops::{LlvmAttrValue, LlvmAttributesAttr};
    use pliron::{linked_list::ContainsLinkedList, operation::Operation};

    for inferred in [false, true] {
        for explicit in [false, true] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "merged_attributes".try_into().unwrap());
            function(&mut ctx, &module, "llvm_nvvm_barrier0", &[], false);
            let callees = if inferred {
                vec!["llvm_nvvm_barrier0"]
            } else {
                vec![]
            };
            function(&mut ctx, &module, "leaf", &callees, true);
            let caller = function(&mut ctx, &module, "caller", &["leaf"], true);
            let entry = caller.get_or_create_entry_block(&mut ctx);
            let call = entry
                .deref(&ctx)
                .iter(&ctx)
                .find_map(|op| Operation::get_op::<CallOp>(op, &ctx))
                .unwrap();
            let mut attrs = LlvmAttributesAttr::new();
            attrs.set("nounwind", LlvmAttrValue::Unit);
            if explicit {
                attrs.set("convergent", LlvmAttrValue::Unit);
            }
            call.set_attr_llvm_call_attrs(&ctx, attrs);

            for ir in export_variants(&ctx, &module) {
                let call = ir
                    .lines()
                    .find(|line| line.contains("call void @leaf()"))
                    .unwrap();
                let attributes: Vec<_> = call.split_whitespace().collect();
                assert!(attributes.contains(&"nounwind"), "{ir}");
                assert_eq!(attributes.contains(&"convergent"), explicit, "{ir}");
                assert_eq!(attributes.contains(&"#0"), inferred && !explicit, "{ir}");
                assert_eq!(
                    definition_is_convergent(&ir, "caller"),
                    inferred || explicit,
                    "{ir}"
                );
            }
        }
    }
}

/// Malformed explicit convergence must not be ignored during graph construction.
/// Both function and call attributes reject non-unit values before emission.
#[test]
fn malformed_explicit_convergence_is_rejected() {
    use llvm_export::ops::{LlvmAttrValue, LlvmAttributesAttr};
    for on_call in [false, true] {
        for value in [LlvmAttrValue::Int(1), LlvmAttrValue::Str("true".into())] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "invalid_convergence".try_into().unwrap());
            function(&mut ctx, &module, "leaf", &[], true);
            let caller = function(&mut ctx, &module, "caller", &["leaf"], true);
            if on_call {
                annotate_first_call(&mut ctx, caller, value);
            } else {
                let mut attrs = LlvmAttributesAttr::new();
                attrs.set("convergent", value);
                caller.set_attr_llvm_func_attrs(&ctx, attrs);
            }
            let error = export_module_to_string(&ctx, &module).unwrap_err();
            assert_eq!(
                error,
                "LLVM function/call `convergent` requires a unit attribute"
            );
        }
    }
}

/// Typed primitive facts replace name classification in every export path.
/// Explicit LLVM convergence still wins over a proven non-convergent primitive.
#[test]
fn intrinsic_facts_propagate_and_explicit_convergence_wins() {
    use llvm_export::{
        intrinsics::{IntrinsicConvergenceAttr, set_intrinsic_convergence},
        ops::{LlvmAttrValue, LlvmAttributesAttr},
    };
    for fact in [None, Some(false), Some(true)] {
        for explicit in [false, true] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "primitive_facts".try_into().unwrap());
            function(&mut ctx, &module, "outer", &["middle"], true);
            function(
                &mut ctx,
                &module,
                "middle",
                &["llvm_nvvm_read_ptx_sreg_tid_x"],
                true,
            );
            let declaration = function(
                &mut ctx,
                &module,
                "llvm_nvvm_read_ptx_sreg_tid_x",
                &[],
                false,
            );
            if let Some(convergent) = fact {
                set_intrinsic_convergence(
                    &mut ctx,
                    declaration.get_operation(),
                    IntrinsicConvergenceAttr { convergent },
                )
                .unwrap();
            }
            if explicit {
                let mut attrs = LlvmAttributesAttr::new();
                attrs.set("convergent", LlvmAttrValue::Unit);
                declaration.set_attr_llvm_func_attrs(&ctx, attrs);
            }
            for ir in export_variants(&ctx, &module) {
                let convergent = explicit || fact.unwrap_or(true);
                for name in ["outer", "middle"] {
                    assert_eq!(definition_is_convergent(&ir, name), convergent, "{ir}");
                }
                let call = ir
                    .lines()
                    .find(|line| line.contains("call void @llvm.nvvm.read.ptx.sreg.tid.x()"))
                    .unwrap();
                assert_eq!(call.contains("#0"), convergent, "{ir}");
                assert!(
                    !ir.contains("intrinsic_convergence"),
                    "internal fact leaked into LLVM: {ir}"
                );
            }
        }
    }
}

/// Declaration reuse cannot overwrite an earlier primitive fact in either order.
/// An identical fact is idempotent and remains available to the read-only analysis.
#[test]
fn conflicting_intrinsic_facts_are_rejected() {
    use llvm_export::intrinsics::{
        IntrinsicConvergenceAttr, intrinsic_convergence, set_intrinsic_convergence,
    };
    for first in [false, true] {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "fact_conflict".try_into().unwrap());
        let declaration = function(&mut ctx, &module, "primitive", &[], false).get_operation();
        let fact = IntrinsicConvergenceAttr { convergent: first };
        set_intrinsic_convergence(&mut ctx, declaration, fact).unwrap();
        set_intrinsic_convergence(&mut ctx, declaration, fact).unwrap();
        let error = set_intrinsic_convergence(
            &mut ctx,
            declaration,
            IntrinsicConvergenceAttr { convergent: !first },
        )
        .unwrap_err();
        assert!(error.contains("conflicting convergence facts"), "{error}");
        assert_eq!(
            intrinsic_convergence(&ctx, declaration).unwrap(),
            Some(fact)
        );
    }
}

/// Malformed or misplaced facts cannot be treated as a proof of non-convergence.
/// Function bodies always require graph analysis rather than a primitive override.
#[test]
fn invalid_intrinsic_fact_transport_is_rejected() {
    use llvm_export::intrinsics::{IntrinsicConvergenceAttr, set_intrinsic_convergence};
    use pliron::builtin::attributes::BoolAttr;
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_fact".try_into().unwrap());
    let declaration = function(&mut ctx, &module, "primitive", &[], false).get_operation();
    declaration.deref_mut(&ctx).attributes.set(
        "cuda_oxide_intrinsic_convergence".try_into().unwrap(),
        BoolAttr::new(false),
    );
    assert!(
        export_module_to_string(&ctx, &module)
            .unwrap_err()
            .contains("wrong attribute type")
    );
    declaration.deref_mut(&ctx).attributes.0.remove(
        &pliron::identifier::Identifier::try_new("cuda_oxide_intrinsic_convergence".into())
            .unwrap(),
    );
    let definition = function(&mut ctx, &module, "defined", &[], true).get_operation();
    let fact = IntrinsicConvergenceAttr { convergent: false };
    assert!(
        set_intrinsic_convergence(&mut ctx, definition, fact)
            .unwrap_err()
            .contains("function declaration")
    );
    definition
        .deref_mut(&ctx)
        .attributes
        .set("cuda_oxide_intrinsic_convergence".try_into().unwrap(), fact);
    assert!(
        export_module_to_string(&ctx, &module)
            .unwrap_err()
            .contains("function declaration")
    );
}
