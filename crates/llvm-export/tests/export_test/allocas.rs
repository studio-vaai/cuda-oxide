/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::{NvvmExportConfig, NvvmIrDialect, export_module_to_string_with_config},
    ops::{
        AllocaOp, BrOp, CondBrOp, ConstantOp, FuncOp, GepIndex, GetElementPtrOp, LoadOp, ReturnOp,
        StoreOp,
    },
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IntegerAttr,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    utils::apint::APInt,
    value::Value,
};
use std::num::NonZero;

use crate::common::module_top_block;

fn one(ctx: &mut Context, block: Ptr<BasicBlock>) -> Value {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let op = ConstantOp::new(ctx, attr.into());
    op.get_operation().insert_at_back(block, ctx);
    op.get_operation().deref(ctx).get_result(0)
}

fn entry_and_body(ir: &str) -> (&str, &str) {
    let target = ir
        .lines()
        .find_map(|line| line.trim().strip_prefix("br label %"))
        .expect("entry branches to the body");
    ir.split_once(&format!("{target}:\n"))
        .expect("body label exists")
}

#[test]
fn fixed_slot_precedes_body_phis_but_initialization_stays_in_body() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "entry_alloca".try_into().unwrap());
    let top = module_top_block(&mut ctx, &module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ty = FuncType::get(&ctx, i32_ty.into(), vec![i32_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "scratch".try_into().unwrap(), ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let input = entry.deref(&ctx).get_argument(0);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let body = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    body.insert_at_back(region, &ctx);
    BrOp::new(&mut ctx, body, vec![input])
        .get_operation()
        .insert_at_back(entry, &ctx);
    let value = body.deref(&ctx).get_argument(0);
    let count = one(&mut ctx, body);
    let slot = AllocaOp::new(&mut ctx, i32_ty.into(), count);
    llvm_export::ops::set_op_alignment(&mut ctx, slot.get_operation(), 4);
    slot.get_operation().insert_at_back(body, &ctx);
    let pointer = slot.get_operation().deref(&ctx).get_result(0);
    StoreOp::new(&mut ctx, value, pointer)
        .get_operation()
        .insert_at_back(body, &ctx);
    let load = LoadOp::new(&mut ctx, pointer, i32_ty.into());
    let result = load.get_operation().deref(&ctx).get_result(0);
    load.get_operation().insert_at_back(body, &ctx);
    ReturnOp::new(&mut ctx, Some(result))
        .get_operation()
        .insert_at_back(body, &ctx);
    func.get_operation().insert_at_back(top, &ctx);
    module.get_operation().deref(&ctx).verify(&ctx).unwrap();

    for dialect in [NvvmIrDialect::Modern, NvvmIrDialect::LegacyLlvm7] {
        let ir =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .unwrap();
        let (entry, body) = entry_and_body(&ir);
        assert_eq!(ir.matches(" = alloca ").count(), 1, "{ir}");
        assert!(entry.contains(" = alloca i32, align 4"), "{ir}");
        assert!(
            body.trim_start().starts_with('%') && body.contains(" = phi i32"),
            "{ir}"
        );
        assert!(
            body.contains("store i32") && body.contains("load i32"),
            "{ir}"
        );
        assert!(!entry.contains("store i32"), "{ir}");
    }
}

#[test]
fn dynamic_alloca_count_stays_in_its_original_block() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "dynamic_alloca".try_into().unwrap());
    let top = module_top_block(&mut ctx, &module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let pointer_ty = PointerType::get(&ctx, 0);
    let ty = FuncType::get(&ctx, pointer_ty.into(), vec![i32_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "dynamic".try_into().unwrap(), ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let count = entry.deref(&ctx).get_argument(0);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let body = BasicBlock::new(&mut ctx, None, vec![]);
    body.insert_at_back(region, &ctx);
    BrOp::new(&mut ctx, body, vec![])
        .get_operation()
        .insert_at_back(entry, &ctx);
    let slot = AllocaOp::new(&mut ctx, i32_ty.into(), count);
    let pointer = slot.get_operation().deref(&ctx).get_result(0);
    slot.get_operation().insert_at_back(body, &ctx);
    ReturnOp::new(&mut ctx, Some(pointer))
        .get_operation()
        .insert_at_back(body, &ctx);
    func.get_operation().insert_at_back(top, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .unwrap();
    let (entry, body) = entry_and_body(&ir);
    assert!(!entry.contains(" = alloca "), "{ir}");
    assert!(body.contains(" = alloca i32, i32 %0"), "{ir}");
}

#[test]
fn loop_scratch_hoists_only_when_its_address_does_not_escape() {
    for escapes in [false, true] {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "loop_alloca".try_into().unwrap());
        let top = module_top_block(&mut ctx, &module);
        let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
        let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
        let pointer_ty = PointerType::get(&ctx, 0);
        let ty = FuncType::get(
            &ctx,
            VoidType::get(&ctx).into(),
            vec![i1_ty.into(), pointer_ty.into()],
            false,
        );
        let func = FuncOp::new(&mut ctx, "loop_scratch".try_into().unwrap(), ty);
        let entry = func.get_or_create_entry_block(&mut ctx);
        let repeat = entry.deref(&ctx).get_argument(0);
        let escape_destination = entry.deref(&ctx).get_argument(1);
        let region = func.get_operation().deref(&ctx).get_region(0);
        let body = BasicBlock::new(&mut ctx, None, vec![]);
        let exit = BasicBlock::new(&mut ctx, None, vec![]);
        body.insert_at_back(region, &ctx);
        exit.insert_at_back(region, &ctx);
        BrOp::new(&mut ctx, body, vec![])
            .get_operation()
            .insert_at_back(entry, &ctx);
        let count = one(&mut ctx, body);
        let slot = AllocaOp::new(&mut ctx, i32_ty.into(), count);
        let pointer = slot.get_operation().deref(&ctx).get_result(0);
        slot.get_operation().insert_at_back(body, &ctx);
        let gep = GetElementPtrOp::new(
            &mut ctx,
            pointer,
            vec![GepIndex::Constant(0)],
            i32_ty.into(),
        );
        let view = gep.get_operation().deref(&ctx).get_result(0);
        gep.get_operation().insert_at_back(body, &ctx);
        StoreOp::new(&mut ctx, count, view)
            .get_operation()
            .insert_at_back(body, &ctx);
        LoadOp::new(&mut ctx, view, i32_ty.into())
            .get_operation()
            .insert_at_back(body, &ctx);
        if escapes {
            StoreOp::new(&mut ctx, view, escape_destination)
                .get_operation()
                .insert_at_back(body, &ctx);
        }
        CondBrOp::new(&mut ctx, repeat, body, vec![], exit, vec![])
            .get_operation()
            .insert_at_back(body, &ctx);
        ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(exit, &ctx);
        func.get_operation().insert_at_back(top, &ctx);
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();

        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::Modern),
        )
        .unwrap();
        let (entry, body) = entry_and_body(&ir);
        assert_eq!(entry.contains(" = alloca "), !escapes, "{ir}");
        assert_eq!(body.contains(" = alloca "), escapes, "{ir}");
        assert!(
            body.contains("getelementptr") && body.contains("store i32"),
            "{ir}"
        );
    }
}
