/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for plain (non-atomic) `StoreOp` / `LoadOp` on aggregate
//! types: they emit `align N`, and that N must be the type's **true ABI
//! alignment** — never a size-derived over-estimate.
//!
//! `align N` on an LLVM load/store is a promise that the address is N-byte
//! aligned; NVPTX then emits `ld/st.global.v{2,4}`, which fault on misaligned
//! addresses. So the alignment the exporter stamps must never exceed what the
//! type actually guarantees:
//!
//! * Array `[N x T]` has ABI alignment `align_of(T)` (Rust & LLVM rule), NOT
//!   the size-rounded `N * align_of(T)`. `[4 x float]` is 4-aligned, not 16.
//!   (Stamping 16 here is the bug these tests now guard against: it only
//!   "works" when the allocator happens to 16-align the buffer.)
//! * Vector `<N x T>` is genuinely width-aligned, so `<4 x float>` is 16. This
//!   is the sound way to get a 16-byte vector access.
//!
//! To get a 16-aligned aggregate access soundly, route data through a vector
//! type or a genuinely over-aligned (`repr(align(16))`) type whose real
//! alignment is threaded onto the op — not a bare `[f32; 4]`.

use dialect_llvm::{
    export::export_module_to_string,
    ops::{FuncOp, LoadOp, ReturnOp, StoreOp},
    types::{ArrayType, FuncType, PointerType, VectorType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        ops::ModuleOp,
        types::{FP32Type, FP64Type, IntegerType, Signedness},
    },
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
};

/// Build a tiny module containing exactly one function:
///
/// ```llvm
/// define void @store_test(<value_type> %v, ptr %p) {
///   store <value_type> %v, ptr %p
///   ret void
/// }
/// ```
fn build_store_module(value_ty_factory: impl FnOnce(&mut Context) -> pliron::context::Ptr<pliron::r#type::TypeObj>) -> String {
    let mut ctx = Context::new();
    dialect_llvm::register(&mut ctx);
    let value_ty = value_ty_factory(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let existing = {
            let region = module_region.deref(&ctx);
            region.iter(&ctx).next()
        };
        if let Some(block) = existing {
            block
        } else {
            let block = BasicBlock::new(&mut ctx, None, vec![]);
            block.insert_at_back(module_region, &ctx);
            block
        }
    };

    let void_ty = VoidType::get(&mut ctx);
    let ptr_ty = PointerType::get_generic(&mut ctx);
    let func_ty = FuncType::get(
        &mut ctx,
        void_ty.to_ptr(),
        vec![value_ty, ptr_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "store_test".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let value_arg = entry.deref(&ctx).get_argument(0);
    let ptr_arg = entry.deref(&ctx).get_argument(1);

    StoreOp::new(&mut ctx, value_arg, ptr_arg)
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    export_module_to_string(&ctx, &module).expect("export succeeds")
}

/// Build a tiny module containing exactly one function:
///
/// ```llvm
/// define <value_type> @load_test(ptr %p) {
///   %v = load <value_type>, ptr %p
///   ret <value_type> %v
/// }
/// ```
fn build_load_module(value_ty_factory: impl FnOnce(&mut Context) -> pliron::context::Ptr<pliron::r#type::TypeObj>) -> String {
    let mut ctx = Context::new();
    dialect_llvm::register(&mut ctx);
    let value_ty = value_ty_factory(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let existing = {
            let region = module_region.deref(&ctx);
            region.iter(&ctx).next()
        };
        if let Some(block) = existing {
            block
        } else {
            let block = BasicBlock::new(&mut ctx, None, vec![]);
            block.insert_at_back(module_region, &ctx);
            block
        }
    };

    let ptr_ty = PointerType::get_generic(&mut ctx);
    let func_ty = FuncType::get(&mut ctx, value_ty, vec![ptr_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "load_test".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let ptr_arg = entry.deref(&ctx).get_argument(0);

    let load = LoadOp::new(&mut ctx, ptr_arg, value_ty);
    load.get_operation().insert_at_back(entry, &ctx);
    let load_result = load.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, Some(load_result))
        .get_operation()
        .insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    export_module_to_string(&ctx, &module).expect("export succeeds")
}

/// Find the line in `ir` matching `keyword` (e.g. "store" or "load"). Panics if
/// not found or if multiple match (the test modules only have one each).
fn find_op_line<'a>(ir: &'a str, keyword: &str) -> &'a str {
    let candidates: Vec<&str> = ir
        .lines()
        .filter(|l| {
            let trim = l.trim_start();
            // start-of-instruction match for "store"/"load"/"%name = load"
            trim.starts_with(keyword) || trim.contains(&format!(" {keyword} "))
        })
        // Ignore `store atomic` / `load atomic` — those already emit align.
        .filter(|l| !l.contains("atomic"))
        .collect();
    assert_eq!(
        candidates.len(),
        1,
        "expected exactly one `{keyword}` line, got {}:\n{ir}",
        candidates.len()
    );
    candidates[0]
}

// ---------------------------------------------------------------------------
// Array stores/loads: `[N x T]`
// ---------------------------------------------------------------------------

// An array's ABI alignment is its ELEMENT alignment, regardless of length.
// Stamping the size (16 for `[4 x float]`) would be a false promise.

#[test]
fn store_array_f32_4_emits_align_4() {
    let ir = build_store_module(|ctx| {
        let f32_ty = FP32Type::get(ctx);
        ArrayType::get(ctx, f32_ty.into(), 4).into()
    });
    let store_line = find_op_line(&ir, "store");
    assert!(
        store_line.contains("align 4"),
        "store of `[4 x float]` must emit `align 4` (element align), not the \
         size-derived 16, got:\n{store_line}\n\nfull IR:\n{ir}"
    );
}

#[test]
fn load_array_f32_4_emits_align_4() {
    let ir = build_load_module(|ctx| {
        let f32_ty = FP32Type::get(ctx);
        ArrayType::get(ctx, f32_ty.into(), 4).into()
    });
    let load_line = find_op_line(&ir, "load");
    assert!(
        load_line.contains("align 4"),
        "load of `[4 x float]` must emit `align 4` (element align), not 16, \
         got:\n{load_line}\n\nfull IR:\n{ir}"
    );
}

#[test]
fn store_array_f32_2_emits_align_4() {
    let ir = build_store_module(|ctx| {
        let f32_ty = FP32Type::get(ctx);
        ArrayType::get(ctx, f32_ty.into(), 2).into()
    });
    let store_line = find_op_line(&ir, "store");
    assert!(
        store_line.contains("align 4"),
        "store of `[2 x float]` must emit `align 4` (element align), not 8, \
         got:\n{store_line}\n\nfull IR:\n{ir}"
    );
}

#[test]
fn store_array_f64_2_emits_align_8() {
    let ir = build_store_module(|ctx| {
        let f64_ty = FP64Type::get(ctx);
        ArrayType::get(ctx, f64_ty.into(), 2).into()
    });
    let store_line = find_op_line(&ir, "store");
    assert!(
        store_line.contains("align 8"),
        "store of `[2 x double]` must emit `align 8` (element align), not 16, \
         got:\n{store_line}\n\nfull IR:\n{ir}"
    );
}

#[test]
fn store_array_u32_4_emits_align_4() {
    let ir = build_store_module(|ctx| {
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        ArrayType::get(ctx, u32_ty.into(), 4).into()
    });
    let store_line = find_op_line(&ir, "store");
    assert!(
        store_line.contains("align 4"),
        "store of `[4 x i32]` must emit `align 4` (element align), not 16, \
         got:\n{store_line}\n\nfull IR:\n{ir}"
    );
}

// ---------------------------------------------------------------------------
// Vector stores/loads: `<N x T>` (LLVM vector type)
// ---------------------------------------------------------------------------

#[test]
fn store_vector_f32_4_emits_align_16() {
    let ir = build_store_module(|ctx| {
        let f32_ty = FP32Type::get(ctx);
        VectorType::get(ctx, f32_ty.into(), 4).into()
    });
    let store_line = find_op_line(&ir, "store");
    assert!(
        store_line.contains("align 16"),
        "store of `<4 x float>` must emit `align 16`, got:\n{store_line}\n\nfull IR:\n{ir}"
    );
}

// ---------------------------------------------------------------------------
// Scalar regression: don't accidentally change scalar alignment.
// ---------------------------------------------------------------------------

#[test]
fn store_scalar_f32_emits_align_4() {
    let ir = build_store_module(|ctx| FP32Type::get(ctx).into());
    let store_line = find_op_line(&ir, "store");
    assert!(
        store_line.contains("align 4"),
        "store of `float` must emit `align 4`, got:\n{store_line}\n\nfull IR:\n{ir}"
    );
}

#[test]
fn store_scalar_i32_emits_align_4() {
    let ir = build_store_module(|ctx| {
        IntegerType::get(ctx, 32, Signedness::Signless).into()
    });
    let store_line = find_op_line(&ir, "store");
    assert!(
        store_line.contains("align 4"),
        "store of `i32` must emit `align 4`, got:\n{store_line}\n\nfull IR:\n{ir}"
    );
}
