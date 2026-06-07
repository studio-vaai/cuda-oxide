/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

/// Phase 1 (Route A): the `FnAbi`-derived pointer-parameter attributes the
/// backend supplies (one typed [`llvm_export::ArgAttrs`] per *source*
/// parameter, handed to `mir-lower` via the driver map) must be remapped onto
/// the flattened LLVM parameters, rendered to their textual fragment at the
/// remap, and emitted into the textual IR by the exporter.
///
/// This is a *placement/mechanism* test — it injects the per-source-arg
/// attributes directly (the real rustc-derived values are validated end-to-end
/// by the `ptr_attributes` example). It models `fn k(out: &mut [f32], inp:
/// &[f32], a: f32)`: each slice flattens to `(ptr, len)`, so a source-arg
/// attribute must land on the data pointer only — never the length or the
/// scalar.
///
/// The attributes are the realistic ones rustc produces: `&mut [f32]` →
/// `noalias` (no `readonly`); `&[f32]` → `noalias readonly` (rustc marks an
/// immutable shared ref to `Freeze` data as both). The checked soundness
/// property is therefore `readonly` placement: it lands on the read-only input
/// and never on the written `&mut` output.
#[test]
fn fnabi_pointer_param_attrs_land_on_flattened_data_pointers() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirSliceType;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::{FP32Type, FunctionType};
    use pliron::identifier::Identifier;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "attrs_module".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    // Kernel signature: (&mut [f32], &[f32], f32).
    let f32_ty = FP32Type::get(&ctx);
    let slice_ty = MirSliceType::get(&mut ctx, f32_ty.into());
    let func_ty = FunctionType::get(
        &mut ctx,
        vec![slice_ty.into(), slice_ty.into(), f32_ty.into()],
        vec![],
    );

    let func_op = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(&mut ctx, func_op, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(&mut ctx, "k".try_into().unwrap());

    // Mark as a kernel so the slice flattening uses the kernel-boundary ABI.
    let insert_attr = |ctx: &mut Context, key: &str, value: &str| {
        let key: Identifier = key.try_into().unwrap();
        func_op
            .deref_mut(ctx)
            .attributes
            .0
            .insert(key, StringAttr::new(value.to_string()).into());
    };
    insert_attr(&mut ctx, "gpu_kernel", "true");

    // Body: one block whose arguments match the (un-flattened) MIR signature,
    // plus a void return. The lowerer builds the flattened entry block and the
    // reconstruction prologue around it.
    {
        let region = func_op.deref(&ctx).get_region(0);
        let block = BasicBlock::new(
            &mut ctx,
            None,
            vec![slice_ty.into(), slice_ty.into(), f32_ty.into()],
        );
        block.insert_at_back(region, &ctx);

        let ret_op = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, &ctx);
    }
    func_op.insert_at_back(module_block, &ctx);

    // Typed per-source-arg attributes, exactly as the backend derives from
    // rustc's FnAbi (keyed by the func's symbol name `k`):
    //   arg0 `out: &mut [f32]` -> noalias (unique borrow), align 4; NOT readonly
    //   arg1 `inp: &[f32]`     -> noalias + readonly (shared ref to Freeze), align 4
    //   arg2 `a: f32`          -> None (no pointer attributes)
    let mut arg_attrs = mir_lower::context::ArgAttrsMap::new();
    arg_attrs.insert(
        "k".to_string(),
        vec![
            Some(llvm_export::ArgAttrs {
                noalias: true,
                nonnull: true,
                noundef: true,
                align: Some(4),
                ..Default::default()
            }),
            Some(llvm_export::ArgAttrs {
                noalias: true,
                readonly: true,
                nonnull: true,
                noundef: true,
                align: Some(4),
                ..Default::default()
            }),
            None,
        ],
    );

    mir_lower::lower_mir_to_llvm_with_arg_attrs(&mut ctx, module_ptr, arg_attrs)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|e| anyhow::anyhow!("export failed: {e}"))?;

    // Isolate the kernel's `define` line (the parameter list lives there).
    let define_line = ir
        .lines()
        .find(|line| line.contains("@k(") && line.trim_start().starts_with("define"))
        .unwrap_or_else(|| panic!("no `define ... @k(` line in IR:\n{ir}"));

    // `out` (flattened param 0): unique `&mut` borrow -> noalias + align 4.
    assert!(
        define_line.contains("ptr noalias nonnull noundef align 4 %v0"),
        "expected `out` data pointer (%v0) to carry `noalias ... align 4`; got:\n{define_line}"
    );
    // `inp` (flattened param 2): shared ref to Freeze data -> noalias + readonly
    // + align 4.
    assert!(
        define_line.contains("ptr noalias readonly nonnull noundef align 4 %v2"),
        "expected `inp` data pointer (%v2) to carry `noalias readonly ... align 4`; got:\n{define_line}"
    );

    // Soundness property: `readonly` lands on the read-only `inp` only, never on
    // the written `&mut out` (marking the write channel readonly would
    // miscompile). After flattening, the `out` data pointer is %v0 and the `inp`
    // data pointer is %v2.
    assert_eq!(
        ir.matches("readonly").count(),
        1,
        "`readonly` must appear exactly once (on the shared `&[f32]` input only); IR:\n{ir}"
    );
    assert!(
        !define_line.contains("ptr noalias readonly nonnull noundef align 4 %v0"),
        "the unique `&mut out` (%v0) must NOT be `readonly`:\n{define_line}"
    );

    // Slice lengths and the scalar are passed through bare (no pointer attrs).
    // After flattening the params are: ptr, i64, ptr, i64, float.
    assert!(
        define_line.contains("i64 %v1") && define_line.contains("i64 %v3"),
        "slice length params must be present and separate from the data pointers:\n{define_line}"
    );
    assert!(
        define_line.contains("float %v4"),
        "scalar `a: f32` must be the last param with no pointer attributes:\n{define_line}"
    );
    // The scalar must not pick up any of the pointer attributes.
    assert!(
        !define_line.contains("float noalias")
            && !define_line.contains("float readonly")
            && !define_line.contains("float align"),
        "scalar param must carry no pointer attributes:\n{define_line}"
    );

    Ok(())
}

/// `DisjointSlice<T>` carries no `FnAbi` attributes (it wraps a raw `*mut T`),
/// so the backend synthesizes `noalias`/`align`/`nonnull`/`noundef` from its
/// `from_raw_parts` contract. Those fragments must flow through the same
/// flattening/emission path as ordinary slices and land on the output's data
/// pointer.
///
/// Models the canonical copy kernel `fn copy(input: &[f32], output:
/// DisjointSlice<f32>)`: the shared input gets `noalias readonly` (rustc marks
/// an immutable shared ref to `Freeze` data as both), while the exclusive
/// `DisjointSlice` output gets `noalias` + `align` but **not** `readonly` (it is
/// written through). The alignment is what lets whole-element stores vectorize.
#[test]
fn disjoint_slice_output_gets_noalias_and_align() -> Result<(), anyhow::Error> {
    use dialect_mir::types::{MirDisjointSliceType, MirSliceType};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::{FP32Type, FunctionType};
    use pliron::identifier::Identifier;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "disjoint_module".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    // Kernel signature: (&[f32], DisjointSlice<f32>).
    let f32_ty = FP32Type::get(&ctx);
    let in_slice = MirSliceType::get(&mut ctx, f32_ty.into());
    let out_slice = MirDisjointSliceType::get(&mut ctx, f32_ty.into());
    let func_ty = FunctionType::get(&mut ctx, vec![in_slice.into(), out_slice.into()], vec![]);

    let func_op = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(&mut ctx, func_op, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(&mut ctx, "copy".try_into().unwrap());

    let insert_attr = |ctx: &mut Context, key: &str, value: &str| {
        let key: Identifier = key.try_into().unwrap();
        func_op
            .deref_mut(ctx)
            .attributes
            .0
            .insert(key, StringAttr::new(value.to_string()).into());
    };
    insert_attr(&mut ctx, "gpu_kernel", "true");

    {
        let region = func_op.deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![in_slice.into(), out_slice.into()]);
        block.insert_at_back(region, &ctx);
        let ret_op = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, &ctx);
    }
    func_op.insert_at_back(module_block, &ctx);

    // Typed per-source-arg attributes, exactly as the backend derives (keyed by
    // the func's symbol name `copy`):
    //   arg0 `input: &[f32]`              -> noalias + readonly (shared ref to Freeze), align 4
    //   arg1 `output: DisjointSlice<f32>` -> noalias (synthesized contract), align 4; NOT readonly
    let mut arg_attrs = mir_lower::context::ArgAttrsMap::new();
    arg_attrs.insert(
        "copy".to_string(),
        vec![
            Some(llvm_export::ArgAttrs {
                noalias: true,
                readonly: true,
                nonnull: true,
                noundef: true,
                align: Some(4),
                ..Default::default()
            }),
            Some(llvm_export::ArgAttrs {
                noalias: true,
                nonnull: true,
                noundef: true,
                align: Some(4),
                ..Default::default()
            }),
        ],
    );

    mir_lower::lower_mir_to_llvm_with_arg_attrs(&mut ctx, module_ptr, arg_attrs)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|e| anyhow::anyhow!("export failed: {e}"))?;
    let define_line = ir
        .lines()
        .find(|line| line.contains("@copy(") && line.trim_start().starts_with("define"))
        .unwrap_or_else(|| panic!("no `define ... @copy(` line in IR:\n{ir}"));

    // Shared input (flattened param 0): noalias + readonly + align.
    assert!(
        define_line.contains("ptr noalias readonly nonnull noundef align 4 %v0"),
        "expected `input` data pointer (%v0) to carry `noalias readonly ... align 4`:\n{define_line}"
    );
    // DisjointSlice output (flattened param 2): noalias + align, but NOT readonly
    // (it is the write channel).
    assert!(
        define_line.contains("ptr noalias nonnull noundef align 4 %v2"),
        "expected `DisjointSlice` output data pointer (%v2) to carry `noalias ... align 4`:\n{define_line}"
    );
    // Soundness: `readonly` lands on the read-only input only, never on the
    // written DisjointSlice output.
    assert_eq!(
        ir.matches("readonly").count(),
        1,
        "`readonly` must appear exactly once (on the shared input only):\n{ir}"
    );

    Ok(())
}

/// Per-attribute applicability gate: each LLVM parameter attribute is emitted
/// only on a lowered parameter class it is valid on. Pointer-only tokens
/// (`noalias`/`readonly`/`nonnull`/`dereferenceable`/`align`) must never land on
/// an integer; `signext`/`zeroext` must never land on a pointer; `noundef`
/// lands on any value.
///
/// Models `fn gate(buf: &mut [f32], n: u32)`. We inject deliberately
/// *over-broad* `ArgAttrs` (flags set that don't apply to the param) and assert
/// the rendered IR keeps only the applicable tokens per param: the slice data
/// pointer (`%v0`) keeps the pointer family but drops the spurious `signext`;
/// the `i32` (`%v2`) keeps `zeroext noundef` and drops the pointer family.
#[test]
fn per_attribute_gate_filters_attrs_by_param_class() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirSliceType;
    use llvm_export::{ArgAttrs, ArgExt};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::{FP32Type, FunctionType, IntegerType, Signedness};
    use pliron::identifier::Identifier;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "gate_module".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    // Signature: (&mut [f32], u32). The slice flattens to (ptr, i64); the u32
    // stays a single i32 param. Flattened params: %v0 ptr, %v1 i64, %v2 i32.
    let f32_ty = FP32Type::get(&ctx);
    let slice_ty = MirSliceType::get(&mut ctx, f32_ty.into());
    let u32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let func_ty = FunctionType::get(&mut ctx, vec![slice_ty.into(), u32_ty.into()], vec![]);

    let func_op = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(&mut ctx, func_op, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(&mut ctx, "gate".try_into().unwrap());

    // Kernel boundary, so the slice flattens to (ptr, len).
    let kernel_key: Identifier = "gpu_kernel".try_into().unwrap();
    func_op
        .deref_mut(&mut ctx)
        .attributes
        .0
        .insert(kernel_key, StringAttr::new("true".to_string()).into());

    {
        let region = func_op.deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![slice_ty.into(), u32_ty.into()]);
        block.insert_at_back(region, &ctx);
        let ret_op = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, &ctx);
    }
    func_op.insert_at_back(module_block, &ctx);

    // Deliberately over-broad attributes to exercise the gate:
    //   arg0 (slice data ptr, pointer class): full pointer family + a spurious
    //        `signext` that must be dropped (ext is integer-only).
    //   arg1 (u32, integer class): pointer family (must all drop) + `zeroext` +
    //        `noundef` (the only two that survive on an integer).
    let mut arg_attrs = mir_lower::context::ArgAttrsMap::new();
    arg_attrs.insert(
        "gate".to_string(),
        vec![
            Some(ArgAttrs {
                noalias: true,
                nonnull: true,
                noundef: true,
                align: Some(4),
                ext: ArgExt::Sign, // spurious on a pointer -> dropped
                ..Default::default()
            }),
            Some(ArgAttrs {
                noalias: true,            // spurious on an integer -> dropped
                readonly: true,           // ditto
                nonnull: true,            // ditto
                noundef: true,            // kept (any value)
                dereferenceable: Some(8), // spurious on an integer -> dropped
                align: Some(2),           // ditto
                ext: ArgExt::Zero,        // kept (integer)
            }),
        ],
    );

    mir_lower::lower_mir_to_llvm_with_arg_attrs(&mut ctx, module_ptr, arg_attrs)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|e| anyhow::anyhow!("export failed: {e}"))?;
    let define_line = ir
        .lines()
        .find(|line| line.contains("@gate(") && line.trim_start().starts_with("define"))
        .unwrap_or_else(|| panic!("no `define ... @gate(` line in IR:\n{ir}"));

    // Pointer param (slice data pointer, %v0): full pointer family, NO signext.
    assert!(
        define_line.contains("ptr noalias nonnull noundef align 4 %v0"),
        "expected slice data pointer (%v0) to carry the pointer family:\n{define_line}"
    );
    // Integer param (%v2): only `zeroext noundef` survives; the pointer family is
    // gated out. Exact match -> any leaked pointer token would break it.
    assert!(
        define_line.contains("i32 zeroext noundef %v2"),
        "expected i32 (%v2) to carry exactly `zeroext noundef`:\n{define_line}"
    );
    // `signext` was set on the pointer arg; it is integer-only and must not appear.
    assert!(
        !define_line.contains("signext"),
        "`signext` is integer-only and must never land on a pointer:\n{define_line}"
    );
    // The pointer family must not leak onto the integer: `noalias` appears once
    // (on the pointer), and `readonly` never (it was only on the gated integer).
    assert_eq!(
        define_line.matches("noalias").count(),
        1,
        "`noalias` must appear once (on the pointer only):\n{define_line}"
    );
    assert!(
        !define_line.contains("readonly"),
        "`readonly` was only on the integer slot and must be gated out:\n{define_line}"
    );

    Ok(())
}

#[test]
fn test_intrinsic_insertion() -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    // Create Module
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    // Create MirFunc
    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&mut ctx, vec![], vec![]);

    // Manual construction of MirFuncOp
    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1, // 1 region
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    // Add body - MirFuncOp has 1 region
    let region = func.get_operation().deref(&ctx).get_region(0);

    // Create block if empty (it is empty by default from Operation::new)
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    // Add ReadPtxSregTidXOp
    let int32_ty = pliron::builtin::types::IntegerType::get(
        &mut ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );

    let tid_op_ptr = Operation::new(
        &mut ctx,
        nvvm::ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![int32_ty.into()],
        vec![],
        vec![],
        0,
    );
    let tid_op = nvvm::ReadPtxSregTidXOp::new(tid_op_ptr);
    tid_op.get_operation().insert_at_back(block, &ctx);

    // Add Return
    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    // Add Func to Module
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    // Run DialectConversion-based lowering
    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Verify result
    let mut found_intrinsic = false;
    let mut found_kernel = false;

    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        if let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) {
            let name = func_op.get_symbol_name(&ctx).to_string();
            if name == "llvm_nvvm_read_ptx_sreg_tid_x" {
                found_intrinsic = true;
                // Intrinsic (declaration) should have 0 regions or empty region
                let num_regions = func_op.get_operation().deref(&ctx).regions().count();
                if num_regions > 0 {
                    assert!(
                        func_op
                            .get_operation()
                            .deref(&ctx)
                            .get_region(0)
                            .deref(&ctx)
                            .iter(&ctx)
                            .next()
                            .is_none()
                    );
                }
            } else if name == "kernel_func" {
                found_kernel = true;
                // Kernel should have body (1 region, not empty)
                assert!(func_op.get_operation().deref(&ctx).regions().count() > 0);
                assert!(
                    func_op
                        .get_operation()
                        .deref(&ctx)
                        .get_region(0)
                        .deref(&ctx)
                        .iter(&ctx)
                        .next()
                        .is_some()
                );
            }
        }
    }

    assert!(found_intrinsic, "Intrinsic function declaration not found");
    assert!(found_kernel, "Kernel function not found");

    Ok(())
}

#[test]
fn test_globaltimer_lowers_to_intrinsic_call() -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&mut ctx, vec![], vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    let i64_ty = pliron::builtin::types::IntegerType::get(
        &mut ctx,
        64,
        pliron::builtin::types::Signedness::Signless,
    );
    let timer_op = Operation::new(
        &mut ctx,
        nvvm::ReadPtxSregGlobaltimerOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    timer_op.insert_at_back(block, &ctx);

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    const INTRINSIC: &str = "llvm_nvvm_read_ptx_sreg_globaltimer";

    let mut found_decl = false;
    let mut found_call = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) else {
            continue;
        };
        let name = func_op.get_symbol_name(&ctx).to_string();

        if name == INTRINSIC {
            // Intrinsic declaration: present with empty body.
            found_decl = true;
            let num_regions = func_op.get_operation().deref(&ctx).regions().count();
            if num_regions > 0 {
                assert!(
                    func_op
                        .get_operation()
                        .deref(&ctx)
                        .get_region(0)
                        .deref(&ctx)
                        .iter(&ctx)
                        .next()
                        .is_none(),
                    "intrinsic declaration must have empty body"
                );
            }
        } else if name == func_name {
            let func_region = func_op.get_operation().deref(&ctx).get_region(0);
            for func_block in func_region.deref(&ctx).iter(&ctx) {
                for body_op in func_block.deref(&ctx).iter(&ctx) {
                    if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                        && let CallOpCallable::Direct(sym) = call.callee(&ctx)
                        && sym.to_string() == INTRINSIC
                    {
                        found_call = true;
                    }
                    assert!(
                        Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx).is_none(),
                        "globaltimer must not lower to inline asm"
                    );
                }
            }
        }
    }

    assert!(
        found_decl,
        "Expected `{INTRINSIC}` declaration in lowered module"
    );
    assert!(
        found_call,
        "Expected call to `{INTRINSIC}` in lowered kernel body"
    );
    Ok(())
}

#[test]
fn test_threadfence_system_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&mut ctx, vec![], vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    let fence_op_ptr = Operation::new(
        &mut ctx,
        nvvm::ThreadfenceSystemOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let fence_op = nvvm::ThreadfenceSystemOp::new(fence_op_ptr);
    fence_op.get_operation().insert_at_back(block, &ctx);

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_inline_asm = false;

    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        if let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) {
            let name = func_op.get_symbol_name(&ctx).to_string();
            if name != func_name {
                continue;
            }

            let func_region = func_op.get_operation().deref(&ctx).get_region(0);
            for func_block in func_region.deref(&ctx).iter(&ctx) {
                for body_op in func_block.deref(&ctx).iter(&ctx) {
                    if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx)
                        && inline_asm
                            .get_attr_inline_asm_template(&ctx)
                            .is_some_and(|s| String::from((*s).clone()) == "membar.sys;")
                    {
                        found_inline_asm = true;
                        assert!(
                            inline_asm
                                .get_attr_inline_asm_convergent(&ctx)
                                .is_some_and(|b| bool::from((*b).clone()))
                        );
                    }
                }
            }
        }
    }

    assert!(
        found_inline_asm,
        "Expected membar.sys inline asm in lowered kernel"
    );
    Ok(())
}

/// Regression cover for the per-call-site address-space coercion pass.
///
/// When a caller passes a pointer in one address space to a callee whose
/// declared parameter lives in a different address space (the
/// `*mut SharedArray<T, N>` / `addrspace(3)` case that surfaces from
/// `block_reduce` and friends), the lowerer must look up the callee's
/// declared signature and insert an `llvm.addrspacecast` so the LLVM-IR
/// verifier sees matching pointer types at the call site.
///
/// This test builds two MIR functions in one module:
///   - `callee(p: *mut i32 in addrspace(3))`
///   - `caller(p: *mut i32 in addrspace(0)) { callee(p) }`
///
/// and asserts the lowered `caller` body contains an `AddrSpaceCastOp`.
#[test]
fn addrspace_coercion_inserts_addrspacecast_at_call_site() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use llvm_export::ops::AddrSpaceCastOp;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_addrspace_coercion".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let shared_ptr_ty = MirPtrType::get_shared(&mut ctx, i32_ty.into(), true);
    let generic_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);

    // Callee: takes a *mut i32 in addrspace(3), returns ().
    let callee_func_ty = FunctionType::get(&mut ctx, vec![shared_ptr_ty.into()], vec![]);
    let callee_func_op = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let callee_func = mir::MirFuncOp::new(
        &mut ctx,
        callee_func_op,
        TypeAttr::new(callee_func_ty.into()),
    );
    callee_func.set_symbol_name(&mut ctx, "callee".try_into().unwrap());
    {
        let region = callee_func.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![shared_ptr_ty.into()]);
        block.insert_at_back(region, &ctx);

        let ret_op = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, &ctx);
    }
    callee_func
        .get_operation()
        .insert_at_back(module_block, &ctx);

    // Caller: takes a *mut i32 in addrspace(0), calls `callee` with that
    // pointer. The lowerer is responsible for inserting an addrspacecast
    // since the callee's declared addrspace differs.
    let caller_func_ty = FunctionType::get(&mut ctx, vec![generic_ptr_ty.into()], vec![]);
    let caller_func_op = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let caller_func = mir::MirFuncOp::new(
        &mut ctx,
        caller_func_op,
        TypeAttr::new(caller_func_ty.into()),
    );
    caller_func.set_symbol_name(&mut ctx, "caller".try_into().unwrap());
    {
        let region = caller_func.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![generic_ptr_ty.into()]);
        block.insert_at_back(region, &ctx);
        let arg = block.deref(&ctx).get_argument(0);

        let call_op_ptr = Operation::new(
            &mut ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![],
            vec![arg],
            vec![],
            0,
        );
        let call_op = mir::MirCallOp::new(call_op_ptr);
        call_op.set_attr_callee(&ctx, StringAttr::new("callee".to_string()));
        call_op_ptr.insert_at_back(block, &ctx);

        let ret_op = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, &ctx);
    }
    caller_func
        .get_operation()
        .insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_addrspace_cast = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "caller" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                if Operation::get_op::<AddrSpaceCastOp>(body_op, &ctx).is_some() {
                    found_addrspace_cast = true;
                }
            }
        }
    }

    assert!(
        found_addrspace_cast,
        "caller body must contain llvm.addrspacecast for the addrspace(0) -> (3) coercion at the call site",
    );
    Ok(())
}

/// Lock the comparison-predicate lowering table to the rustc_codegen_ssa
/// reference (`bin_op_to_fcmp_predicate` / `bin_op_to_icmp_predicate`):
///
/// | MIR op   | float `fcmp`      | signed `icmp` | unsigned `icmp` |
/// |----------|-------------------|---------------|-----------------|
/// | `mir.eq` | `oeq` (ordered)   | `eq`          | `eq`            |
/// | `mir.ne` | `une` (UNordered) | `ne`          | `ne`            |
/// | `mir.lt` | `olt`             | `slt`         | `ult`           |
/// | `mir.le` | `ole`             | `sle`         | `ule`           |
/// | `mir.gt` | `ogt`             | `sgt`         | `ugt`           |
/// | `mir.ge` | `oge`             | `sge`         | `uge`           |
///
/// `ne` is the one float predicate that must be UNordered: Rust requires
/// `a != b == !(a == b)`, so `x != x` must be true for NaN (issue #123;
/// the ordered `one` folds the canonical NaN check to `false`).
///
/// The test also locks fastmath flags to *empty* on every lowered `fcmp`:
/// a future `nnan` default would make `fcmp nnan une x, x` poison for NaN
/// and silently re-break NaN detection while the predicate assertion above
/// stays green.
#[test]
fn test_cmp_predicate_lowering() -> Result<(), anyhow::Error> {
    use llvm_export::attributes::{FCmpPredicateAttr, FastmathFlagsAttr, ICmpPredicateAttr};
    use llvm_export::op_interfaces::FastMathFlags;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let f32_ty = pliron::builtin::types::FP32Type::get(&ctx);
    let i32_signed = pliron::builtin::types::IntegerType::get(
        &mut ctx,
        32,
        pliron::builtin::types::Signedness::Signed,
    );
    let u32_unsigned = pliron::builtin::types::IntegerType::get(
        &mut ctx,
        32,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let bool_ty = pliron::builtin::types::IntegerType::get(
        &mut ctx,
        1,
        pliron::builtin::types::Signedness::Signless,
    );

    // Args: (f32, f32, i32, u32). The integer args carry pre-conversion
    // signedness, which is what selects signed vs unsigned icmp predicates.
    let arg_tys: Vec<pliron::context::Ptr<pliron::r#type::TypeObj>> = vec![
        f32_ty.into(),
        f32_ty.into(),
        i32_signed.into(),
        u32_unsigned.into(),
    ];
    let func_name = "cmp_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&mut ctx, arg_tys.clone(), vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, arg_tys);
        b.insert_at_back(region, &ctx);
        b
    };
    let fa = block.deref(&ctx).get_argument(0);
    let fb = block.deref(&ctx).get_argument(1);
    let si = block.deref(&ctx).get_argument(2);
    let ui = block.deref(&ctx).get_argument(3);

    // One comparison op per table row, in a fixed program order. The raw
    // `Operation::new` construction mirrors how the importer builds these
    // ops (mir-importer translator/rvalue.rs BinaryOp arm).
    let cmp_infos = [
        // Floats: all six predicates.
        (mir::MirEqOp::get_concrete_op_info(), fa, fb),
        (mir::MirNeOp::get_concrete_op_info(), fa, fb),
        (mir::MirLtOp::get_concrete_op_info(), fa, fb),
        (mir::MirLeOp::get_concrete_op_info(), fa, fb),
        (mir::MirGtOp::get_concrete_op_info(), fa, fb),
        (mir::MirGeOp::get_concrete_op_info(), fa, fb),
        // Signed integers: eq/ne are sign-agnostic, the rest must be s*.
        (mir::MirEqOp::get_concrete_op_info(), si, si),
        (mir::MirNeOp::get_concrete_op_info(), si, si),
        (mir::MirLtOp::get_concrete_op_info(), si, si),
        (mir::MirLeOp::get_concrete_op_info(), si, si),
        (mir::MirGtOp::get_concrete_op_info(), si, si),
        (mir::MirGeOp::get_concrete_op_info(), si, si),
        // Unsigned integers: the relational predicates must be u*.
        (mir::MirLtOp::get_concrete_op_info(), ui, ui),
        (mir::MirLeOp::get_concrete_op_info(), ui, ui),
        (mir::MirGtOp::get_concrete_op_info(), ui, ui),
        (mir::MirGeOp::get_concrete_op_info(), ui, ui),
    ];
    for (info, lhs, rhs) in cmp_infos {
        let op = Operation::new(
            &mut ctx,
            info,
            vec![bool_ty.into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        op.insert_at_back(block, &ctx);
    }

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op_ptr.insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Collect lowered predicates in program order.
    let mut fcmp_preds = Vec::new();
    let mut icmp_preds = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != func_name {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                if let Some(fcmp) = Operation::get_op::<llvm::FCmpOp>(body_op, &ctx) {
                    fcmp_preds.push(fcmp.predicate(&ctx));
                    assert_eq!(
                        fcmp.fast_math_flags(&ctx),
                        FastmathFlagsAttr::default(),
                        "fcmp must carry empty fastmath flags: nnan would poison NaN checks"
                    );
                }
                if let Some(icmp) = Operation::get_op::<llvm::ICmpOp>(body_op, &ctx) {
                    icmp_preds.push(icmp.predicate(&ctx));
                }
            }
        }
    }

    assert_eq!(
        fcmp_preds,
        vec![
            FCmpPredicateAttr::OEQ,
            FCmpPredicateAttr::UNE,
            FCmpPredicateAttr::OLT,
            FCmpPredicateAttr::OLE,
            FCmpPredicateAttr::OGT,
            FCmpPredicateAttr::OGE,
        ],
        "float comparison predicates must mirror rustc: ordered except Ne (une)"
    );
    assert_eq!(
        icmp_preds,
        vec![
            ICmpPredicateAttr::EQ,
            ICmpPredicateAttr::NE,
            ICmpPredicateAttr::SLT,
            ICmpPredicateAttr::SLE,
            ICmpPredicateAttr::SGT,
            ICmpPredicateAttr::SGE,
            ICmpPredicateAttr::ULT,
            ICmpPredicateAttr::ULE,
            ICmpPredicateAttr::UGT,
            ICmpPredicateAttr::UGE,
        ],
        "integer comparison predicates must respect pre-conversion signedness"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: build a void-returning kernel with a single NVVM op, lower it, and
// assert the kernel body contains an InlineAsmOp whose template includes the
// given `expected_asm` substring.
// ---------------------------------------------------------------------------

/// Build a kernel whose entry block contains `op` + `mir.return`, lower to LLVM,
/// and verify an `InlineAsmOp` with `expected_asm` in its template exists.
fn assert_inline_asm_lowering(
    ctx: &mut Context,
    module_ptr: pliron::context::Ptr<Operation>,
    expected_asm: &str,
) -> Result<(), anyhow::Error> {
    mir_lower::lower_mir_to_llvm(ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found = false;
    let module_op = module_ptr.deref(ctx);
    let region = module_op.get_region(0);
    let block = region.deref(ctx).iter(ctx).next().unwrap();

    for op in block.deref(ctx).iter(ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        if func_op.get_symbol_name(ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(ctx).get_region(0);
        for func_block in func_region.deref(ctx).iter(ctx) {
            for body_op in func_block.deref(ctx).iter(ctx) {
                if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, ctx)
                    && inline_asm
                        .get_attr_inline_asm_template(ctx)
                        .is_some_and(|s| String::from((*s).clone()).contains(expected_asm))
                {
                    found = true;
                }
            }
        }
    }

    assert!(
        found,
        "Expected inline asm containing `{expected_asm}` in lowered kernel"
    );
    Ok(())
}

/// Helper: fresh context with all dialects registered.
fn make_test_ctx() -> Context {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);
    ctx
}

/// Helper: build a module + MirFuncOp("kernel_func") with given arg types,
/// returning the module ptr and entry block.
fn build_test_kernel(
    ctx: &mut Context,
    arg_tys: Vec<pliron::context::Ptr<pliron::r#type::TypeObj>>,
) -> (
    pliron::context::Ptr<Operation>,
    pliron::context::Ptr<pliron::basic_block::BasicBlock>,
) {
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::types::FunctionType;

    let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
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
    func.set_symbol_name(ctx, "kernel_func".try_into().unwrap());

    let region = func.get_operation().deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, arg_tys);
    entry.insert_at_back(region, ctx);

    let module_region = module_ptr.deref(ctx).get_region(0);
    let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, ctx);

    (module_ptr, entry)
}

/// Helper: append a mir.return (void) to a block.
fn append_return(ctx: &mut Context, block: pliron::context::Ptr<pliron::basic_block::BasicBlock>) {
    let ret = Operation::new(
        ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(block, ctx);
}

// ---------------------------------------------------------------------------
// cvt.f16x2 intrinsic lowering test
// ---------------------------------------------------------------------------

#[test]
fn test_cvt_f16x2_f32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);

    let lo_val = entry.deref(&ctx).get_argument(0);
    let hi_val = entry.deref(&ctx).get_argument(1);

    // CvtF16x2F32Op: 2 f32 operands, 1 i32 result
    let op = Operation::new(
        &mut ctx,
        nvvm::CvtF16x2F32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "cvt.rn.f16x2.f32")
}
