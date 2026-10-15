/*
 * SPDX-FileCopyrightText: Copyright (c) The pliron contributors
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CFG cleanup for annotated MIR functions. A MIR assertion exposes its success
//! successor and has an implicit trapping failure path. Unlike an unconditional
//! goto, it must survive merging even when its success block has one predecessor.

use dialect_mir::ops::MirGotoOp;
use pliron::attribute::AttrObj;
use pliron::context::{Context, Ptr};
use pliron::graph::walkers::{
    IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op,
};
use pliron::irbuild::{
    inserter::{Inserter, OpInsertionPoint},
    listener::DummyListener,
    rewriter::IRRewriter,
    rewriter::Rewriter,
};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::{op_cast, op_impls};
use pliron::operation::Operation;
use pliron::opts::constants::{BranchOpFoldInterface, ConstFoldInterface};
use pliron::opts::simplify_cfg::remove_blocks_inside_op;
use pliron::result::Result;
use rustc_hash::FxHashSet;

fn constant_operands(op: Ptr<Operation>, ctx: &Context) -> Vec<Option<AttrObj>> {
    op.deref(ctx)
        .operands()
        .map(|value| {
            let definition = value.defining_op()?;
            let dynamic = Operation::get_op_dyn(definition, ctx);
            let fold = op_cast::<dyn ConstFoldInterface>(dynamic.as_ref())?;
            let unknown = vec![None; definition.deref(ctx).get_num_operands()];
            fold.check_fold(ctx, &unknown)[value.find_index(ctx)].clone()
        })
        .collect()
}

pub(crate) fn simplify_cfg(function: Ptr<Operation>, ctx: &mut Context) -> Result<()> {
    let mut candidates = Vec::new();
    walk_op(
        ctx,
        &mut candidates,
        &WALKCONFIG_PREORDER_FORWARD,
        function,
        |ctx, candidates, node| {
            if let IRNode::Operation(op) = node
                && op_impls::<dyn BranchOpFoldInterface>(Operation::get_op_dyn(op, ctx).as_ref())
            {
                candidates.push((op, constant_operands(op, ctx)));
            }
        },
    );
    let mut rewriter = IRRewriter::<DummyListener>::default();
    for (op, operands) in candidates {
        rewriter.set_insertion_point_before_operation(op);
        let dynamic = Operation::get_op_dyn(op, ctx);
        op_cast::<dyn BranchOpFoldInterface>(dynamic.as_ref())
            .unwrap()
            .fold_in_place(ctx, &operands, &mut rewriter);
    }
    remove_blocks_inside_op(function, ctx, &mut rewriter);

    let mut regions = Vec::new();
    walk_op(
        ctx,
        &mut regions,
        &WALKCONFIG_PREORDER_FORWARD,
        function,
        |ctx, regions, node| {
            if let IRNode::Operation(op) = node {
                regions.extend(op.deref(ctx).regions());
            }
        },
    );
    for region in regions {
        if !region.deref(ctx).has_ssa_dominance(ctx) {
            continue;
        }
        let Some(entry) = region.deref(ctx).get_head() else {
            continue;
        };
        let mut pending = vec![entry];
        let mut visited = FxHashSet::default();
        while let Some(block) = pending.pop() {
            if !visited.insert(block) {
                continue;
            }
            loop {
                let terminator = { block.deref(ctx).get_terminator(ctx) };
                let Some(terminator) = terminator else {
                    break;
                };
                if Operation::get_op::<MirGotoOp>(terminator, ctx).is_none() {
                    break;
                }
                let successor = terminator.deref(ctx).get_successor(0);
                if successor == entry || successor == block || successor.num_preds(ctx) != 1 {
                    break;
                }
                let formal: Vec<_> = successor.deref(ctx).arguments().collect();
                let actual: Vec<_> = terminator.deref(ctx).operands().collect();
                assert_eq!(formal.len(), actual.len());
                for (formal, actual) in formal.into_iter().zip(actual) {
                    rewriter.replace_value_uses_with(ctx, formal, actual);
                }
                rewriter.erase_operation(ctx, terminator);
                let operations: Vec<_> = successor.deref(ctx).iter(ctx).collect();
                for operation in operations {
                    rewriter.move_operation(ctx, operation, OpInsertionPoint::AtBlockEnd(block));
                }
                rewriter.erase_block(ctx, successor);
            }
            pending.extend(block.deref(ctx).succs(ctx));
        }
    }
    Ok(())
}
