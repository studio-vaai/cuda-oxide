/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Select fixed stack slots for emission in the true LLVM entry block.

use pliron::{
    builtin::attributes::IntegerAttr,
    context::{Context, Ptr},
    graph::traversals::region::sccs_in_topological_order,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    value::Value,
};
use rustc_hash::FxHashSet;

use crate::ops::{self, FuncOp};

#[derive(Default)]
pub(super) struct EntryAllocas {
    pub operations: Vec<Ptr<Operation>>,
    pub members: FxHashSet<Ptr<Operation>>,
}

/// A constant-sized allocation outside a cycle executes at most once. In a
/// cycle, reusing a slot is safe only when no iteration can retain or observe
/// the allocation's address. Keep escaping loop slots and dynamic counts at
/// their original sites rather than changing their lifetime or identity.
pub(super) fn entry_allocas(ctx: &Context, func: &FuncOp) -> EntryAllocas {
    let region = func.get_operation().deref(ctx).get_region(0);
    let candidates: Vec<_> = region
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| {
            block.deref(ctx).iter(ctx).filter_map(move |operation| {
                let object = Operation::get_op_dyn(operation, ctx);
                object.as_ref().downcast_ref::<ops::AllocaOp>()?;
                let count = operation.deref(ctx).get_operand(0).defining_op()?;
                let count_object = Operation::get_op_dyn(count, ctx);
                let constant = count_object.as_ref().downcast_ref::<ops::ConstantOp>()?;
                let attribute = constant.get_value(ctx);
                attribute.downcast_ref::<IntegerAttr>()?;
                Some((block, operation))
            })
        })
        .collect();
    if candidates.is_empty() {
        return EntryAllocas::default();
    }
    let cyclic_blocks: FxHashSet<_> = sccs_in_topological_order(ctx, &region)
        .into_iter()
        .filter(|component| component.is_cyclic)
        .flat_map(|component| component.nodes)
        .collect();
    let operations: Vec<_> = candidates
        .into_iter()
        .filter(|(block, operation)| {
            !cyclic_blocks.contains(block)
                || pointer_is_local_scratch(ctx, operation.deref(ctx).get_result(0))
        })
        .map(|(_, operation)| operation)
        .collect();
    let members = operations.iter().copied().collect();
    EntryAllocas {
        operations,
        members,
    }
}

/// Follow pointer views, accepting only loads and stores through them. Calls,
/// PHIs, pointer comparisons/conversions, returns, and storing the pointer as
/// data can expose distinct loop allocations, so they conservatively stop us.
fn pointer_is_local_scratch(ctx: &Context, pointer: Value) -> bool {
    let mut pending = vec![pointer];
    let mut visited = FxHashSet::default();
    while let Some(pointer) = pending.pop() {
        if !visited.insert(pointer) {
            continue;
        }
        for usage in pointer.uses(ctx) {
            let operation = usage.user_op();
            let index = usage.find_index(ctx);
            let object = Operation::get_op_dyn(operation, ctx);
            let op = object.as_ref();
            if (op.is::<ops::LoadOp>() && index == 0) || (op.is::<ops::StoreOp>() && index == 1) {
                continue;
            }
            if index == 0
                && (op.is::<ops::GetElementPtrOp>()
                    || op.is::<ops::BitcastOp>()
                    || op.is::<ops::AddrSpaceCastOp>())
            {
                pending.push(operation.deref(ctx).get_result(0));
                continue;
            }
            return false;
        }
    }
    true
}
