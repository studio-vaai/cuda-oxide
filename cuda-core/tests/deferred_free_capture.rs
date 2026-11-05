/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A dropped stream-ordered buffer is freed off its allocation stream.
//!
//! Frees run on the completion worker after the drop. Enqueued on the
//! allocation stream, a free that runs while that stream captures becomes a
//! memory node of the caller's graph: every replay would free the memory again,
//! and graph edits such as removing dependencies are refused.

use cuda_core::{sys, CudaContext, DeviceBuffer};
use std::time::{Duration, Instant};

/// Bytes the device's current memory pool hands out.
fn pool_used(ordinal: i32) -> u64 {
    let mut device = 0;
    assert_eq!(unsafe { sys::cuDeviceGet(&mut device, ordinal) }, 0);
    let mut pool = std::ptr::null_mut();
    assert_eq!(unsafe { sys::cuDeviceGetMemPool(&mut pool, device) }, 0);
    let mut used: u64 = 0;
    let status = unsafe {
        sys::cuMemPoolGetAttribute(
            pool,
            sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            (&raw mut used).cast(),
        )
    };
    assert_eq!(status, 0, "cuMemPoolGetAttribute");
    used
}

/// Memory-free nodes in `graph`.
fn free_nodes(graph: sys::CUgraph) -> usize {
    let mut count = 0;
    let status = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut count) };
    assert_eq!(status, 0, "cuGraphGetNodes(count)");
    if count == 0 {
        return 0;
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    let status = unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count) };
    assert_eq!(status, 0, "cuGraphGetNodes");
    nodes
        .into_iter()
        .filter(|&node| {
            let mut ty = 0;
            assert_eq!(unsafe { sys::cuGraphNodeGetType(node, &mut ty) }, 0);
            ty == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEM_FREE
        })
        .count()
}

#[test]
fn deferred_stream_ordered_free_stays_out_of_a_later_capture() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create stream");
    const BYTES: usize = 1 << 20;
    let buffer = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, BYTES) }
        .expect("stream-ordered allocation");
    stream.synchronize().expect("allocation completes");
    let used = pool_used(0);
    assert!(used >= BYTES as u64, "the allocation comes from the pool");
    drop(buffer);

    ctx.bind_to_thread().expect("bind context");
    let status = unsafe {
        sys::cuStreamBeginCapture_v2(
            stream.cu_stream(),
            sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
        )
    };
    assert_eq!(status, 0, "cuStreamBeginCapture");
    // The completion worker polls every millisecond; give it many turns while
    // the allocation stream captures.
    std::thread::sleep(Duration::from_millis(100));
    let mut graph = std::ptr::null_mut();
    let status = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
    assert_eq!(status, 0, "cuStreamEndCapture");
    let captured = free_nodes(graph);
    assert_eq!(unsafe { sys::cuGraphDestroy(graph) }, 0);
    assert_eq!(captured, 0, "the deferred free joined the capture");

    // The memory still returns to the pool, once the private stream runs the free.
    let deadline = Instant::now() + Duration::from_secs(10);
    while pool_used(0) + BYTES as u64 > used {
        assert!(
            Instant::now() < deadline,
            "the dropped buffer stays allocated"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
