/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Non-bulk asynchronous copy (LDGSTS) intrinsics: `cp.async.*`.
//!
//! These are the *per-element* asynchronous global→shared copy instructions
//! introduced on Ampere (sm_80+), as opposed to the descriptor-based TMA bulk
//! copies in [`crate::tma`]. They are the right tool for **software-pipelined
//! prefetch of scattered gathers**: each thread issues a small (4/8/16 B) copy
//! of a value it computed the address for, overlapping the global-load latency
//! with independent compute.
//!
//! # Workflow
//!
//! ```text
//! 1. Issue copies:   cp_async_ca_shared_global_16(smem_stage_k, &global[idx])   (×N)
//! 2. Commit group:   cp_async_commit_group()
//! 3. ... issue the NEXT stage's copies + commit ...
//! 4. Wait:           cp_async_wait_group(1)   // keep at most 1 group in flight
//! 5. Read the staged values from shared memory
//! ```
//!
//! # Requirements
//!
//! - **PTX ISA**: 7.0+
//! - **Architecture**: sm_80+ (Ampere and newer)
//!
//! # Caching variants
//!
//! `.ca` caches in L1 **and** L2 (good when the gathered data is reused across
//! threads in the block). The 16-byte width is the natural unit for a
//! [`crate::cusimd::CuSimd<f32, 4>`] (one `Vec4` position).

/// Asynchronously copy **16 bytes** from global to shared memory (`.ca` cache hint).
///
/// Lowers to `cp.async.ca.shared.global [dst], [src], 16;` — a single LDGSTS
/// instruction. The copy is *not* complete on return: track completion with
/// [`cp_async_commit_group`] + [`cp_async_wait_group`] before reading `dst`.
///
/// # Parameters
///
/// - `dst`: destination in shared memory (lowered to address space 3)
/// - `src`: source in global memory (lowered to address space 1)
///
/// # Safety
///
/// - `dst` must point to at least 16 bytes of valid shared memory, 16-B aligned.
/// - `src` must point to at least 16 bytes of valid global memory, 16-B aligned.
/// - The destination must not be read until a matching
///   [`cp_async_wait_group`] has retired the group containing this copy.
///
/// # PTX
///
/// ```ptx
/// cp.async.ca.shared.global [%dst], [%src], 16;
/// ```
#[inline(never)]
pub unsafe fn cp_async_ca_shared_global_16(dst: *mut u8, src: *const u8) {
    let _ = (dst, src);
    // Lowered to: @llvm.nvvm.cp.async.ca.shared.global.16(ptr addrspace(3), ptr addrspace(1))
    unreachable!("cp_async_ca_shared_global_16 called outside CUDA kernel context")
}

/// Commit all preceding `cp.async` copies into a single completion group.
///
/// # PTX
///
/// ```ptx
/// cp.async.commit_group;
/// ```
#[inline(never)]
pub fn cp_async_commit_group() {
    // Lowered to inline PTX: cp.async.commit_group;
    unreachable!("cp_async_commit_group called outside CUDA kernel context")
}

/// Wait until at most `n` previously committed `cp.async` groups are pending.
///
/// Use `n = 0` to wait for *all* outstanding groups. For a depth-2 pipeline,
/// pass `n = 1` to keep the most recently committed group in flight while the
/// older one is guaranteed complete.
///
/// `n` must be a compile-time constant (it lowers to an immediate PTX operand).
///
/// # PTX
///
/// ```ptx
/// cp.async.wait_group N;
/// ```
#[inline(never)]
pub fn cp_async_wait_group(n: u32) {
    let _ = n;
    // Lowered to inline PTX: cp.async.wait_group $0;
    unreachable!("cp_async_wait_group called outside CUDA kernel context")
}
