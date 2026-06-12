/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Programmatic Dependent Launch (PDL) intrinsics for CUDA device code.
//!
//! PDL lets a *secondary* (dependent) kernel launch while the *primary*
//! kernel in the same stream is still running, overlapping the secondary's
//! prologue (and any work that does not consume the primary's output) with
//! the primary's tail:
//!
//! - [`griddepcontrol_launch_dependents()`] -> PTX `griddepcontrol.launch_dependents`
//!   (CUDA C++ `cudaTriggerProgrammaticLaunchCompletion()`)
//! - [`griddepcontrol_wait()`] -> PTX `griddepcontrol.wait`
//!   (CUDA C++ `cudaGridDependencySynchronize()`)
//!
//! The functions are compiler-recognized stubs. Their bodies never execute;
//! the cuda-oxide compiler replaces each call with the corresponding inline
//! PTX instruction. Requires compute capability 9.0+.
//!
//! # Launch-side requirement
//!
//! The device intrinsics only have an effect when the *secondary* kernel is
//! launched with the `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
//! launch attribute on the same stream as the primary — see
//! `cuda_core::launch_kernel_programmatic_on_stream` or the
//! `programmatic: true` field of `cuda_launch!`. Without the attribute the
//! stream serializes as usual and both intrinsics degenerate to no-ops, so
//! kernels can share one code path for both launch modes. Inside CUDA
//! graphs the same effect is expressed with programmatic edges
//! (`cudaGraphDependencyTypeProgrammatic`); stream capture records the
//! launch attribute automatically.
//!
//! # Memory-visibility contract
//!
//! Concurrency is *opportunistic*: the secondary kernel may launch anywhere
//! between the primary's trigger and its completion, and the trigger makes
//! **no** memory-visibility guarantee. A secondary kernel MUST call
//! [`griddepcontrol_wait()`] before reading anything the primary wrote;
//! `wait` blocks until all upstream grids have completed and their writes
//! to global memory are visible. Code before `wait` must not depend on the
//! primary's output. Relying on the primary and secondary running
//! concurrently (e.g. spinning on a flag the primary sets after the
//! trigger) can deadlock.

/// Signal that dependent (secondary) kernels may launch.
///
/// Once every thread block of the executing grid has either called this or
/// exited, grids launched with the programmatic-stream-serialization
/// attribute on the same stream become eligible to launch. Calling it from
/// one thread per block is sufficient; repeated calls are harmless.
///
/// This carries no memory-ordering semantics — the dependent kernel still
/// observes the primary's writes only after its own
/// [`griddepcontrol_wait()`] returns.
///
/// This is equivalent to CUDA C++ `cudaTriggerProgrammaticLaunchCompletion()`.
#[inline(never)]
pub fn griddepcontrol_launch_dependents() {
    // Lowered to inline PTX: griddepcontrol.launch_dependents;
    unreachable!("griddepcontrol_launch_dependents called outside CUDA kernel context")
}

/// Block until all upstream (primary) grids have completed and flushed
/// their global-memory writes.
///
/// Every thread that subsequently reads data produced by the primary kernel
/// must call this first (it is a per-thread instruction, not a collective).
/// If the kernel was launched without programmatic stream serialization
/// there is nothing to wait on and the instruction is a no-op.
///
/// This is equivalent to CUDA C++ `cudaGridDependencySynchronize()`.
#[inline(never)]
pub fn griddepcontrol_wait() {
    // Lowered to inline PTX: griddepcontrol.wait;
    unreachable!("griddepcontrol_wait called outside CUDA kernel context")
}
