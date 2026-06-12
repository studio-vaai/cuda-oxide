/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Programmatic Dependent Launch (PDL) Example
//!
//! Demonstrates the PDL intrinsics and launch attribute:
//! - `griddepcontrol_launch_dependents()` -- primary kernel signals that
//!   dependent kernels may launch (PTX `griddepcontrol.launch_dependents`)
//! - `griddepcontrol_wait()` -- secondary kernel blocks until the primary
//!   has completed and flushed its writes (PTX `griddepcontrol.wait`)
//! - `cuda_launch! { ..., programmatic: true }` -- launches the secondary
//!   with `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
//!
//! The primary kernel produces data, triggers, then spins in a long "tail";
//! the secondary kernel spins in a long "prologue" (work independent of the
//! primary's output), waits, then consumes the data. With PDL the secondary's
//! prologue overlaps the primary's tail; without it the kernels serialize.
//! Both kernels timestamp their phases with `%globaltimer` so the overlap is
//! measured directly, and the consumed data is verified in both modes.
//!
//! Requires compute capability 9.0+ (Hopper / Blackwell).
//!
//! Build and run with:
//!   cargo oxide run pdl

use std::sync::Arc;

use cuda_core::{CudaContext, CudaModule, DeviceBuffer, LaunchConfig};
use cuda_device::pdl::{griddepcontrol_launch_dependents, griddepcontrol_wait};
use cuda_device::{DisjointSlice, debug, device, kernel, thread};
use cuda_host::{cuda_launch, cuda_module};

// Bring the `#[cuda_module]`-generated kernel marker types into scope for
// the `cuda_launch!` lookups below.
use crate::kernels::*;

/// Timestamp slots written by the kernels (ns, `%globaltimer`).
const T_PRIMARY_TRIGGER: usize = 0;
const T_PRIMARY_END: usize = 1;
const T_SECONDARY_START: usize = 2;
const T_SECONDARY_POSTWAIT: usize = 3;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Busy-wait for `spin_ns` nanoseconds on the global timer.
    #[device]
    fn spin(spin_ns: u64) {
        let start = debug::globaltimer();
        while debug::globaltimer().wrapping_sub(start) < spin_ns {}
    }

    /// Record a `%globaltimer` timestamp in `times[slot]`.
    ///
    /// Raw-pointer write because the slot is not the calling thread's own
    /// index; callers guarantee each (kernel, slot) pair has one writer.
    #[device]
    fn stamp(times: &mut [u64], slot: usize) {
        unsafe {
            *times.as_mut_ptr().add(slot) = debug::globaltimer();
        }
    }

    /// Primary kernel: produce data, allow dependents to launch, then run a
    /// long tail that does not touch the produced data.
    #[kernel]
    pub fn primary(mut data: DisjointSlice<u32>, times: &mut [u64], spin_ns: u64) {
        let idx = thread::index_1d();
        let i = idx.get();

        // Phase 1: produce the data the secondary kernel will consume.
        if let Some(d) = data.get_mut(idx) {
            *d = i as u32 * 3 + 7;
        }

        if i == 0 {
            stamp(times, T_PRIMARY_TRIGGER);
        }

        // Phase 2: dependents may launch from here on. No memory-visibility
        // promise is made at the trigger -- the secondary's
        // griddepcontrol_wait() is what orders its reads after our writes.
        griddepcontrol_launch_dependents();

        // Phase 3: tail work (independent of `data`).
        spin(spin_ns);

        if i == 0 {
            stamp(times, T_PRIMARY_END);
        }
    }

    /// Secondary kernel: run a long prologue that is independent of the
    /// primary's output, then wait and consume the data.
    #[kernel]
    pub fn secondary(data: &[u32], mut out: DisjointSlice<u32>, times: &mut [u64], spin_ns: u64) {
        let idx = thread::index_1d();
        let i = idx.get();

        if i == 0 {
            stamp(times, T_SECONDARY_START);
        }

        // Prologue: work that does NOT read the primary's output. Under a
        // programmatic launch this overlaps the primary's tail.
        spin(spin_ns);

        // Block until the primary grid has completed and flushed its writes.
        // (A no-op when launched without the programmatic attribute.)
        griddepcontrol_wait();

        if i == 0 {
            stamp(times, T_SECONDARY_POSTWAIT);
        }

        // Consume: only safe after griddepcontrol_wait().
        if let Some(o) = out.get_mut(idx) {
            *o = data[i] * 2;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

/// Spin duration for the primary tail / secondary prologue (2 ms each).
const SPIN_NS: u64 = 2_000_000;
const N: usize = 512;

struct RunResult {
    /// Wall-clock overlap between secondary start and primary end (ns):
    /// positive when the secondary started before the primary ended.
    overlap_ns: i64,
    /// The secondary's post-wait timestamp is not earlier than the
    /// primary's end timestamp.
    wait_after_primary_end: bool,
    /// Data verification.
    data_ok: bool,
}

fn run_pair(ctx: &Arc<CudaContext>, module: &Arc<CudaModule>, programmatic: bool) -> RunResult {
    let stream = ctx.default_stream();

    let mut data = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let mut times = DeviceBuffer::<u64>::zeroed(&stream, 4).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (4, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    cuda_launch! {
        kernel: primary,
        stream: stream,
        module: module,
        config: cfg,
        args: [slice_mut(data), slice_mut(times), SPIN_NS]
    }
    .expect("primary launch failed");

    cuda_launch! {
        kernel: secondary,
        stream: stream,
        module: module,
        config: cfg,
        programmatic: programmatic,
        args: [slice(data), slice_mut(out), slice_mut(times), SPIN_NS]
    }
    .expect("secondary launch failed");

    stream.synchronize().unwrap();

    let out_host = out.to_host_vec(&stream).unwrap();
    let times_host = times.to_host_vec(&stream).unwrap();

    let data_ok = out_host
        .iter()
        .enumerate()
        .all(|(i, &v)| v == (i as u32 * 3 + 7) * 2);

    let overlap_ns = times_host[T_PRIMARY_END] as i64 - times_host[T_SECONDARY_START] as i64;
    let wait_after_primary_end = times_host[T_SECONDARY_POSTWAIT] >= times_host[T_PRIMARY_END];

    RunResult {
        overlap_ns,
        wait_after_primary_end,
        data_ok,
    }
}

fn main() {
    println!("=== Programmatic Dependent Launch (PDL) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);
    if major < 9 {
        // Gate before load_module_from_file: the driver rejects the
        // griddepcontrol PTX instructions on pre-Hopper devices.
        println!("\nskipping: Programmatic Dependent Launch requires sm_90+ (Hopper)");
        return;
    }

    let module = ctx
        .load_module_from_file("pdl.ptx")
        .expect("Failed to load PTX module");

    // Warm-up pass (module load, instruction cache) -- discard results.
    let _ = run_pair(&ctx, &module, false);

    println!("--- serialized launch (control) ---");
    let control = run_pair(&ctx, &module, false);
    println!(
        "  data correct: {}",
        if control.data_ok { "yes" } else { "NO" }
    );
    println!(
        "  secondary started {:.3} ms {} primary end",
        control.overlap_ns.abs() as f64 / 1e6,
        if control.overlap_ns > 0 {
            "BEFORE"
        } else {
            "after"
        },
    );

    println!("\n--- programmatic launch (PDL) ---");
    let pdl = run_pair(&ctx, &module, true);
    println!("  data correct: {}", if pdl.data_ok { "yes" } else { "NO" });
    println!(
        "  wait returned after primary end: {}",
        if pdl.wait_after_primary_end {
            "yes"
        } else {
            "NO"
        },
    );
    println!(
        "  secondary started {:.3} ms {} primary end",
        pdl.overlap_ns.abs() as f64 / 1e6,
        if pdl.overlap_ns > 0 {
            "BEFORE"
        } else {
            "after"
        },
    );

    // Correctness is a hard failure in both modes; the wait ordering is a
    // hard failure in PDL mode.
    if !control.data_ok || !pdl.data_ok {
        println!("\n✗ FAILED: secondary read wrong data");
        std::process::exit(1);
    }
    if !pdl.wait_after_primary_end {
        println!("\n✗ FAILED: griddepcontrol_wait returned before primary end");
        std::process::exit(1);
    }

    // Overlap is opportunistic per the CUDA programming model -- the driver
    // may serialize (e.g. fully occupied GPU). Report rather than fail, but
    // on an idle GPU the secondary's 2 ms prologue should overlap the
    // primary's 2 ms tail almost entirely.
    if pdl.overlap_ns > 0 {
        println!(
            "\n✓ PDL overlap observed: {:.3} ms of the secondary prologue ran \
             during the primary tail",
            pdl.overlap_ns as f64 / 1e6,
        );
    } else {
        println!(
            "\n⚠ no overlap observed (PDL concurrency is opportunistic; GPU \
             may be busy) -- correctness checks still passed"
        );
    }

    if control.overlap_ns > 0 {
        println!(
            "⚠ unexpected: control (serialized) run also overlapped by {:.3} ms",
            control.overlap_ns as f64 / 1e6,
        );
    }

    println!("\n=== PDL example PASS ===");
}
