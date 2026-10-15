// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{
    DisjointSlice, DynamicSharedArray, SharedArray, device, kernel, launch_contract, thread,
};
use cuda_host::cuda_module;
mod host_cache_sentinel;

// Required associated items have no definition to query during cache hashing.
#[allow(dead_code)]
trait MetadataFenceFixture {
    type Item;
    const REQUIRED: u32;
}

#[derive(Clone, Copy)]
struct Pair {
    x: u32,
    y: u32,
}

#[cuda_module]
mod shared {
    use super::*;
    pub static mut SCRATCH: SharedArray<u32, 32> = SharedArray::UNINIT;
    #[device]
    #[inline(never)]
    pub fn read_scratch(index: usize) -> u32 {
        let ptr = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCRATCH) };
        unsafe { *ptr.add(index) }
    }
    // Exercise device pointer remainder rather than a library intrinsic.
    #[allow(clippy::manual_is_multiple_of)]
    #[device]
    pub fn read_dynamic(index: usize) -> u32 {
        let ptr = DynamicSharedArray::<u32>::get();
        let alignment_error = if (ptr as usize) % 256 == 0 { 0 } else { 1000 };
        unsafe { *ptr.add(index) + alignment_error }
    }
    // Retain the device ABI call; the ordinary host shim adds its own inline hint.
    #[allow(unused_attributes)]
    #[device]
    #[inline(never)]
    pub fn pair<const MULTIPLIER: u32>(value: u32) -> Pair {
        Pair {
            x: value + 2,
            y: value * MULTIPLIER,
        }
    }
    // Shares the helper LTOIR unit but only the second kernel calls it.
    #[device]
    #[inline(never)]
    pub fn only_second(value: u32) -> u32 {
        super::second_dependency::identity(value) + 777
    }
    #[device]
    #[inline(never)]
    pub fn unused(value: u32) -> u32 {
        value + 999
    }
}
// The first kernel reaches `shared`, whose unrelated helper reaches this unit.
// It must appear only in the second kernel's native dependency closure.
#[cuda_module]
mod second_dependency {
    use super::*;
    #[device]
    #[inline(never)]
    pub fn identity(value: u32) -> u32 {
        value
    }
}
#[cuda_module]
mod first {
    use super::*;
    #[device]
    #[inline(never)]
    pub fn helper(value: u32) -> u32 {
        let checked = if value < 32 { Some(value) } else { None };
        checked.expect("device schedule requires control")
    }
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = 128, dynamic_shared_alignment = 128)]
    pub fn first(mut out: DisjointSlice<u32>) {
        let i = thread::index_1d();
        let value = helper(i.get() as u32);
        static mut LOCAL: SharedArray<u32, 32> = SharedArray::UNINIT;
        let ptr = unsafe { SharedArray::as_raw_mut_ptr(&raw mut shared::SCRATCH) };
        let local = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LOCAL) };
        let dynamic = DynamicSharedArray::<u32>::get();
        unsafe {
            *ptr.add(value as usize) = value;
            *local.add(value as usize) = 7;
            *dynamic.add(value as usize) = value;
        }
        thread::sync_threads();
        if let Some(v) = out.get_mut(i) {
            let pair = shared::pair::<3>(shared::read_scratch(value as usize));
            let biases = &[1u32, 3, 5];
            let kernel_bias = 10u32;
            *v = pair.x
                + pair.y
                + kernel_bias
                + biases[value as usize % 3]
                + shared::read_dynamic(value as usize)
                - value
                + unsafe { *local.add(value as usize) }
                - 7;
        }
    }
}
#[cuda_module]
mod second {
    use super::*;
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = 128, dynamic_shared_alignment = 256)]
    pub fn second(mut out: DisjointSlice<u32>) {
        let i = thread::index_1d();
        let value = first::helper(i.get() as u32);
        static mut LOCAL: SharedArray<u32, 64> = SharedArray::UNINIT;
        let ptr = unsafe { SharedArray::as_raw_mut_ptr(&raw mut shared::SCRATCH) };
        let local = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LOCAL) };
        let dynamic = DynamicSharedArray::<u32>::get();
        unsafe {
            *ptr.add(value as usize) = value;
            *local.add(value as usize) = 11;
            *dynamic.add(value as usize) = value;
        }
        thread::sync_threads();
        if let Some(v) = out.get_mut(i) {
            let pair = shared::pair::<5>(shared::read_scratch(value as usize));
            let biases = &[2u32, 4, 6];
            *v = pair.x + pair.y + 20 + shared::only_second(value) - value - 777
                + biases[value as usize % 3]
                + shared::read_dynamic(value as usize)
                - value
                + unsafe { *local.add(value as usize) }
                - 11;
        }
    }
}
#[inline(never)]
fn host_value() -> u32 {
    7
}
#[track_caller]
#[inline(never)]
fn caller_line() -> u32 {
    std::panic::Location::caller().line()
}
#[inline(never)]
fn host_line() -> u32 {
    caller_line()
}
fn argument_value(name: &str, default: u32) -> u32 {
    std::env::args()
        .find_map(|arg| arg.strip_prefix(name).map(|value| value.parse().unwrap()))
        .unwrap_or(default)
}
#[inline(never)]
fn closure_value() -> u32 {
    let thunk = || 7u32;
    thunk()
}
pub fn run() {
    assert_eq!(host_cache_sentinel::value(), 37);
    let started = std::time::Instant::now();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let load = std::time::Instant::now();
    let files = std::env::args().any(|argument| argument == "--files");
    let stem = "incremental_modules";
    let first = unsafe {
        if files {
            first::from_module(
                cuda_host::ltoir::load_kernel_cuda_module(
                    &context,
                    stem,
                    first::LoadedModule::MODULE_ID,
                )
                .unwrap(),
            )
            .unwrap()
        } else {
            first::load(&context).unwrap()
        }
    };
    let second = unsafe {
        if files {
            second::from_module(
                cuda_host::ltoir::load_kernel_cuda_module(
                    &context,
                    stem,
                    second::LoadedModule::MODULE_ID,
                )
                .unwrap(),
            )
            .unwrap()
        } else {
            second::load(&context).unwrap()
        }
    };
    println!("module_load_seconds={:.6}", load.elapsed().as_secs_f64());
    let first_config = first
        .prepare_first(cuda_core::LaunchConfig1D::new(1, 32, 128))
        .unwrap();
    let second_config = second
        .prepare_second(cuda_core::LaunchConfig1D::new(1, 32, 128))
        .unwrap();
    let mut a = DeviceBuffer::<u32>::zeroed(&stream, 32).unwrap();
    let mut b = DeviceBuffer::<u32>::zeroed(&stream, 32).unwrap();
    first.first(&stream, &first_config, &mut a).unwrap();
    second.second(&stream, &second_config, &mut b).unwrap();
    let a = a.to_host_vec(&stream).unwrap();
    let b = b.to_host_vec(&stream).unwrap();
    for i in 0..32 {
        assert_eq!(
            a[i],
            i as u32 * 4 + 12 + [1u32, 3, 5][i % 3] + argument_value("--first-delta=", 0)
        );
        assert_eq!(
            b[i],
            i as u32 * 6 + 22 + [2u32, 4, 6][i % 3] + argument_value("--second-delta=", 0)
        );
    }
    assert_eq!(host_value(), argument_value("--host-value=", 7));
    assert_eq!(closure_value(), 7);
    println!("host_caller_line={}", host_line());
    println!(
        "PASS: both modules, shared aggregate device ABI; run_seconds={:.6}",
        started.elapsed().as_secs_f64()
    );
}
