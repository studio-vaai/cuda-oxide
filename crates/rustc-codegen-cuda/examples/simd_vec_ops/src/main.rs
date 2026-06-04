/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Vector load/store lowering test.
//!
//! Each `v_*` kernel does `output[idx] = input[idx]` over a genuine
//! `#[repr(simd)]` vector type `<N x T>`. Because the codegen lowers Vector-ABI
//! types to a real LLVM vector (not a struct-wrapped array `{ [N x T] }`), and
//! a vector type's ABI alignment *is* its width, the whole-vector load/store
//! fuses into a SOUND vectorized PTX instruction (`ld.v2`/`ld.v4`,
//! `st.v2`/`st.v4` — 64/128-bit transactions) instead of N scalar `ld`/`st`.
//!
//! The `scalar_f32x4` kernel uses a plain `#[repr(C)]` array wrapper (the
//! CuSimd shape) and is expected to stay scalar — the contrast that motivates
//! the vector-type path.
//!
//! Combination matrix (all single-instruction hardware vectors, ≤128 bits,
//! plus two oversized cases that split into multiple vectors):
//!
//! | Elem (size) | lanes -> bits                          |
//! |-------------|----------------------------------------|
//! | f32/u32/i32 | x2 -> 64, x4 -> 128                    |
//! | f64/u64/i64 | x2 -> 128   (x4 -> 256 = 2 vectors)    |
//! | u16/i16     | x2 -> 32, x4 -> 64, x8 -> 128          |
//! | wide f32    | x8 -> 256 (2 vectors)                  |
//!
//! Build: `cargo oxide build simd_vec_ops`, then `./analyze.sh`.

#![feature(repr_simd)]

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Defines a `#[repr(simd)]` vector type `$name = <$n x $t>`.
macro_rules! simd_ty {
    ($name:ident, $t:ty, $n:literal) => {
        #[repr(simd)]
        #[derive(Clone, Copy)]
        pub struct $name([$t; $n]);
    };
}

// ---- 32-bit element vectors ----
simd_ty!(F32x2, f32, 2);
simd_ty!(F32x4, f32, 4);
simd_ty!(U32x2, u32, 2);
simd_ty!(U32x4, u32, 4);
simd_ty!(I32x2, i32, 2);
simd_ty!(I32x4, i32, 4);

// ---- 64-bit element vectors ----
simd_ty!(F64x2, f64, 2);
simd_ty!(U64x2, u64, 2);
simd_ty!(I64x2, i64, 2);

// ---- 16-bit element vectors ----
simd_ty!(U16x2, u16, 2);
simd_ty!(U16x4, u16, 4);
simd_ty!(U16x8, u16, 8);
simd_ty!(I16x2, i16, 2);
simd_ty!(I16x4, i16, 4);
simd_ty!(I16x8, i16, 8);

// ---- oversized (>128 bits): must split into multiple vectors ----
simd_ty!(F32x8, f32, 8);
simd_ty!(F64x4, f64, 4);

/// Scalar contrast: a plain `#[repr(C)]` array wrapper (the CuSimd shape,
/// align 4). Expected to stay scalar — its alignment (4) is below its width
/// (16), so the aligned-aggregate gate does not fire.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ArrF32x4([f32; 4]);

/// Aligned wrapper around `[f32; 4]` (align 16 = width). The codegen lowers
/// this to `<4 x float>` (aligned-aggregate gate), so it vectorizes SOUNDLY —
/// the `align 16` it promises is genuinely guaranteed by the type. This is the
/// "wrapper that vectorizes" case, in contrast to `ArrF32x4` above.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct WArrF32x4([f32; 4]);

#[cuda_module]
mod kernels {
    use super::*;

    // Every kernel is `output[idx] = input[idx]`: a whole-vector load followed
    // by a whole-vector store. Only the element type / lane count varies.

    // ===== 32-bit element vectors =====

    #[kernel]
    pub fn v_f32x2(input: &[F32x2], mut output: DisjointSlice<F32x2>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_f32x4(input: &[F32x4], mut output: DisjointSlice<F32x4>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_u32x2(input: &[U32x2], mut output: DisjointSlice<U32x2>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_u32x4(input: &[U32x4], mut output: DisjointSlice<U32x4>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_i32x2(input: &[I32x2], mut output: DisjointSlice<I32x2>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_i32x4(input: &[I32x4], mut output: DisjointSlice<I32x4>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    // ===== 64-bit element vectors =====

    #[kernel]
    pub fn v_f64x2(input: &[F64x2], mut output: DisjointSlice<F64x2>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_u64x2(input: &[U64x2], mut output: DisjointSlice<U64x2>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_i64x2(input: &[I64x2], mut output: DisjointSlice<I64x2>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    // ===== 16-bit element vectors =====

    #[kernel]
    pub fn v_u16x2(input: &[U16x2], mut output: DisjointSlice<U16x2>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_u16x4(input: &[U16x4], mut output: DisjointSlice<U16x4>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_u16x8(input: &[U16x8], mut output: DisjointSlice<U16x8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_i16x2(input: &[I16x2], mut output: DisjointSlice<I16x2>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_i16x4(input: &[I16x4], mut output: DisjointSlice<I16x4>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_i16x8(input: &[I16x8], mut output: DisjointSlice<I16x8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    // ===== oversized: split into multiple vectors =====

    #[kernel]
    pub fn v_f32x8(input: &[F32x8], mut output: DisjointSlice<F32x8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    #[kernel]
    pub fn v_f64x4(input: &[F64x4], mut output: DisjointSlice<F64x4>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    // ===== contrast: repr(C) array wrapper (align 4) stays scalar =====

    #[kernel]
    pub fn scalar_f32x4(input: &[ArrF32x4], mut output: DisjointSlice<ArrF32x4>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }

    // ===== aligned wrapper around [f32;4] (align 16) now vectorizes =====

    #[kernel]
    pub fn w_f32x4(input: &[WArrF32x4], mut output: DisjointSlice<WArrF32x4>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = output.get_mut(idx) {
            *o = input[i];
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================
//
// No GPU launch is required to inspect the generated PTX: `cargo oxide build`
// emits `simd_vec_ops.ptx` containing every kernel above.
fn main() {
    println!("simd_vec_ops: device kernels compiled.");
    println!("Run ./analyze.sh to check that each v_* kernel vectorized.");
}
