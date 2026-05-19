// LOCAL EXPERIMENT for studio-vaai cloth-solver-cuda — DO NOT UPSTREAM.
//
// Vector escape-hatch intrinsics for cuda-oxide on Blackwell (sm_120).
//
// Two flavors:
//
// 1. Raw `st.global.v4.f32` / `ld.global.v4.f32` free functions — work
//    around libNVVM scalarizing `[f32; 4]` stores/loads into 4 × LDG/STG.E.32
//    instead of 1 × LDG/STG.E.128 at 16-byte-aligned sites.
//
// 2. `DeviceAtomicCuSimdF32x4` (atomic-style wrapper type matching the
//    upstream `cuda_device::atomic::DeviceAtomicF32` API) — lowers to native
//    `red.global.add.v4.f32`. SASS `REDG.E.ADD.F32x4`, single instruction,
//    no CAS loop. Available on sm_9.0+ per the CUDA Programming Guide.
//
// Block-scope (smem) atomic-add escape hatch removed — see memory
// `reference_cuda_oxide_codegen_gaps` Gap 2: ptxas on sm_120 emits
// ATOMS.CAST.SPIN for both `atomicrmw fadd ... syncscope("block")` AND
// inline-PTX `atom.shared.cta.add.f32`. The CAS-spin pattern IS the native
// form of f32 shared atomic-add on Blackwell — codegen can't help.

use crate::atomic::AtomicOrdering;
use crate::cusimd::CuSimd;
use core::cell::UnsafeCell;

// =============================================================================
// Raw FQDN-dispatched intrinsics
//
// Bodies are empty (NOT `unreachable!()`): cuda-oxide's collector treats
// `unreachable!()`-bodied functions as intrinsic placeholders and skips them,
// but COLLAPSED-DOWN MIR of any *caller* of a diverging intrinsic also
// becomes ≤2 basic blocks of `panic` calls and the collector skips the caller
// too. Empty bodies keep callers visible; mir-importer rewrites every call
// site to `InlineAsmOp` before these bodies would ever execute.
// =============================================================================

/// `st.global.v4.f32 [p], {a, b, c, d};` — `p` MUST be 16-byte aligned.
#[inline(never)]
pub unsafe fn st_global_v4_f32(p: *mut f32, a: f32, b: f32, c: f32, d: f32) {
    let _ = (p, a, b, c, d);
}

/// `ld.global.v4.f32 {a, b, c, d}, [p];` — `p` MUST be 16-byte aligned.
#[inline(never)]
pub unsafe fn ld_global_v4_f32(p: *const f32) -> [f32; 4] {
    let _ = p;
    [0.0, 0.0, 0.0, 0.0]
}

/// `red.global.add.v4.f32 [p], {a, b, c, d};` — `p` MUST be 16-byte aligned.
///
/// Native single-instruction vector atomic-add on Blackwell (sm_9.0+). SASS
/// opcode: `REDG.E.ADD.F32x4`. Prefer `DeviceAtomicCuSimdF32x4::fetch_add`.
#[inline(never)]
#[doc(hidden)]
pub unsafe fn atomic_add_global_v4_f32(p: *mut f32, a: f32, b: f32, c: f32, d: f32) {
    let _ = (p, a, b, c, d);
}

// =============================================================================
// DeviceAtomicCuSimdF32x4 — vector global atomic-add
//
// Mirrors `cuda_device::atomic::DeviceAtomicF32`s shape (`from_ptr` + method
// calls) but operates on `[f32; 4]` 16-byte-aligned slots and lowers to a
// single native `REDG.E.ADD.F32x4` SASS instruction.
//
// API caveat vs `DeviceAtomicF32::fetch_add`: `fetch_add` here returns `()`
// rather than the previous `CuSimd<f32, 4>` value. The `red` PTX form (which
// the lowering uses) does not produce per-element previous values. Use four
// scalar `DeviceAtomicF32::fetch_add` calls if you need them.
// =============================================================================

/// Device-scope vector atomic over `[f32; 4]` in global memory.
///
/// Lowering: `fetch_add` → `red.global.add.v4.f32`.
///
/// # Safety contract
///
/// * The underlying address must be 16-byte aligned.
/// * Mirrors the alignment/validity/aliasing rules of
///   `DeviceAtomicF32::from_ptr`.
#[repr(transparent)]
pub struct DeviceAtomicCuSimdF32x4 {
    inner: UnsafeCell<[f32; 4]>,
}

unsafe impl Sync for DeviceAtomicCuSimdF32x4 {}

impl DeviceAtomicCuSimdF32x4 {
    /// Create a new atomic with the given initial value.
    pub const fn new(val: [f32; 4]) -> Self {
        DeviceAtomicCuSimdF32x4 {
            inner: UnsafeCell::new(val),
        }
    }

    /// Reinterpret a 16-byte-aligned raw pointer as a vector atomic view.
    ///
    /// # Safety
    ///
    /// * `ptr` MUST be 16-byte aligned.
    /// * Memory at `ptr` must be valid for the entire lifetime `'a`.
    #[inline(always)]
    pub const unsafe fn from_ptr<'a>(ptr: *mut f32) -> &'a Self {
        unsafe { &*(ptr as *const Self) }
    }

    /// Atomically add `val` element-wise via `red.global.add.v4.f32`.
    ///
    /// The ordering argument is accepted for API symmetry with
    /// `DeviceAtomicF32::fetch_add` but does not affect lowering — the `red`
    /// form always emits with PTX-default ordering (relaxed on Blackwell).
    ///
    /// Returns `()` rather than the previous value (see type-level caveat).
    #[inline(always)]
    pub fn fetch_add(&self, val: CuSimd<f32, 4>, order: AtomicOrdering) {
        let _ = order;
        let ptr = self.inner.get() as *mut f32;
        let arr = val.to_array();
        // SAFETY: alignment guaranteed by `from_ptr`/`new`.
        unsafe { atomic_add_global_v4_f32(ptr, arr[0], arr[1], arr[2], arr[3]) }
    }
}
