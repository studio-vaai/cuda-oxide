// LOCAL EXPERIMENT for studio-vaai cloth-solver-cuda — DO NOT UPSTREAM.
//
// Vector escape-hatch intrinsic for cuda-oxide on Blackwell (sm_120):
// native single-instruction global vector atomic-add via
// `red.global.add.v4.f32` (SASS `REDG.E.ADD.F32x4`, sm_9.0+).
//
// Plain v4 load/store are NOT here anymore — Gap 1 (StoreOp/LoadOp omitted
// `align N` in the LLVM IR export, defeating libNVVM's vector fuser) was
// fixed at the export layer in this same fork. Kernels can now use plain
// `*mut CuSimd<f32, 4>` / `*const CuSimd<f32, 4>` pointer assignments and
// get `st.global.v4.f32` / `ld.global.v4.f32` for free.
//
// Block-scope (smem) vector atomic-add is also not here — ptxas on sm_120
// rewrites both `atomicrmw fadd ... syncscope("block")` and inline-PTX
// `atom.shared.cta.add.f32` to the same `ATOMS.CAST.SPIN` CAS-loop because
// Blackwell has no native f32 shared atomic-add. Codegen can't help.

use crate::atomic::AtomicOrdering;
use crate::cusimd::CuSimd;
use core::cell::UnsafeCell;

// =============================================================================
// Raw FQDN-dispatched intrinsic
//
// Empty body (NOT `unreachable!()`) on purpose: cuda-oxide's collector treats
// `unreachable!()`-bodied functions as intrinsic placeholders and skips them,
// but COLLAPSED-DOWN MIR of any *caller* of a diverging intrinsic also
// becomes ≤2 basic blocks of `panic` calls and the collector skips the caller
// too. Empty body keeps callers visible; mir-importer rewrites every call
// site to `InlineAsmOp` before this body would ever execute.
// =============================================================================

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
// Mirrors `cuda_device::atomic::DeviceAtomicF32`'s shape (`from_ptr` + method
// calls) but operates on `[f32; 4]` 16-byte-aligned slots and lowers to a
// single native `REDG.E.ADD.F32x4` SASS instruction.
//
// `fetch_add` returns `()` rather than the previous `CuSimd<f32, 4>` value:
// the `red` PTX form discards per-element previous values. Use four scalar
// `DeviceAtomicF32::fetch_add` calls if you need them.
//
// No `load` / `store` methods — those would be misleading (plain `LDG.E.128`
// / `STG.E.128` are NOT atomic in the C++ memory model; they're just
// single-transaction). For plain 16-byte coalesced load/store, dereference
// a `*mut CuSimd<f32, 4>` / `*const CuSimd<f32, 4>` directly — the Gap 1
// fix in `dialect-llvm/src/export.rs` makes that lower correctly.
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
