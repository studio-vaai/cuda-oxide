// LOCAL EXPERIMENT for studio-vaai cloth-solver-cuda — DO NOT UPSTREAM.
//
// Two-argument transcendental math intrinsics that don't have a
// `core::intrinsics::*` placeholder (Rust's `f32::atan2` calls a `cmath`
// extern in `std`, which `cuda-device` (no_std) can't reach). These
// stubs route via FQDN dispatch in mir-importer → MIR placeholder call
// → mir-lower → `__nv_*` libdevice symbol, same path as the existing
// `core::intrinsics::sqrtf32` → `__nv_sqrtf`.

/// `atan2(y, x)` for `f32` — lowered to `__nv_atan2f` (libdevice).
///
/// Body is empty (NOT `unreachable!()`): cuda-oxide's collector treats
/// diverging-body intrinsics as placeholders and may skip callers that
/// collapse to panic-only MIR. An empty body keeps callers visible;
/// mir-importer rewrites every call site before this body would ever run.
#[inline(never)]
#[doc(hidden)]
pub fn atan2f(y: f32, x: f32) -> f32 {
    let _ = (y, x);
    0.0
}

/// `atan2(y, x)` for `f64` — lowered to `__nv_atan2` (libdevice).
#[inline(never)]
#[doc(hidden)]
pub fn atan2(y: f64, x: f64) -> f64 {
    let _ = (y, x);
    0.0
}
