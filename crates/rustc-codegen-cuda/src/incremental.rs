// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Cached per-module NVVM/LTO compilation. All three cache boundaries
//! precede the expensive operation they memoize; a hit never translates MIR.
use crate::collector::{self, CollectedFunction, CollectionResult};
use crate::device_codegen::{self, DeviceCodegenConfig};
use cuda_artifact_finalizer::{
    DebugPolicy, FinalizationOptions, Finalizer, FinalizerOutput, LlvmNvptxCompiler, NamedInput,
    PtxAssembler,
};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hash::{StableHash, StableHasher};
use rustc_hir::def::DefKind;
use rustc_middle::ty::{EarlyBinder, TyCtxt, TyKind, TypingEnv};
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;
type Error = Box<dyn std::error::Error>;
const MODULE_MARKER: &str = "__cuda_oxide_module_v1";
const NVVM_MODULE_MARKER: &str = "__cuda_oxide_module_nvvm_v1";

fn module_in_namespace(module: &str, namespace: &str) -> bool {
    module == namespace
        || module
            .strip_prefix(namespace)
            .is_some_and(|suffix| suffix.starts_with("::"))
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Cache files are self-checking and published with a single atomic rename.
/// Truncation, stale contents, or interrupted writes cause a miss.
fn read_cache(path: &Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 32 || Sha256::digest(&bytes[32..]).as_slice() != &bytes[..32] {
        return None;
    }
    Some(bytes[32..].to_vec())
}
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    std::fs::create_dir_all(path.parent().ok_or("output has no parent")?)?;
    // Two modules can share one dependency closure and cache key. Give every
    // writer its own temporary path before atomically replacing the entry.
    static NEXT_WRITE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let write = NEXT_WRITE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp.{}.{write}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}
pub(crate) fn write_cache(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut contents = Sha256::digest(bytes).to_vec();
    contents.extend_from_slice(bytes);
    atomic_write(path, &contents)
}
fn pack(metadata: serde_json::Value, payload: &[u8]) -> Result<Vec<u8>, Error> {
    let metadata = serde_json::to_vec(&metadata)?;
    let mut bytes = (metadata.len() as u64).to_le_bytes().to_vec();
    bytes.extend(metadata);
    bytes.extend(payload);
    Ok(bytes)
}
fn unpack(bytes: &[u8]) -> Result<(serde_json::Value, &[u8]), Error> {
    let length =
        u64::from_le_bytes(bytes.get(..8).ok_or("truncated cache header")?.try_into()?) as usize;
    let end = length.checked_add(8).ok_or("cache header overflow")?;
    let metadata = serde_json::from_slice(bytes.get(8..end).ok_or("truncated cache metadata")?)?;
    Ok((metadata, &bytes[end..]))
}
fn cached<F>(root: &Path, stage: &str, key: &str, generate: F) -> Result<(Vec<u8>, bool), Error>
where
    F: FnOnce() -> Result<Vec<u8>, Error>,
{
    let path = root.join(stage).join(key);
    if let Some(bytes) = read_cache(&path) {
        return Ok((bytes, true));
    }
    let bytes = generate()?;
    write_cache(&path, &bytes)?;
    Ok((bytes, false))
}

type CompiledUnit = (Vec<u8>, BTreeSet<String>, [u8; 32]);
struct LinkedUnit {
    stored_cubin: Vec<u8>,
    compilation_phases: Option<serde_json::Value>,
    hit: bool,
    digest_seconds: f64,
    cache_seconds: f64,
    seconds: f64,
}

// Native source-IR compilation, with NVIDIA overrides for selected namespaces.
struct SourceCompiler {
    llvm: LlvmNvptxCompiler,
    assembler: PtxAssembler,
    nvvm_modules: BTreeSet<String>,
}

struct NativeSourceUnit<'a> {
    name: &'a str,
    deps: &'a BTreeSet<String>,
    kernels: &'a BTreeSet<String>,
    semantic_key: &'a str,
}

impl SourceCompiler {
    fn from_env(
        tcx: TyCtxt<'_>,
        finalizer: &Finalizer,
        options: &FinalizationOptions,
        nvvm_modules: BTreeSet<String>,
    ) -> Result<Option<Self>, Error> {
        use cuda_artifact_finalizer::{
            NativeCompilerPreference, SourceCompilerHandshakeV1, rust_llvm_library,
        };
        use reserved_oxide_symbols::{
            SOURCE_COMPILER_HANDSHAKE_ENV, SOURCE_COMPILER_PROVENANCE_ENV,
        };
        let requested = std::env::var("CUDA_OXIDE_NATIVE_COMPILER").ok();
        let preference = NativeCompilerPreference::parse(requested.as_deref())?;
        if preference == NativeCompilerPreference::Nvvm
            || (preference == NativeCompilerPreference::Auto
                && (options.debug_policy() != DebugPolicy::None
                    || options.target().uses_legacy_llvm()))
        {
            if let Ok(expected) = std::env::var(SOURCE_COMPILER_PROVENANCE_ENV)
                && expected != "nvvm"
            {
                return Err("source compiler differs from Cargo's native compiler identity".into());
            }
            return Ok(None);
        }
        let discovered = (|| -> Result<Self, Error> {
            if options.target().uses_legacy_llvm() {
                return Err("LLVM native compilation requires a modern NVVM source dialect (sm_100 or newer)".into());
            }
            if options.debug_policy() != DebugPolicy::None {
                return Err(
                    "LLVM source compilation requires optimized device code without debug info"
                        .into(),
                );
            }
            let cached = std::env::var(SOURCE_COMPILER_HANDSHAKE_ENV)
                .ok()
                .and_then(|json| serde_json::from_str::<SourceCompilerHandshakeV1>(&json).ok())
                .filter(SourceCompilerHandshakeV1::has_consistent_provenance);
            let llvm = LlvmNvptxCompiler::from_path_with_provenance(
                &rust_llvm_library(tcx.sess.opts.sysroot.path())?,
                finalizer.compiler().libdevice_bytes(),
                cached.as_ref().map(|hint| &hint.llvm),
            )?;
            let assembler =
                PtxAssembler::discover_with_provenance(cached.as_ref().map(|hint| &hint.ptxas))?;
            if !assembler.supports_ptx90() {
                return Err("LLVM native compilation requires CUDA 13 or newer ptxas".into());
            }
            let handshake = llvm
                .handshake(&assembler)
                .ok_or("source compiler provenance changed")?;
            if let Ok(expected) = std::env::var(SOURCE_COMPILER_PROVENANCE_ENV)
                && expected != crate::materialize::digest_hex(&handshake.provenance_sha256)
            {
                return Err("source compiler differs from Cargo's native compiler identity".into());
            }
            Ok(Self {
                llvm,
                assembler,
                nvvm_modules,
            })
        })();
        match discovered {
            Ok(compiler) => Ok(Some(compiler)),
            Err(_)
                if preference == NativeCompilerPreference::Auto
                    && std::env::var_os(SOURCE_COMPILER_PROVENANCE_ENV).is_none() =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn link_cached(
        &self,
        cache: &Path,
        sources: &BTreeMap<String, Vec<u8>>,
        unit: NativeSourceUnit<'_>,
        options: &FinalizationOptions,
    ) -> Result<LinkedUnit, Error> {
        let NativeSourceUnit {
            name,
            deps,
            kernels,
            semantic_key,
        } = unit;
        let started = Instant::now();
        let inputs: Vec<_> = deps
            .iter()
            .map(|dep| NamedInput::new(dep, &sources[dep]))
            .collect();
        let llvm_key = self
            .llvm
            .artifact_digest(&inputs, kernels, options)
            .ok_or("LLVM provenance changed")?;
        // The PTX assembler's complete tool/options recipe is part of the
        // native key before compilation, independently of NVIDIA LTO caches.
        let key = self
            .assembler
            .artifact_digest(NamedInput::new(name, digest(&llvm_key).as_bytes()), options)?
            .ok_or("ptxas provenance changed")?;
        let exact_path = cache.join("cubin-v1-llvm-source").join(digest(&key));
        // Use the same verified reachable Rust closure as the NVIDIA route.
        // Changing an unrelated helper in a shared IR unit must not recompile
        // every kernel that happens to use another helper from that unit.
        let identity = serde_json::to_vec(&(deps, semantic_key))?;
        let reachable = self
            .llvm
            .artifact_digest(
                &[NamedInput::new("reachable-source-ir-v1", &identity)],
                kernels,
                options,
            )
            .ok_or("LLVM provenance changed")?;
        let reachable_key = self
            .assembler
            .artifact_digest(
                NamedInput::new(name, digest(&reachable).as_bytes()),
                options,
            )?
            .ok_or("ptxas provenance changed")?;
        let digest_seconds = started.elapsed().as_secs_f64();
        let cache_started = Instant::now();
        let mut migrated = false;
        let mut compilation_phases = None;
        let (stored_cubin, hit) = cached(
            cache,
            "cubin-v2-llvm-reachable",
            &digest(&reachable_key),
            || {
                // Exact source/tool/options migration never adopts an image from
                // a different compiler or from an unverified source closure.
                if let Some(bytes) = read_cache(&exact_path) {
                    migrated = true;
                    return Ok(bytes);
                }
                let llvm = self.llvm.compile_with_report(&inputs, kernels, options)?;
                let assembler_started = Instant::now();
                let report = self
                    .assembler
                    .assemble_ptx_with_report(NamedInput::new(name, &llvm.ptx), options)?;
                compilation_phases = Some(json!({
                    "llvm_parse_merge_seconds": llvm.parse_merge_seconds,
                    "llvm_optimize_seconds": llvm.optimize_seconds,
                    "llvm_emit_ptx_seconds": llvm.emit_ptx_seconds,
                    "ptxas_seconds": assembler_started.elapsed().as_secs_f64(),
                }));
                // A tool mutation while either stage ran must never publish a hit.
                if self.llvm.artifact_digest(&inputs, kernels, options) != Some(llvm_key)
                    || self.assembler.ptxas_digest().is_none()
                {
                    return Err("source compiler provenance changed".into());
                }
                let usage: Vec<_> = report
                    .resource_usage
                    .iter()
                    .map(|u| {
                        (
                            &u.kernel,
                            u.registers,
                            u.stack_frame_bytes,
                            u.spill_store_bytes,
                            u.spill_load_bytes,
                        )
                    })
                    .collect();
                let stored = pack(json!(usage), &report.image)?;
                write_cache(&exact_path, &stored)?;
                Ok(stored)
            },
        )?;
        Ok(LinkedUnit {
            stored_cubin,
            compilation_phases,
            hit: hit || migrated,
            digest_seconds,
            cache_seconds: cache_started.elapsed().as_secs_f64(),
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

fn link_cached_unit(
    cache: &Path,
    units: &BTreeMap<String, CompiledUnit>,
    deps: &BTreeSet<String>,
    semantic_key: &str,
    finalizer: &Finalizer,
    options: &FinalizationOptions,
) -> Result<LinkedUnit, Error> {
    let inputs: Vec<_> = deps
        .iter()
        .map(|dep| NamedInput::new(dep, &units[dep].0))
        .collect();
    let link_started = Instant::now();
    // Keep the original ordered LTOIR inputs for compilation. Cache reuse is
    // governed by the complete reachable Rust function closure, rather than
    // unrelated definitions sharing a helper's LTOIR translation unit.
    let identity = serde_json::to_vec(&(deps, semantic_key))?;
    let link_key = finalizer
        .linker()
        .artifact_digest(
            &[NamedInput::new("reachable-kernel-v1", &identity)],
            options,
            FinalizerOutput::Cubin,
        )
        .ok_or("nvJitLink provenance is unavailable; cannot safely cache")?;
    let link_key = digest(&link_key);
    let digest_seconds = link_started.elapsed().as_secs_f64();
    let cache_started = Instant::now();
    let mut migrated = false;
    let (stored_cubin, hit) = cached(cache, "cubin-v4-reachable-functions", &link_key, || {
        // Migrate only an image whose exact current LTOIR inputs are verified.
        let digested_inputs: Vec<_> = deps
            .iter()
            .map(|dep| NamedInput::new(dep, &units[dep].2))
            .collect();
        let old_key = finalizer
            .linker()
            .artifact_digest(&digested_inputs, options, FinalizerOutput::Cubin)
            .ok_or("nvJitLink provenance changed")?;
        let exact_inputs_cache = cache
            .join("cubin-v3-digested-inputs")
            .join(digest(&old_key));
        if let Some(bytes) = read_cache(&exact_inputs_cache) {
            migrated = true;
            return Ok(bytes);
        }
        let old_key = finalizer
            .linker()
            .artifact_digest(&inputs, options, FinalizerOutput::Cubin)
            .ok_or("nvJitLink provenance changed")?;
        if let Some(bytes) = read_cache(&cache.join("cubin-v2").join(digest(&old_key))) {
            migrated = true;
            return Ok(bytes);
        }
        let report = finalizer.link_ltoir_with_report(&inputs, options, FinalizerOutput::Cubin)?;
        let usage: Vec<_> = report
            .resource_usage
            .iter()
            .map(|usage| {
                (
                    &usage.kernel,
                    usage.registers,
                    usage.stack_frame_bytes,
                    usage.spill_store_bytes,
                    usage.spill_load_bytes,
                )
            })
            .collect();
        let stored = pack(json!(usage), &report.image)?;
        // Preserve the exact ordered LTOIR/tool/options identity as well as
        // the reachable Rust closure. A backend or host-only compiler change
        // can invalidate the closure's fence while producing identical LTOIR.
        // Only a fresh native compile populates this key: a semantic cache hit
        // may have been produced from different, unreachable LTOIR contents.
        write_cache(&exact_inputs_cache, &stored)?;
        Ok(stored)
    })?;
    let hit = hit || migrated;
    let cache_seconds = cache_started.elapsed().as_secs_f64();
    Ok(LinkedUnit {
        stored_cubin,
        compilation_phases: None,
        hit,
        digest_seconds,
        cache_seconds,
        seconds: link_started.elapsed().as_secs_f64(),
    })
}

// Timing records only advise queue order. They never enter cache identity or
// compiler options, and invalid/missing/unwritable records are ignored.
fn native_cost_path(
    cache: &Path,
    name: &str,
    finalizer: &Finalizer,
    options: &FinalizationOptions,
) -> Option<PathBuf> {
    let key = finalizer.linker().artifact_digest(
        &[NamedInput::new("native-cost-v1", name.as_bytes())],
        options,
        FinalizerOutput::Cubin,
    )?;
    Some(cache.join("native-link-cost-v1").join(digest(&key)))
}

fn recorded_native_cost(
    cache: &Path,
    name: &str,
    finalizer: &Finalizer,
    options: &FinalizationOptions,
) -> Option<f64> {
    let bytes = read_cache(&native_cost_path(cache, name, finalizer, options)?)?;
    decode_native_cost(&bytes)
}

fn decode_native_cost(bytes: &[u8]) -> Option<f64> {
    let seconds: f64 = serde_json::from_slice(bytes).ok()?;
    (seconds.is_finite() && seconds > 0.0 && seconds <= 3600.0).then_some(seconds)
}

fn timing_priorities(
    estimates: &BTreeMap<String, u64>,
    costs: &BTreeMap<String, f64>,
) -> BTreeMap<String, u64> {
    let mut ratios: Vec<_> = costs
        .iter()
        .filter_map(|(name, seconds)| {
            estimates
                .get(name)
                .filter(|cost| **cost != 0)
                .map(|estimate| seconds / *estimate as f64)
        })
        .collect();
    if ratios.is_empty() {
        return estimates.clone();
    }
    ratios.sort_by(f64::total_cmp);
    let scale = ratios[ratios.len() / 2];
    estimates
        .iter()
        .map(|(name, estimate)| {
            let seconds = costs.get(name).copied().unwrap_or(*estimate as f64 * scale);
            (name.clone(), (seconds * 1_000_000.0).ceil() as u64)
        })
        .collect()
}

/// Finalization operates on immutable LTOIR and cannot access the Rust session.
/// Reserve Cargo jobserver tokens before starting extra workers; the first
/// worker uses this rustc process's existing token while the main thread waits.
struct NativeLinkInputs<'a> {
    closure_keys: &'a BTreeMap<String, String>,
    units: &'a BTreeMap<String, CompiledUnit>,
    sources: &'a BTreeMap<String, Vec<u8>>,
    kernels: &'a BTreeMap<String, BTreeSet<String>>,
}

fn link_native_modules(
    jobs: &[(&String, &BTreeSet<String>)],
    inputs: NativeLinkInputs<'_>,
    source_compiler: Option<&SourceCompiler>,
    cache: &Path,
    finalizer: &Finalizer,
    options: &FinalizationOptions,
    generate_host: impl FnOnce(),
) -> Result<BTreeMap<String, LinkedUnit>, Error> {
    let NativeLinkInputs {
        closure_keys,
        units,
        sources,
        kernels,
    } = inputs;
    parallel_finalization(
        jobs.len(),
        "native",
        |index| {
            let (name, deps) = jobs[index];
            let result = if let Some(compiler) =
                source_compiler.filter(|compiler| !compiler.nvvm_modules.contains(name))
            {
                compiler.link_cached(
                    cache,
                    sources,
                    NativeSourceUnit {
                        name,
                        deps,
                        kernels: &kernels[name],
                        semantic_key: &closure_keys[name],
                    },
                    options,
                )
            } else {
                link_cached_unit(cache, units, deps, &closure_keys[name], finalizer, options)
            }
            .map_err(|error| format!("{name}: {error}"))?;
            if !result.hit
                && result.seconds > 0.0
                && let Some(path) = native_cost_path(cache, name, finalizer, options)
                && let Ok(bytes) = serde_json::to_vec(&result.seconds)
            {
                // Advisory metadata must not turn a successful native link into a failure.
                let _ = write_cache(&path, &bytes);
            }
            eprintln!(
                "[device-modules] {name}: cubin {} ({:.3}s)",
                if result.hit { "hit" } else { "linked" },
                result.seconds
            );
            Ok((name.clone(), result))
        },
        generate_host,
    )
    .map(|results| results.into_iter().collect())
}

/// Immutable compiler inputs can run off the rustc thread. The callback remains
/// on that thread, and each extra worker holds a Cargo jobserver token. Drain all
/// started jobs before returning an error or publishing any module selector.
fn parallel_finalization<T: Send>(
    count: usize,
    stage: &str,
    finalize: impl Fn(usize) -> Result<T, String> + Sync,
    main_thread: impl FnOnce(),
) -> Result<Vec<T>, Error> {
    if count == 0 {
        main_thread();
        return Ok(Vec::new());
    }
    let limit = match std::env::var("CUDA_OXIDE_LINK_JOBS") {
        Ok(value) => value.parse::<std::num::NonZeroUsize>()?.get(),
        // Large GPU crates benefit from more than eight independent links.
        // Cargo tokens remain the hard resource limit; cap the automatic
        // budget so each nvJitLink instance can also parallelize internally.
        Err(_) => std::thread::available_parallelism().map_or(8, |cpus| cpus.get().min(16)),
    }
    .min(count);
    let client = rustc_data_structures::jobserver::client();
    let mut tokens = Vec::new();
    for _ in 1..limit {
        match client.try_acquire() {
            Ok(Some(token)) => tokens.push(token),
            _ => break,
        }
    }
    let workers = tokens.len() + 1;
    eprintln!("[device-modules] {stage} finalization: {workers} worker(s)");
    finalization_workers(count, workers, finalize, main_thread)
}

fn finalization_workers<T: Send>(
    count: usize,
    workers: usize,
    finalize: impl Fn(usize) -> Result<T, String> + Sync,
    main_thread: impl FnOnce(),
) -> Result<Vec<T>, Error> {
    debug_assert!(workers != 0);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let (send, receive) = std::sync::mpsc::channel();
    let completed = std::thread::scope(|scope| {
        for _ in 0..workers {
            let send = send.clone();
            let finalize = &finalize;
            let next = &next;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if index >= count {
                        break;
                    }
                    send.send((index, finalize(index)))
                        .expect("finalization receiver exists");
                }
            });
        }
        drop(send);
        main_thread();
        receive.into_iter().collect::<Vec<_>>()
    });
    // Restore source order for deterministic manifests and input ordering even
    // when workers complete in another order.
    let mut ordered: Vec<_> = (0..count).map(|_| None).collect();
    for (index, result) in completed {
        ordered[index] = Some(result);
    }
    ordered
        .into_iter()
        .map(|result| {
            result
                .ok_or("missing finalization result")?
                .map_err(|error| error.into())
        })
        .collect()
}

struct PendingLtoUnit {
    name: String,
    ir: Vec<u8>,
    lto_key: String,
    dependencies: BTreeSet<String>,
    timing: serde_json::Value,
    frontend_seconds: f64,
}

fn compile_lto_unit(
    job: &PendingLtoUnit,
    cache: &Path,
    finalizer: &Finalizer,
    options: &FinalizationOptions,
) -> Result<(String, CompiledUnit, serde_json::Value), Error> {
    let started = Instant::now();
    let (lto, hit) = cached(cache, "ltoir", &job.lto_key, || {
        Ok(finalizer
            .compiler()
            .compile_nvvm_ir_to_ltoir(&job.name, &job.ir, options)?)
    })?;
    let seconds = started.elapsed().as_secs_f64();
    let mut timing = job.timing.clone();
    timing["ltoir_hit"] = json!(hit);
    timing["ltoir_seconds"] = json!(seconds);
    timing["seconds"] = json!(job.frontend_seconds + seconds);
    eprintln!(
        "[device-modules] {}: LTOIR {} ({seconds:.3}s)",
        job.name,
        if hit { "hit" } else { "compiled" }
    );
    let digest = Sha256::digest(&lto).into();
    Ok((
        job.name.clone(),
        (lto, job.dependencies.clone(), digest),
        timing,
    ))
}

fn qualified_module(tcx: TyCtxt<'_>, id: DefId) -> String {
    let path = tcx.def_path_str(id);
    if id.is_local() {
        let crate_name = tcx.crate_name(id.krate);
        if path.is_empty() {
            crate_name.to_string()
        } else {
            format!("{crate_name}::{path}")
        }
    } else {
        path
    }
}

fn module_owner(tcx: TyCtxt<'_>, id: DefId, marked: &BTreeSet<String>) -> String {
    let mut current = tcx.opt_parent(id);
    let mut nearest = None;
    while let Some(parent) = current {
        if tcx.def_kind(parent) == DefKind::Mod {
            let name = qualified_module(tcx, parent);
            if marked.contains(&name) {
                return name;
            }
            if nearest.is_none() {
                nearest = Some(name);
            }
        }
        current = tcx.opt_parent(parent);
    }
    nearest.unwrap_or_else(|| tcx.crate_name(id.krate).to_string())
}

pub(crate) fn associated_has_definition(tcx: TyCtxt<'_>, id: DefId) -> bool {
    match tcx.hir_node_by_def_id(id.expect_local()) {
        rustc_hir::Node::ImplItem(_) => true,
        rustc_hir::Node::TraitItem(item) => matches!(
            item.kind,
            rustc_hir::TraitItemKind::Type(_, Some(_))
                | rustc_hir::TraitItemKind::Const(_, Some(_))
        ),
        _ => false,
    }
}

/// The local lowering pass cannot see callers in other units. Conservatively
/// share the maximum declared dynamic-shared alignment across the artifact.
/// Overalignment preserves every contract and lets helpers remain reusable.
fn dynamic_shared_alignment<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
) -> Result<u64, Error> {
    let mut alignment = 0;
    for function in functions {
        let body = tcx.instance_mir(function.instance.def);
        for block in body.basic_blocks.iter() {
            let rustc_middle::mir::TerminatorKind::Call { func, .. } = &block.terminator().kind
            else {
                continue;
            };
            let ty = tcx.instantiate_and_normalize_erasing_regions(
                function.instance.args,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, func.ty(&body.local_decls, tcx)),
            );
            let TyKind::FnDef(id, args) = ty.kind() else {
                continue;
            };
            if tcx.def_path_str(*id) != "cuda_device::shared::__dynamic_shared_alignment" {
                continue;
            }
            let args = args
                .no_bound_vars()
                .ok_or("unresolved dynamic shared alignment binder")?;
            let value = args
                .first()
                .and_then(|arg| arg.as_const())
                .and_then(|value| value.try_to_target_usize(tcx))
                .ok_or("unresolved dynamic shared alignment constant")?;
            alignment = alignment.max(value);
        }
    }
    Ok(alignment)
}

/// Conservative type/constant metadata fence. Function bodies are deliberately
/// excluded: editing a kernel must not invalidate every other module. Until
/// dependency capture is finer, any local type/constant or external crate
/// metadata change invalidates all units. Rust-inlined helpers are in body MIR.
pub(crate) fn metadata_fingerprint(tcx: TyCtxt<'_>) -> String {
    metadata_fingerprint_with_spans(tcx, false)
}
pub(crate) fn metadata_fingerprint_with_spans(tcx: TyCtxt<'_>, spans: bool) -> String {
    tcx.with_stable_hashing_context(|mut hcx| {
        let mut hasher = StableHasher::new();
        hcx.while_hashing_spans(spans, |hcx| {
            for &krate in tcx.crates(()) {
                tcx.crate_hash(krate).stable_hash(hcx, &mut hasher);
            }
            for id in tcx.hir_crate_items(()).definitions() {
                let id = id.to_def_id();
                match tcx.def_kind(id) {
                    DefKind::Struct | DefKind::Enum | DefKind::Union => {
                        let adt = tcx.adt_def(id);
                        adt.variants().stable_hash(hcx, &mut hasher);
                        adt.repr().stable_hash(hcx, &mut hasher);
                    }
                    DefKind::Field | DefKind::TyAlias => {
                        tcx.type_of(id).stable_hash(hcx, &mut hasher);
                    }
                    DefKind::AssocTy if associated_has_definition(tcx, id) => {
                        tcx.type_of(id).stable_hash(hcx, &mut hasher);
                    }
                    DefKind::Const { .. } | DefKind::AnonConst => {
                        if tcx.is_mir_available(id) {
                            tcx.mir_for_ctfe(id).stable_hash(hcx, &mut hasher);
                        }
                    }
                    DefKind::AssocConst { .. } if associated_has_definition(tcx, id) => {
                        if tcx.is_mir_available(id) {
                            tcx.mir_for_ctfe(id).stable_hash(hcx, &mut hasher);
                        }
                    }
                    DefKind::Fn | DefKind::AssocFn
                        if tcx.is_const_fn(id) && tcx.is_mir_available(id) =>
                    {
                        tcx.mir_for_ctfe(id).stable_hash(hcx, &mut hasher);
                    }
                    _ => {}
                }
            }
        });
        format!("{:?}", hasher.finish::<Fingerprint>())
    })
}

fn input_fingerprint<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
    declarations: &HashSet<String>,
    externs: &[collector::DeviceExternDecl],
    metadata: &str,
    name: &str,
) -> String {
    let mir = tcx.with_stable_hashing_context(|mut hcx| {
        let mut hasher = StableHasher::new();
        // Source positions affect debug output, but must not invalidate a
        // position-independent release unit when another module grows.
        let spans = device_codegen::device_debug_kind(tcx.sess.opts.debuginfo)
            != llvm_export::export::DebugKind::Off;
        hcx.while_hashing_spans(spans, |hcx| {
            for function in functions {
                function.instance.stable_hash(hcx, &mut hasher);
                function.export_name.stable_hash(hcx, &mut hasher);
                function.is_kernel.stable_hash(hcx, &mut hasher);
                let body = tcx.instance_mir(function.instance.def);
                if declarations.contains(&function.export_name) {
                    body.arg_count.stable_hash(hcx, &mut hasher);
                    for local in body.local_decls.iter().take(body.arg_count + 1) {
                        local.ty.stable_hash(hcx, &mut hasher);
                    }
                } else {
                    // Earlier lowering omitted `#[inline(never)]`. Invalidate
                    // only affected definitions so old cached IR cannot silently
                    // discard the newly preserved policy. Declaration-only
                    // users reuse their IR and relink with the new helper LTOIR.
                    if matches!(
                        tcx.codegen_fn_attrs(function.instance.def_id()).inline,
                        rustc_hir::attrs::InlineAttr::Never
                    ) {
                        "device-noinline-v1".stable_hash(hcx, &mut hasher);
                    }
                    // Plain #[inline] was previously omitted from device IR.
                    // Invalidate these definitions while leaving declaration-only
                    // callers and definitions with other policies reusable.
                    if matches!(
                        tcx.codegen_fn_attrs(function.instance.def_id()).inline,
                        rustc_hir::attrs::InlineAttr::Hint
                    ) {
                        "device-inlinehint-v1".stable_hash(hcx, &mut hasher);
                    }
                    body.stable_hash(hcx, &mut hasher);
                    if tcx.is_mir_available(function.instance.def_id()) {
                        tcx.promoted_mir(function.instance.def_id())
                            .stable_hash(hcx, &mut hasher);
                    }
                    tcx.body_codegen_attrs(function.instance.def_id())
                        .stable_hash(hcx, &mut hasher);
                }
            }
            for external in externs {
                external.export_name.stable_hash(hcx, &mut hasher);
                tcx.fn_sig(external.def_id).stable_hash(hcx, &mut hasher);
                tcx.body_codegen_attrs(external.def_id)
                    .stable_hash(hcx, &mut hasher);
            }
        });
        format!("{:?}", hasher.finish::<Fingerprint>())
    });
    let mut env: Vec<_> = std::env::vars_os()
        .filter(|(key, _)| {
            let key = key.to_string_lossy();
            key.starts_with("CUDA_OXIDE_")
                && !matches!(
                    key.as_ref(),
                    "CUDA_OXIDE_KERNELS_ONLY"
                        | "CUDA_OXIDE_LINK_JOBS"
                        | "CUDA_OXIDE_NATIVE_LINK_ORDER"
                        | "CUDA_OXIDE_INTERNAL_RUSTC_WRAPPER"
                        | "CUDA_OXIDE_UPSTREAM_RUSTC_WRAPPER"
                        | "CUDA_OXIDE_HOST_KEY_TRACE"
                        | crate::materialize::MATERIALIZER_HANDSHAKE_ENV
                )
        })
        .map(|(key, value)| {
            (
                key.as_encoded_bytes().to_vec(),
                value.as_encoded_bytes().to_vec(),
            )
        })
        .collect();
    env.sort();
    let mut cfg: Vec<_> = tcx.sess.config.iter().map(|x| format!("{x:?}")).collect();
    cfg.sort();
    let mut options = tcx.sess.opts.clone();
    options.crate_types.clear();
    options.output_types = rustc_session::config::OutputTypes::new(&[]);
    digest(
        json!([
            // Entry-block allocas and unnamed local slots change emitted device
            // IR even when source MIR is unchanged.
            "device-module-input-v5-unnamed-locals",
            llvm_export::export::CONVERGENCE_POLICY_VERSION,
            name,
            metadata,
            mir,
            env,
            cfg,
            format!("{:?}", options.dep_tracking_hash(false)),
            tcx.sess.target.llvm_target.as_ref(),
            tcx.sess.target.data_layout.as_ref()
        ])
        .to_string()
        .as_bytes(),
    )
}

pub(crate) fn compile<'tcx>(
    tcx: TyCtxt<'tcx>,
    collection: &CollectionResult<'tcx>,
    config: &DeviceCodegenConfig,
    generate_host: impl FnOnce(),
) -> Result<PathBuf, Error> {
    let started = Instant::now();
    if !collection
        .functions
        .iter()
        .any(|function| function.is_kernel)
    {
        generate_host();
        return crate::write_filtered_artifact_anchor_object(
            &config.output_dir,
            &config.output_name,
            tcx.sess.target.llvm_target.as_ref(),
        );
    }
    if collection.requires_ptx_bundle_merge {
        return Err(
            "incremental modules do not yet support cross-crate generic kernel bundle merging"
                .into(),
        );
    }
    if device_codegen::has_device_globals(tcx, &collection.functions) {
        return Err("incremental modules cannot duplicate ordinary device globals across cubins; use a shared buffer or the package compilation mode".into());
    }
    let marked: BTreeSet<_> = tcx
        .hir_crate_items(())
        .definitions()
        .filter_map(|id| {
            let id = id.to_def_id();
            (tcx.opt_item_name(id)
                .is_some_and(|n| n.as_str() == MODULE_MARKER))
            .then(|| qualified_module(tcx, tcx.parent(id)))
        })
        .collect();
    let nvvm_namespaces: BTreeSet<_> = tcx
        .hir_crate_items(())
        .definitions()
        .filter_map(|id| {
            let id = id.to_def_id();
            tcx.opt_item_name(id)
                .is_some_and(|n| n.as_str() == NVVM_MODULE_MARKER)
                .then(|| qualified_module(tcx, tcx.parent(id)))
        })
        .collect();
    let kernel_owners: BTreeSet<_> = collection
        .functions
        .iter()
        .filter(|f| f.is_kernel)
        .map(|f| module_owner(tcx, f.instance.def_id(), &marked))
        .collect();
    let nvvm_modules = kernel_owners
        .iter()
        .filter(|name| {
            nvvm_namespaces
                .iter()
                .any(|namespace| module_in_namespace(name, namespace))
        })
        .cloned()
        .collect();
    let mut owners = HashMap::new();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, function) in collection.functions.iter().enumerate() {
        let owner = module_owner(tcx, function.instance.def_id(), &marked);
        // Referencing a helper must not retain unrelated kernel entries from
        // its module through llvm.used. Helpers in kernel-bearing modules get
        // a dependency-only unit; the executable still has the module's ID.
        let unit = if !function.is_kernel && kernel_owners.contains(&owner) {
            format!("{owner}::$device_functions")
        } else {
            owner
        };
        owners.insert(function.instance, unit.clone());
        groups.entry(unit).or_default().push(index);
    }
    let cache = config.output_dir.join("cache/modules/v1");
    let finalizer = crate::materialize::native_finalizer_from_env()?;
    let compiler_provenance = finalizer
        .compiler()
        .provenance_digest()
        .ok_or("libNVVM provenance is unavailable; cannot safely cache")?;
    let alignment = dynamic_shared_alignment(tcx, &collection.functions)?;
    let metadata = format!(
        "{}:{}:{alignment}",
        metadata_fingerprint(tcx),
        digest(&compiler_provenance)
    );
    let target = std::env::var("CUDA_OXIDE_TARGET")
        .map_err(|_| "native modules require a GPU or --arch/CUDA_OXIDE_TARGET for cross-compilation; use --no-incremental-modules for package compilation")?;
    let debug = device_codegen::device_debug_kind(tcx.sess.opts.debuginfo);
    let policy = match debug {
        llvm_export::export::DebugKind::Off => DebugPolicy::None,
        llvm_export::export::DebugKind::LineTables => DebugPolicy::LineTables,
        llvm_export::export::DebugKind::Full => DebugPolicy::Full,
    };
    let options = FinalizationOptions::new(target.parse::<libnvvm_sys::CudaArch>()?)
        .with_fma_contraction(std::env::var_os("CUDA_OXIDE_NO_FMA").is_none())
        .with_debug_policy(policy);
    let source_compiler = SourceCompiler::from_env(tcx, &finalizer, &options, nvvm_modules)?;
    let mut source_units = BTreeMap::new();
    let mut unit_dependencies = BTreeMap::new();
    let mut unit_kernels = BTreeMap::new();
    let mut units: BTreeMap<String, CompiledUnit> = BTreeMap::new();
    let mut timings = Vec::new();
    let mut pending_lto = Vec::new();
    let mut bounds_by_kernel = BTreeMap::new();
    let mut closure_keys = BTreeMap::new();
    let mut unit_costs = BTreeMap::new();
    let order = std::env::var("CUDA_OXIDE_NATIVE_LINK_ORDER");
    let (estimate_costs, use_history) = match order.as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("history") => (true, true),
        Ok("instructions") => (true, false),
        Ok("bytes") => (false, false),
        _ => {
            return Err(
                "CUDA_OXIDE_NATIVE_LINK_ORDER must be history, instructions or bytes".into(),
            );
        }
    };
    let mut priority_seconds = 0.0;
    for (name, indices) in &groups {
        let unit_started = Instant::now();
        let mut roots: Vec<_> = indices
            .iter()
            .map(|&i| collection.functions[i].clone())
            .collect();
        roots.sort_by(|a, b| a.export_name.cmp(&b.export_name));
        unit_kernels.insert(
            name.clone(),
            roots
                .iter()
                .filter(|root| root.is_kernel)
                .map(|root| root.export_name.clone())
                .collect::<BTreeSet<_>>(),
        );
        let reachable = collector::collect_from_roots(tcx, &roots);
        let reachable_instances: HashSet<_> =
            reachable.functions.iter().map(|f| f.instance).collect();
        let mut functions: Vec<_> = collection
            .functions
            .iter()
            .filter(|f| {
                reachable_instances.contains(&f.instance)
                    && (!f.is_kernel || roots.iter().any(|r| r.instance == f.instance))
            })
            .cloned()
            .collect();
        functions.sort_by(|a, b| a.export_name.cmp(&b.export_name));
        let declarations: HashSet<_> = functions
            .iter()
            .filter(|f| !roots.iter().any(|r| r.instance == f.instance))
            .map(|f| f.export_name.clone())
            .collect();
        let dependencies: BTreeSet<_> = functions
            .iter()
            .map(|f| owners[&f.instance].clone())
            .collect();
        let key = input_fingerprint(
            tcx,
            &functions,
            &declarations,
            &reachable.device_externs,
            &metadata,
            name,
        );
        if roots.iter().any(|root| root.is_kernel) {
            closure_keys.insert(
                name.clone(),
                input_fingerprint(
                    tcx,
                    &functions,
                    &HashSet::new(),
                    &reachable.device_externs,
                    &metadata,
                    name,
                ),
            );
        }
        let ir_started = Instant::now();
        let (stored_ir, ir_hit) = cached(&cache, "nvvm-v2", &key, || {
            let local = DeviceCodegenConfig {
                minimum_dynamic_shared_alignment: alignment,
                output_dir: cache.join("work").join(&key),
                output_name: format!("unit_{}", &digest(name.as_bytes())[..16]),
                verbose: config.verbose,
                dump_rustc_mir: config.dump_rustc_mir,
                dump_mir_dialect: config.dump_mir_dialect,
                dump_llvm_dialect: config.dump_llvm_dialect,
            };
            let result = device_codegen::generate_device_code_partition(
                tcx,
                &functions,
                &declarations,
                &reachable.device_externs,
                &local,
            )?;
            if result.target != target {
                return Err("partition target disagrees with finalizer target".into());
            }
            let bounds: BTreeMap<_, _> = result
                .kernel_launch_bounds
                .iter()
                .map(|(name, bounds)| (name, (bounds.max_threads, bounds.min_blocks)))
                .collect();
            pack(
                json!(bounds),
                &result.artifact.ok_or("missing NVVM artifact")?.bytes,
            )
        })?;
        let (bounds, ir) = unpack(&stored_ir)?;
        if source_compiler.is_some() {
            source_units.insert(name.clone(), ir.to_vec());
        }
        if estimate_costs {
            let priority_started = Instant::now();
            unit_costs.insert(name.clone(), crate::native_link_priority::Unit::from_ir(ir));
            priority_seconds += priority_started.elapsed().as_secs_f64();
        }
        let bounds: BTreeMap<String, (u32, Option<u32>)> = serde_json::from_value(bounds)?;
        bounds_by_kernel.extend(bounds.into_iter().map(|(name, (max_threads, min_blocks))| {
            (
                name,
                mir_importer::KernelLaunchBounds {
                    max_threads,
                    min_blocks,
                },
            )
        }));
        unit_dependencies.insert(name.clone(), dependencies.clone());
        let ir_seconds = ir_started.elapsed().as_secs_f64();
        let compiler_key = finalizer
            .compiler()
            .artifact_digest(name, ir, &options)
            .ok_or("libNVVM provenance is unavailable; cannot safely cache")?;
        let lto_key = digest(&compiler_key);
        let lto_started = Instant::now();
        let mut timing = json!({"module":name, "definitions":roots.len(), "declarations":declarations.len(),
            "nvvm_cache_key":key,"ltoir_cache_key":lto_key,
            "nvvm_hit":ir_hit,"nvvm_seconds":ir_seconds});
        if let Some(lto) = read_cache(&cache.join("ltoir").join(&lto_key)) {
            timing["ltoir_hit"] = json!(true);
            timing["ltoir_seconds"] = json!(lto_started.elapsed().as_secs_f64());
            timing["seconds"] = json!(unit_started.elapsed().as_secs_f64());
            timings.push(timing);
            let digest = Sha256::digest(&lto).into();
            units.insert(name.clone(), (lto, dependencies, digest));
        } else {
            pending_lto.push(PendingLtoUnit {
                name: name.clone(),
                ir: ir.to_vec(),
                lto_key,
                dependencies,
                timing,
                frontend_seconds: unit_started.elapsed().as_secs_f64(),
            });
        }
        eprintln!(
            "[device-modules] {name}: NVVM {} ({ir_seconds:.3}s)",
            if ir_hit { "hit" } else { "compiled" }
        );
    }
    if let Some(compiler) = &source_compiler {
        let mut required = BTreeSet::new();
        for name in &compiler.nvvm_modules {
            if unit_kernels.get(name).is_none_or(BTreeSet::is_empty) {
                return Err(format!(
                    "NVIDIA compiler policy names a module without kernels: {name}"
                )
                .into());
            }
            required.extend(unit_dependencies[name].iter().cloned());
        }
        let mut nvidia = Vec::new();
        for mut job in pending_lto {
            if required.contains(&job.name) {
                nvidia.push(job);
            } else {
                job.timing["ltoir_skipped"] = json!(true);
                job.timing["ltoir_hit"] = serde_json::Value::Null;
                job.timing["ltoir_seconds"] = json!(0.0);
                job.timing["seconds"] = json!(job.frontend_seconds);
                timings.push(job.timing);
                // This unit supplies source IR only. NVIDIA native jobs can
                // reference only the validated required set above.
                units.insert(job.name, (Vec::new(), job.dependencies, [0; 32]));
            }
        }
        pending_lto = nvidia;
    }
    let compiled = parallel_finalization(
        pending_lto.len(),
        "LTOIR",
        |index| {
            let job = &pending_lto[index];
            compile_lto_unit(job, &cache, &finalizer, &options)
                .map_err(|error| format!("{}: {error}", job.name))
        },
        || {},
    )?;
    for (name, unit, timing) in compiled {
        units.insert(name, unit);
        timings.push(timing);
    }
    timings.sort_by(|a, b| a["module"].as_str().cmp(&b["module"].as_str()));
    // File loaders already consume the manifest. Omitting embedded payloads
    // avoids copying every cached cubin into a new host object on each edit.
    let files_only = std::env::var_os("CUDA_OXIDE_MODULE_FILES_ONLY").is_some();
    let mut link_jobs = BTreeMap::new();
    for (name, indices) in &groups {
        if !indices
            .iter()
            .any(|&index| collection.functions[index].is_kernel)
        {
            continue;
        }
        // collect_from_roots already visits every transitive function callee.
        // Expanding whole helper units here would add dependencies of their
        // unrelated definitions, often pulling most of the crate into a link.
        link_jobs.insert(name.clone(), units[name].1.clone());
    }
    if let Some(compiler) = &source_compiler
        && let Some(name) = compiler
            .nvvm_modules
            .iter()
            .find(|name| !link_jobs.contains_key(*name))
    {
        return Err(
            format!("NVIDIA compiler policy names a module without kernels: {name}").into(),
        );
    }
    let mut priorities = BTreeMap::new();
    let priority_started = Instant::now();
    for (name, deps) in &link_jobs {
        if !estimate_costs {
            break;
        }
        let kernels: Vec<_> = groups[name]
            .iter()
            .map(|&index| &collection.functions[index])
            .filter(|function| function.is_kernel)
            .map(|function| function.export_name.clone())
            .collect();
        if let Some(cost) = crate::native_link_priority::estimate(
            deps.iter().map(|name| &unit_costs[name]),
            &kernels,
        ) {
            priorities.insert(name.clone(), cost);
        }
    }
    let recorded_costs: BTreeMap<_, _> = if use_history {
        link_jobs
            .keys()
            .filter_map(|name| {
                recorded_native_cost(&cache, name, &finalizer, &options)
                    .map(|seconds| (name.clone(), seconds))
            })
            .collect()
    } else {
        BTreeMap::new()
    };
    if !recorded_costs.is_empty() {
        priorities = timing_priorities(&priorities, &recorded_costs);
        eprintln!(
            "[device-modules] native scheduling: {} recorded module cost(s)",
            recorded_costs.len()
        );
    }
    // Shared units include unrelated helpers. Estimate reachable instructions
    // and repeated inlining to start expensive links before short kernels.
    // Input ordering, cache identity, and publication remain unchanged.
    let mut ordered_jobs: Vec<_> = link_jobs.iter().collect();
    ordered_jobs.sort_by_cached_key(|(name, deps)| {
        std::cmp::Reverse(priorities.get(*name).copied().unwrap_or_else(|| {
            deps.iter()
                .map(|name| source_units.get(name).map_or(units[name].0.len(), Vec::len) as u64)
                .sum()
        }))
    });
    eprintln!(
        "[device-modules] native scheduling: {} estimated module(s) ({:.3}s)",
        priorities.len(),
        priority_seconds + priority_started.elapsed().as_secs_f64()
    );
    let linked_modules = link_native_modules(
        &ordered_jobs,
        NativeLinkInputs {
            closure_keys: &closure_keys,
            units: &units,
            sources: &source_units,
            kernels: &unit_kernels,
        },
        source_compiler.as_ref(),
        &cache,
        &finalizer,
        &options,
        generate_host,
    )?;
    let mut blob = Vec::new();
    let mut modules = BTreeMap::new();
    let package = std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| config.output_name.clone());
    for (name, indices) in &groups {
        let kernels: Vec<_> = indices
            .iter()
            .map(|&i| &collection.functions[i])
            .filter(|f| f.is_kernel)
            .collect();
        if kernels.is_empty() {
            continue;
        } // Helpers are inputs, never executable modules.
        let deps = &link_jobs[name];
        let linked = &linked_modules[name];
        let stored_cubin = &linked.stored_cubin;
        let hit = linked.hit;
        let digest_seconds = linked.digest_seconds;
        let cache_seconds = linked.cache_seconds;
        let (usage, cubin) = unpack(stored_cubin)?;
        let usage: Vec<(String, Option<u32>, u64, u64, u64)> = serde_json::from_value(usage)?;
        let usage: Vec<_> = usage
            .into_iter()
            .map(
                |(kernel, registers, stack_frame_bytes, spill_store_bytes, spill_load_bytes)| {
                    cuda_artifact_finalizer::KernelResourceUsage {
                        kernel,
                        registers,
                        stack_frame_bytes,
                        spill_store_bytes,
                        spill_load_bytes,
                    }
                },
            )
            .collect();
        crate::emit_launch_bounds_spill_warnings(
            tcx,
            &bounds_by_kernel,
            &collection.functions,
            &usage,
        );
        let cubin_digest = digest(cubin);
        let filename = format!("{cubin_digest}.cubin");
        let relative = PathBuf::from(format!("{}.modules", config.output_name)).join(&filename);
        let publish_started = Instant::now();
        let published = config.output_dir.join(&relative);
        // Leave unchanged immutable images in place. Validate full bytes before
        // skipping the write so corruption is repaired from the checked cache.
        if std::fs::read(&published).ok().as_deref() != Some(cubin) {
            atomic_write(&published, cubin)?;
        }
        let publish_seconds = publish_started.elapsed().as_secs_f64();
        if !files_only {
            let bundle_name = name.clone();
            let compile_options =
                crate::embedded_compile_options(options.allow_fma_contraction(), debug, true);
            let mut spec = oxide_artifacts::ArtifactBundleSpec::new(&bundle_name, &target)
                .with_compile_options(compile_options)
                .with_payload(oxide_artifacts::ArtifactPayloadSpec::new(
                    oxide_artifacts::ArtifactPayloadKind::Cubin,
                    &filename,
                    cubin,
                ));
            for kernel in &kernels {
                spec = spec.with_entry(oxide_artifacts::ArtifactEntrySpec::new(
                    &kernel.export_name,
                    oxide_artifacts::ArtifactEntryKind::Kernel,
                ));
            }
            blob.extend(oxide_artifacts::build_artifact_blob(&spec)?);
        }
        modules.insert(
            name.clone(),
            json!({"path":relative,"sha256":cubin_digest,
            "inputs":deps,"kernels":kernels.iter().map(|f| &f.export_name).collect::<Vec<_>>(),
            "native_compilation_phases": linked.compilation_phases,
            "compiler": if source_compiler.as_ref().is_some_and(|compiler| !compiler.nvvm_modules.contains(name)) { "llvm-source" } else { "nvvm-lto" },
            "link_hit":hit,"link_seconds":linked.seconds,
            "link_digest_seconds":digest_seconds,"cubin_cache_seconds":cache_seconds,"publish_seconds":publish_seconds}),
        );
    }
    if modules.is_empty() {
        // Device-only crates need a retained anchor, but no standalone cubin.
        return crate::write_filtered_artifact_anchor_object(
            &config.output_dir,
            &config.output_name,
            tcx.sess.target.llvm_target.as_ref(),
        );
    }
    let path = if files_only {
        crate::write_filtered_artifact_anchor_object(
            &config.output_dir,
            &config.output_name,
            tcx.sess.target.llvm_target.as_ref(),
        )?
    } else {
        let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
        let anchor = reserved_oxide_symbols::artifact_anchor_symbol(&package, &version);
        let object = if std::env::var_os(reserved_oxide_symbols::DEVICE_CODEGEN_CRATE_ENV).is_some()
        {
            let target_anchor = reserved_oxide_symbols::artifact_anchor_symbol_v2(
                &package,
                &version,
                &config.output_name,
                std::env::var("CARGO_BIN_NAME").ok().as_deref(),
            );
            oxide_artifacts::build_host_object_for_target_with_legacy_anchor(
                &blob,
                tcx.sess.target.llvm_target.as_ref(),
                &target_anchor,
                &anchor,
            )?
        } else {
            oxide_artifacts::build_host_object_for_target(
                &blob,
                tcx.sess.target.llvm_target.as_ref(),
                Some(&anchor),
            )?
        };
        crate::write_artifact_object(
            &config.output_dir,
            &config.output_name,
            tcx.sess.target.llvm_target.as_ref(),
            &object,
            "modules",
        )?
    };
    let manifest = json!({"version":1,"package":package,"crate":tcx.crate_name(LOCAL_CRATE).as_str(),
        "target":target,"fma_contraction":options.allow_fma_contraction(),"debug":format!("{policy:?}"),
        "modules":modules,"compilation_units":timings,"seconds":started.elapsed().as_secs_f64()});
    atomic_write(
        &config
            .output_dir
            .join(format!("{}.modules.json", config.output_name)),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiler_namespace_override_respects_rust_module_boundaries() {
        assert!(module_in_namespace("crate::kernels", "crate::kernels"));
        assert!(module_in_namespace(
            "crate::kernels::nested",
            "crate::kernels"
        ));
        assert!(!module_in_namespace(
            "crate::kernels_extra",
            "crate::kernels"
        ));
        assert!(!module_in_namespace("crate::other", "crate::kernels"));
    }
    #[test]
    fn no_history_keeps_cold_estimates() {
        let estimates = BTreeMap::from([("large".into(), 900), ("small".into(), 10)]);
        assert_eq!(timing_priorities(&estimates, &BTreeMap::new()), estimates);
    }
    #[test]
    fn recorded_slow_kernel_precedes_larger_ir() {
        let estimates = BTreeMap::from([("complex".into(), 100), ("simple".into(), 1000)]);
        let costs = BTreeMap::from([("complex".into(), 9.0), ("simple".into(), 0.3)]);
        let scores = timing_priorities(&estimates, &costs);
        assert!(scores["complex"] > scores["simple"]);
    }
    #[test]
    fn unseen_kernels_keep_estimated_relative_order() {
        let estimates = BTreeMap::from([
            ("seen".into(), 100),
            ("large".into(), 900),
            ("small".into(), 10),
        ]);
        let costs = BTreeMap::from([("seen".into(), 2.0)]);
        let scores = timing_priorities(&estimates, &costs);
        assert!(scores["large"] > scores["seen"] && scores["seen"] > scores["small"]);
    }

    #[test]
    fn invalid_native_costs_are_ignored() {
        for bytes in [
            b"0".as_slice(),
            b"-1",
            b"3600.1",
            b"null",
            b"NaN",
            b"broken",
        ] {
            assert_eq!(decode_native_cost(bytes), None);
        }
        assert_eq!(decode_native_cost(b"0.25"), Some(0.25));
        assert_eq!(decode_native_cost(b"3600"), Some(3600.0));
    }
    #[test]
    fn unrelated_history_does_not_rescale_new_modules() {
        let estimates = BTreeMap::from([("new".into(), 42)]);
        let costs = BTreeMap::from([("old".into(), 10.0)]);
        assert_eq!(timing_priorities(&estimates, &costs), estimates);
    }
    #[test]
    fn parallel_finalization_preserves_input_order_and_callback_thread() {
        let caller = std::thread::current().id();
        let result = finalization_workers(
            32,
            4,
            |index| {
                std::thread::sleep(std::time::Duration::from_micros((32 - index) as u64));
                Ok(index)
            },
            || assert_eq!(std::thread::current().id(), caller),
        )
        .unwrap();
        assert_eq!(result, (0..32).collect::<Vec<_>>());
    }

    #[test]
    fn parallel_finalization_drains_jobs_after_failure() {
        let completed = std::sync::atomic::AtomicUsize::new(0);
        let result = finalization_workers(
            32,
            4,
            |index| {
                completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index == 0 {
                    Err("compiler failure".into())
                } else {
                    Ok(index)
                }
            },
            || {},
        );
        assert_eq!(result.unwrap_err().to_string(), "compiler failure");
        assert_eq!(completed.load(std::sync::atomic::Ordering::Relaxed), 32);
    }

    #[test]
    fn cached_finalization_runs_callback_without_workers() {
        let mut called = 0;
        let result = parallel_finalization::<()>(
            0,
            "test",
            |_| panic!("cached jobs must not compile"),
            || called += 1,
        )
        .unwrap();
        assert!(result.is_empty());
        assert_eq!(called, 1);
    }

    #[test]
    fn concurrent_cache_writers_publish_complete_entries() {
        let dir = std::env::temp_dir().join(format!(
            "oxide-concurrent-cache-test-{}",
            std::process::id()
        ));
        let path = dir.join("shared-entry");
        let payload = vec![42u8; 65536];
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let path = &path;
                let payload = &payload;
                scope.spawn(move || {
                    for _ in 0..16 {
                        write_cache(path, payload).unwrap();
                        assert_eq!(read_cache(path).as_deref(), Some(payload.as_slice()));
                    }
                });
            }
        });
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn cache_rejects_truncated_or_corrupted_payloads() {
        let dir =
            std::env::temp_dir().join(format!("oxide-module-cache-test-{}", std::process::id()));
        let path = dir.join("entry");
        write_cache(&path, b"correct").unwrap();
        assert_eq!(read_cache(&path).as_deref(), Some(b"correct".as_slice()));
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[33] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_cache(&path).is_none());
        std::fs::write(&path, b"partial").unwrap();
        assert!(read_cache(&path).is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
