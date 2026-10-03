// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Cached per-module NVVM/LTO compilation. All three cache boundaries
//! precede the expensive operation they memoize; a hit never translates MIR.
use crate::collector::{self, CollectedFunction, CollectionResult};
use crate::device_codegen::{self, DeviceCodegenConfig};
use cuda_artifact_finalizer::{
    DebugPolicy, FinalizationOptions, Finalizer, FinalizerOutput, NamedInput,
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
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}
fn write_cache(path: &Path, bytes: &[u8]) -> Result<(), Error> {
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
                        | "CUDA_OXIDE_INTERNAL_RUSTC_WRAPPER"
                        | "CUDA_OXIDE_UPSTREAM_RUSTC_WRAPPER"
                        | "CUDA_OXIDE_HOST_KEY_TRACE"
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
            // Entry-block stack allocation changes emitted device IR even when
            // source MIR is unchanged.
            "device-module-input-v4-entry-allocas",
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
) -> Result<PathBuf, Error> {
    let started = Instant::now();
    if !collection
        .functions
        .iter()
        .any(|function| function.is_kernel)
    {
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
    let kernel_owners: BTreeSet<_> = collection
        .functions
        .iter()
        .filter(|f| f.is_kernel)
        .map(|f| module_owner(tcx, f.instance.def_id(), &marked))
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
    let finalizer = Finalizer::discover()?;
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
    type CompiledUnit = (Vec<u8>, BTreeSet<String>, [u8; 32]);
    let mut units: BTreeMap<String, CompiledUnit> = BTreeMap::new();
    let mut timings = Vec::new();
    let mut bounds_by_kernel = BTreeMap::new();
    for (name, indices) in &groups {
        let unit_started = Instant::now();
        let mut roots: Vec<_> = indices
            .iter()
            .map(|&i| collection.functions[i].clone())
            .collect();
        roots.sort_by(|a, b| a.export_name.cmp(&b.export_name));
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
        let ir_seconds = ir_started.elapsed().as_secs_f64();
        let compiler_key = finalizer
            .compiler()
            .artifact_digest(name, ir, &options)
            .ok_or("libNVVM provenance is unavailable; cannot safely cache")?;
        let lto_started = Instant::now();
        let (lto, lto_hit) = cached(&cache, "ltoir", &digest(&compiler_key), || {
            Ok(finalizer
                .compiler()
                .compile_nvvm_ir_to_ltoir(name, ir, &options)?)
        })?;
        timings.push(json!({"module":name, "definitions":roots.len(), "declarations":declarations.len(),
            "nvvm_hit":ir_hit,"nvvm_seconds":ir_seconds,"ltoir_hit":lto_hit,
            "ltoir_seconds":lto_started.elapsed().as_secs_f64(),"seconds":unit_started.elapsed().as_secs_f64()}));
        eprintln!(
            "[device-modules] {name}: NVVM {} ({ir_seconds:.3}s), LTOIR {}",
            if ir_hit { "hit" } else { "compiled" },
            if lto_hit { "hit" } else { "compiled" }
        );
        let lto_digest: [u8; 32] = Sha256::digest(&lto).into();
        units.insert(name.clone(), (lto, dependencies, lto_digest));
    }
    // File loaders already consume the manifest. Omitting embedded payloads
    // avoids copying every cached cubin into a new host object on each edit.
    let files_only = std::env::var_os("CUDA_OXIDE_MODULE_FILES_ONLY").is_some();
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
        let mut deps = units[name].1.clone();
        loop {
            let previous = deps.len();
            for dependency in deps.clone() {
                deps.extend(units[&dependency].1.iter().cloned());
            }
            if deps.len() == previous {
                break;
            }
        }
        let inputs: Vec<_> = deps
            .iter()
            .map(|dep| NamedInput::new(dep, &units[dep].0))
            .collect();
        let link_started = Instant::now();
        // Hash each immutable LTOIR once. Repeated links reference its digest,
        // retaining input order, names, tool provenance and all linker options.
        let digested_inputs: Vec<_> = deps
            .iter()
            .map(|dep| NamedInput::new(dep, &units[dep].2))
            .collect();
        let link_key = finalizer
            .linker()
            .artifact_digest(&digested_inputs, &options, FinalizerOutput::Cubin)
            .ok_or("nvJitLink provenance is unavailable; cannot safely cache")?;
        let link_key = digest(&link_key);
        let digest_seconds = link_started.elapsed().as_secs_f64();
        let cache_started = Instant::now();
        let mut migrated = false;
        let (stored_cubin, hit) = cached(&cache, "cubin-v3-digested-inputs", &link_key, || {
            // Adopt already verified v2 images without paying for a cold link.
            let old_key = finalizer
                .linker()
                .artifact_digest(&inputs, &options, FinalizerOutput::Cubin)
                .ok_or("nvJitLink provenance changed")?;
            if let Some(bytes) = read_cache(&cache.join("cubin-v2").join(digest(&old_key))) {
                migrated = true;
                return Ok(bytes);
            }
            let report =
                finalizer.link_ltoir_with_report(&inputs, &options, FinalizerOutput::Cubin)?;
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
            pack(json!(usage), &report.image)
        })?;
        let hit = hit || migrated;
        let cache_seconds = cache_started.elapsed().as_secs_f64();
        let (usage, cubin) = unpack(&stored_cubin)?;
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
            "link_hit":hit,"link_seconds":link_started.elapsed().as_secs_f64(),
            "link_digest_seconds":digest_seconds,"cubin_cache_seconds":cache_seconds,"publish_seconds":publish_seconds}),
        );
        eprintln!(
            "[device-modules] {name}: cubin {} ({:.3}s)",
            if hit { "hit" } else { "linked" },
            link_started.elapsed().as_secs_f64()
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
