// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Optional whole-host reuse for edits confined to reserved CUDA kernel entries.
//! Public host functions and device helpers keep their ordinary Rust semantics.
//! Only the generated, GPU-only kernel entry's body is excluded from the key.
use rustc_codegen_ssa::{CompiledModule, CompiledModules, ModuleKind};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hash::{StableHash, StableHasher};
use rustc_middle::mono::{CodegenUnit, MonoItem};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::DebugInfo;
use rustc_session::config::{OutputType, OutputTypes};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
type Error = Box<dyn std::error::Error>;
#[derive(Clone)]
pub(crate) struct Request {
    root: PathBuf,
    key: String,
    contract: String,
}
pub(crate) fn kernels_only(name: &str) -> bool {
    std::env::var("CUDA_OXIDE_KERNELS_ONLY").is_ok_and(|selected| selected == name)
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn source_positions(tcx: TyCtxt<'_>, body: &rustc_middle::mir::Body<'_>) -> Vec<String> {
    // Rust's stable Span hash also includes the source-file checksum, which
    // changes for a GPU-only edit. Preserve observable locations themselves.
    let mut result = Vec::new();
    let mut span = |span: rustc_span::Span| {
        if span.is_dummy() {
            return;
        }
        for s in [span, span.source_callsite()] {
            for position in [s.lo(), s.hi()] {
                let location = tcx.sess.source_map().lookup_char_pos(position);
                result.push(format!(
                    "{:?}:{}:{:?}",
                    location.file.name, location.line, location.col
                ));
            }
        }
    };
    span(body.span);
    for scope in body.source_scopes.iter() {
        span(scope.span);
        if let Some((_, callsite)) = scope.inlined {
            span(callsite);
        }
    }
    for block in body.basic_blocks.iter() {
        if matches!(
            block.terminator().kind,
            rustc_middle::mir::TerminatorKind::Call { .. }
                | rustc_middle::mir::TerminatorKind::Assert { .. }
        ) {
            span(block.terminator().source_info.span);
        }
    }
    result
}

pub(crate) fn request<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    output: &Path,
) -> Option<Request> {
    let only = kernels_only(tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE).as_str());
    if !only && tcx.crate_types() != [rustc_structures::CrateType::Rlib] {
        return None;
    }
    if tcx.sess.opts.output_types.keys().any(|kind| {
        !matches!(
            kind,
            OutputType::DepInfo | OutputType::Metadata | OutputType::Exe | OutputType::Object
        )
    }) {
        return None;
    }
    if std::env::var_os("CUDA_OXIDE_REUSE_HOST_FOR_KERNEL_EDITS").is_none()
        || std::env::var_os("CUDA_OXIDE_MODULE_FILES_ONLY").is_none()
        || tcx.sess.opts.debuginfo != DebugInfo::None
        || tcx.sess.instrument_coverage()
        || !tcx.sess.sanitizers().is_empty()
    {
        return None;
    }
    // The source contract covers all CPU/helper HIR, attributes, constants,
    // layouts, external crate metadata and observable source positions. Refuse
    // global assembly on the pre-mono path, just as the full mono path does.
    if cgus.is_empty()
        && tcx.hir_crate_items(()).definitions().any(|local| {
            matches!(
                tcx.def_kind(local.to_def_id()),
                rustc_hir::def::DefKind::GlobalAsm
            )
        })
    {
        return None;
    }
    let mut items = BTreeMap::new();
    for cgu in cgus {
        for (&item, data) in cgu.items() {
            if matches!(item, MonoItem::GlobalAsm(_)) {
                return None;
            }
            // CGU placement and inlined copies are irrelevant when reusing
            // the complete host image. Export linkage still belongs in the key.
            items
                .entry(item.symbol_name(tcx).name.to_string())
                .or_insert((item, data.linkage, data.visibility));
        }
    }
    let mir_started = std::time::Instant::now();
    let mir = if only {
        String::new()
    } else {
        tcx.with_stable_hashing_context(|mut hcx| {
            let mut hash = StableHasher::new();
            // Monomorphizations share instance MIR before substituting their
            // arguments. Hash each body once; symbol/instance/ABI identity is
            // still covered separately for every emitted host mono item.
            let mut body_fingerprints = std::collections::HashMap::new();
            // Host panic/track_caller locations can be observable even with debug
            // information disabled. Retain source positions in the host key.
            hcx.while_hashing_spans(false, |hcx| {
                for id in tcx.hir_crate_items(()).definitions() {
                    let id = id.to_def_id();
                    if tcx.is_foreign_item(id)
                        && matches!(
                            tcx.def_kind(id),
                            rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::Static { .. }
                        )
                    {
                        tcx.type_of(id).stable_hash(hcx, &mut hash);
                        tcx.codegen_fn_attrs(id).stable_hash(hcx, &mut hash);
                    }
                    let has_ctfe = match tcx.def_kind(id) {
                        rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn => {
                            tcx.is_const_fn(id)
                        }
                        rustc_hir::def::DefKind::Const { .. }
                        | rustc_hir::def::DefKind::AnonConst => true,
                        rustc_hir::def::DefKind::AssocConst { .. } => {
                            crate::incremental::associated_has_definition(tcx, id)
                        }
                        _ => false,
                    };
                    if has_ctfe && tcx.is_mir_available(id) {
                        source_positions(tcx, tcx.mir_for_ctfe(id)).stable_hash(hcx, &mut hash);
                    }
                }
                for (name, &(item, linkage, visibility)) in &items {
                    name.stable_hash(hcx, &mut hash);
                    item.stable_hash(hcx, &mut hash);
                    linkage.stable_hash(hcx, &mut hash);
                    visibility.stable_hash(hcx, &mut hash);
                    match item {
                        MonoItem::Fn(instance) => {
                            tcx.body_codegen_attrs(instance.def_id())
                                .stable_hash(hcx, &mut hash);
                            let body = tcx.instance_mir(instance.def);
                            if is_kernel_entry(tcx, instance.def_id()) {
                                body.arg_count.stable_hash(hcx, &mut hash);
                                for local in body.local_decls.iter().take(body.arg_count + 1) {
                                    local.ty.stable_hash(hcx, &mut hash);
                                }
                            } else {
                                let fingerprint =
                                    body_fingerprints.entry(instance.def).or_insert_with(|| {
                                        let mut body_hash = StableHasher::new();
                                        body.stable_hash(hcx, &mut body_hash);
                                        source_positions(tcx, body)
                                            .stable_hash(hcx, &mut body_hash);
                                        if tcx.is_mir_available(instance.def_id()) {
                                            tcx.promoted_mir(instance.def_id())
                                                .stable_hash(hcx, &mut body_hash);
                                            for promoted in
                                                tcx.promoted_mir(instance.def_id()).iter()
                                            {
                                                source_positions(tcx, promoted)
                                                    .stable_hash(hcx, &mut body_hash);
                                            }
                                        }
                                        body_hash.finish::<Fingerprint>()
                                    });
                                fingerprint.stable_hash(hcx, &mut hash);
                            }
                        }
                        MonoItem::Static(id) => {
                            tcx.codegen_fn_attrs(id).stable_hash(hcx, &mut hash);
                            tcx.mir_for_ctfe(id).stable_hash(hcx, &mut hash);
                            source_positions(tcx, tcx.mir_for_ctfe(id)).stable_hash(hcx, &mut hash);
                        }
                        MonoItem::GlobalAsm(_) => unreachable!(),
                    }
                }
            });
            format!("{:?}", hash.finish::<Fingerprint>())
        })
    };
    let mut cfg: Vec<_> = tcx.sess.config.iter().map(|x| format!("{x:?}")).collect();
    cfg.sort();
    let mut host_options = tcx.sess.opts.clone();
    host_options.crate_types.clear();
    host_options.output_types = OutputTypes::new(&[]);
    let metadata_started = std::time::Instant::now();
    let metadata = crate::incremental::metadata_fingerprint(tcx);
    let metadata_seconds = metadata_started.elapsed().as_secs_f64();
    let source_started = std::time::Instant::now();
    let key_parts = json!([
        "host-kernel-edit-v3",
        mir,
        metadata,
        cfg,
        format!("{:?}", host_options.dep_tracking_hash(false)),
        tcx.sess.target.llvm_target.as_ref(),
        tcx.sess.target.data_layout.as_ref()
    ]);
    let contract = source_contract(tcx, &key_parts)?;
    eprintln!(
        "[device-modules] host proof components: metadata={metadata_seconds:.3}s source={:.3}s total={:.3}s",
        source_started.elapsed().as_secs_f64(),
        mir_started.elapsed().as_secs_f64()
    );
    let key = digest(
        json!([
            if cgus.is_empty() {
                "host-kernel-edit-v6-source-contract"
            } else {
                "host-kernel-edit-v5-body-fingerprints"
            },
            key_parts,
            contract
        ])
        .to_string()
        .as_bytes(),
    );
    Some(Request {
        contract,
        root: output
            .join("cache/host/v2")
            .join(tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE).as_str()),
        key,
    })
}
// The source contract identifies host semantics before collecting CPU mono
// items. Cover every host/helper body, including generic and unused bodies,
// all type/trait/impl HIR, external metadata and observable source positions.
// Kernel bodies are excluded only when host HIR never references their entries.
fn is_kernel_entry(tcx: TyCtxt<'_>, id: rustc_hir::def_id::DefId) -> bool {
    tcx.opt_item_name(id)
        .is_some_and(|name| reserved_oxide_symbols::is_kernel_symbol(name.as_str()))
}
fn source_contract(tcx: TyCtxt<'_>, key_parts: &Value) -> Option<String> {
    use rustc_hir::{
        def::DefKind,
        intravisit::{self, Visitor},
    };
    struct HostLocations<'tcx> {
        tcx: TyCtxt<'tcx>,
        positions: Vec<(usize, usize, usize)>,
        files: Vec<String>,
        file_indices: std::collections::HashMap<rustc_span::BytePos, usize>,
        cached_positions: std::collections::HashMap<rustc_span::BytePos, (usize, usize, usize)>,
        kernel_reference: bool,
    }
    impl HostLocations<'_> {
        fn span(&mut self, span: rustc_span::Span) {
            if span.is_dummy() {
                return;
            }
            for span in [span, span.source_callsite()] {
                let pos = span.lo();
                let value = if let Some(value) = self.cached_positions.get(&pos) {
                    *value
                } else {
                    let p = self.tcx.sess.source_map().lookup_char_pos(pos);
                    let index = *self
                        .file_indices
                        .entry(p.file.start_pos)
                        .or_insert_with(|| {
                            let index = self.files.len();
                            self.files.push(format!("{:?}", p.file.name));
                            index
                        });
                    let value = (index, p.line, p.col.0);
                    self.cached_positions.insert(pos, value);
                    value
                };
                self.positions.push(value);
            }
        }
    }
    impl<'tcx> Visitor<'tcx> for HostLocations<'tcx> {
        fn visit_expr(&mut self, expr: &'tcx rustc_hir::Expr<'tcx>) {
            self.span(expr.span);
            intravisit::walk_expr(self, expr);
        }
        fn visit_pat(&mut self, pat: &'tcx rustc_hir::Pat<'tcx>) {
            self.span(pat.span);
            intravisit::walk_pat(self, pat);
        }
        fn visit_stmt(&mut self, stmt: &'tcx rustc_hir::Stmt<'tcx>) {
            self.span(stmt.span);
            intravisit::walk_stmt(self, stmt);
        }
        fn visit_block(&mut self, block: &'tcx rustc_hir::Block<'tcx>) {
            self.span(block.span);
            intravisit::walk_block(self, block);
        }
        fn visit_path(&mut self, path: &rustc_hir::Path<'tcx>, _id: rustc_hir::HirId) {
            self.span(path.span);
            if let Some(def) = path.res.opt_def_id() {
                self.kernel_reference |= is_kernel_entry(self.tcx, def);
            }
            intravisit::walk_path(self, path);
        }
    }
    let mut locations = HostLocations {
        tcx,
        positions: Vec::new(),
        files: Vec::new(),
        file_indices: std::collections::HashMap::new(),
        cached_positions: std::collections::HashMap::new(),
        kernel_reference: false,
    };
    let hash = tcx.with_stable_hashing_context(|mut hcx| {
        let mut hash = StableHasher::new();
        hcx.while_hashing_spans(false, |hcx| {
            for local in tcx.hir_crate_items(()).definitions() {
                let id = local.to_def_id();
                id.stable_hash(hcx, &mut hash);
                tcx.hir_attrs(tcx.local_def_id_to_hir_id(local))
                    .stable_hash(hcx, &mut hash);
                match tcx.def_kind(id) {
                    DefKind::Fn | DefKind::AssocFn if is_kernel_entry(tcx, id) => {
                        tcx.fn_sig(id).stable_hash(hcx, &mut hash);
                        tcx.generics_of(id).stable_hash(hcx, &mut hash);
                        tcx.param_env(id).stable_hash(hcx, &mut hash);
                        tcx.codegen_fn_attrs(id).stable_hash(hcx, &mut hash);
                    }
                    DefKind::Fn | DefKind::AssocFn => {
                        // The complete HIR declaration includes the signature,
                        // generics and predicates. Avoid eager type/attribute
                        // queries for unchanged CPU functions in shader mode.
                        // Bodies and local/dependency type identities are hashed
                        // separately below and in the common metadata fence.
                        tcx.hir_node_by_def_id(local).stable_hash(hcx, &mut hash);
                        locations.span(tcx.def_span(id));
                    }
                    _ => {
                        tcx.hir_node_by_def_id(local).stable_hash(hcx, &mut hash);
                    }
                }
            }
            // `definitions()` lists item owners, not closure bodies. Cover
            // every local body, including generic/unused CPU closures and consts.
            for local in tcx.hir_body_owners() {
                let id = local.to_def_id();
                if is_kernel_entry(tcx, id) {
                    continue;
                }
                if matches!(tcx.def_kind(id), DefKind::Closure | DefKind::AnonConst) {
                    let mut parent = tcx.opt_parent(id);
                    let mut gpu_body = false;
                    while let Some(id) = parent {
                        if is_kernel_entry(tcx, id) {
                            gpu_body = true;
                            break;
                        }
                        parent = tcx.opt_parent(id);
                    }
                    if gpu_body {
                        continue;
                    }
                }
                id.stable_hash(hcx, &mut hash);
                let body = tcx.hir_body_owned_by(local);
                body.stable_hash(hcx, &mut hash);
                locations.visit_body(body);
            }
            locations.files.stable_hash(hcx, &mut hash);
            locations.positions.stable_hash(hcx, &mut hash);
        });
        format!("{:?}", hash.finish::<Fingerprint>())
    });
    if locations.kernel_reference {
        return None;
    }
    Some(digest(
        json!([
            "kernel-source-contract-v4",
            hash,
            &key_parts.as_array()?[2..]
        ])
        .to_string()
        .as_bytes(),
    ))
}

/// Retain only ordinary Rust work products whose original codegen dependency
/// node is still green. Returning an empty map would remove reusable CGUs from
/// the next incremental session after a whole-host cache hit.
pub(crate) fn retained_work_products(tcx: TyCtxt<'_>) -> rustc_middle::dep_graph::WorkProductMap {
    let mut products = rustc_middle::dep_graph::WorkProductMap::default();
    if !tcx.dep_graph.is_fully_enabled() {
        return products;
    }
    for (id, product) in tcx
        .dep_graph
        .previous_work_products()
        .to_sorted_stable_ord()
    {
        if product.cgu_name == "metadata" {
            continue;
        }
        let node =
            CodegenUnit::new(rustc_span::Symbol::intern(&product.cgu_name)).codegen_dep_node(tcx);
        if tcx.dep_graph.try_mark_green(tcx, &node).is_some() {
            products.insert(*id, product.clone());
        }
    }
    eprintln!(
        "[device-modules] retained {} green host work products",
        products.len()
    );
    products
}

pub(crate) fn publish_contract(request: &Request) -> Result<(), Error> {
    // Publish only after the full host link succeeded. Keep just the latest
    // host contract so historical cache entries cannot approve a stale binary.
    atomic_write(
        &request.root.join("latest-contract.json"),
        &serde_json::to_vec(&json!({"contract":request.contract,"key":request.key}))?,
    )
}
fn contract_request(request: &Request) -> Option<Request> {
    let record: Value =
        serde_json::from_slice(&std::fs::read(request.root.join("latest-contract.json")).ok()?)
            .ok()?;
    if record.get("contract")?.as_str()? != request.contract {
        return None;
    }
    let key = record.get("key")?.as_str()?;
    if key.len() != 64 || !key.bytes().all(|x| x.is_ascii_hexdigit()) {
        return None;
    }
    Some(Request {
        key: key.into(),
        ..request.clone()
    })
}
pub(crate) fn read_contract(request: &Request) -> Option<CompiledModules> {
    read_record(&contract_request(request)?)
}
/// Restore verified host objects from the latest matching source contract before
/// collecting the host monomorphization graph. Keep the original full-MIR key.
pub(crate) fn read_for_contract(request: &Request) -> Option<(Request, CompiledModules)> {
    let request = contract_request(request)?;
    let modules = read(&request)?;
    Some((request, modules))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    std::fs::create_dir_all(path.parent().ok_or("cache path has no parent")?)?;
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}
fn store_file(request: &Request, path: &Option<PathBuf>) -> Result<Value, Error> {
    let Some(path) = path else {
        return Ok(Value::Null);
    };
    let bytes = std::fs::read(path)?;
    let hash = digest(&bytes);
    let name = format!(
        "{hash}.{}",
        path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("object")
    );
    atomic_write(&request.root.join("objects").join(&name), &bytes)?;
    Ok(json!({"file":name,"sha256":hash}))
}
fn store_module(request: &Request, module: &CompiledModule) -> Result<Value, Error> {
    Ok(
        json!({"name":module.name,"kind":if module.kind==ModuleKind::Allocator {"allocator"} else {"regular"},
        "object":store_file(request,&module.object)?,"dwarf":store_file(request,&module.dwarf_object)?,
        "bytecode":store_file(request,&module.bytecode)?,"assembly":store_file(request,&module.assembly)?,
        "llvm_ir":store_file(request,&module.llvm_ir)?,"global_asm":store_file(request,&module.global_asm_object)?}),
    )
}
pub(crate) fn write(request: &Request, modules: &CompiledModules) -> Result<(), Error> {
    let stored: Result<Vec<_>, _> = modules
        .modules
        .iter()
        .map(|m| store_module(request, m))
        .collect();
    let allocator = modules
        .allocator_module
        .as_ref()
        .map(|m| store_module(request, m))
        .transpose()?;
    atomic_write(
        &request.root.join(format!("{}.json", request.key)),
        &serde_json::to_vec(&json!({"key":request.key,"modules":stored?,"allocator":allocator}))?,
    )
}
fn load_file(request: &Request, value: &Value) -> Option<Option<PathBuf>> {
    if value.is_null() {
        return Some(None);
    }
    let name = value.get("file")?.as_str()?;
    if Path::new(name).components().count() != 1 {
        return None;
    }
    let hash = value.get("sha256")?.as_str()?;
    if hash.len() != 64 || !name.starts_with(&format!("{hash}.")) {
        return None;
    }
    let path = request.root.join("objects").join(name);
    if digest(&std::fs::read(&path).ok()?) != hash {
        return None;
    }
    Some(Some(path))
}
fn load_module(request: &Request, value: &Value) -> Option<CompiledModule> {
    Some(CompiledModule {
        name: value.get("name")?.as_str()?.to_string(),
        kind: match value.get("kind")?.as_str()? {
            "regular" => ModuleKind::Regular,
            "allocator" => ModuleKind::Allocator,
            _ => return None,
        },
        object: load_file(request, value.get("object")?)?,
        dwarf_object: load_file(request, value.get("dwarf")?)?,
        bytecode: load_file(request, value.get("bytecode")?)?,
        assembly: load_file(request, value.get("assembly")?)?,
        llvm_ir: load_file(request, value.get("llvm_ir")?)?,
        global_asm_object: load_file(request, value.get("global_asm")?)?,
        links_from_incr_cache: Vec::new(),
    })
}
pub(crate) fn read_record(request: &Request) -> Option<CompiledModules> {
    let record: Value = serde_json::from_slice(
        &std::fs::read(request.root.join(format!("{}.json", request.key))).ok()?,
    )
    .ok()?;
    if record.get("key")?.as_str()? != request.key {
        return None;
    }
    let modules = record
        .get("modules")?
        .as_array()?
        .iter()
        .map(|v| load_module(request, v))
        .collect::<Option<Vec<_>>>()?;
    let allocator = record.get("allocator")?;
    Some(CompiledModules {
        modules,
        allocator_module: if allocator.is_null() {
            None
        } else {
            Some(load_module(request, allocator)?)
        },
    })
}
pub(crate) fn read(request: &Request) -> Option<CompiledModules> {
    let mut result = read_record(request)?;
    // rustc owns and deletes the objects passed to its linker. Hand it fresh
    // working copies, preserving the verified cache for successive edits.
    let work = request
        .root
        .join("work")
        .join(&request.key)
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&work).ok()?;
    for module in result
        .modules
        .iter_mut()
        .chain(result.allocator_module.iter_mut())
    {
        if Path::new(&module.name).components().count() != 1 {
            return None;
        }
        for (kind, path) in [
            ("object", &mut module.object),
            ("dwarf", &mut module.dwarf_object),
            ("bytecode", &mut module.bytecode),
            ("assembly", &mut module.assembly),
            ("llvm_ir", &mut module.llvm_ir),
            ("global_asm", &mut module.global_asm_object),
        ] {
            if let Some(original) = path {
                let destination = work.join(format!(
                    "{}.{kind}.{}",
                    module.name,
                    original.extension()?.to_str()?
                ));
                std::fs::copy(&original, &destination).ok()?;
                *original = destination;
            }
        }
    }
    Some(result)
}
