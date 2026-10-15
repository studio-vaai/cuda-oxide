/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;

pub(super) const MATERIALIZE_ENV: &str = reserved_oxide_symbols::MATERIALIZE_CUBIN_ENV;
pub(super) const EXPECTED_PROVENANCE_ENV: &str =
    reserved_oxide_symbols::MATERIALIZER_PROVENANCE_ENV;
pub(super) const MATERIALIZER_HANDSHAKE_ENV: &str =
    reserved_oxide_symbols::MATERIALIZER_HANDSHAKE_ENV;
pub(super) const CODEGEN_FINGERPRINT_ENV: &str = reserved_oxide_symbols::CODEGEN_FINGERPRINT_ENV;
pub(super) const DEVICE_CODEGEN_CRATE_ENV: &str = reserved_oxide_symbols::DEVICE_CODEGEN_CRATE_ENV;
pub(super) const BACKEND_IDENTITY_CFG: &str = "cuda_oxide_internal_backend_identity";
pub(super) const LEGACY_CODEGEN_FINGERPRINT_CFG: &str = "cuda_oxide_internal_codegen_env";
pub(super) const LEGACY_MATERIALIZER_PROVENANCE_CFG: &str =
    "cuda_oxide_internal_materializer_provenance";
const MATERIALIZER_HANDSHAKE_CACHE: &str = ".oxide-artifacts/materializer-handshake/v1.json";

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct MaterializationMode {
    pub(super) prepared: Option<PreparedMaterialization>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PreparedMaterialization {
    pub(super) provenance_sha256_hex: String,
    pub(super) tool_identity_handshake_json: String,
}

impl MaterializationMode {
    pub(super) fn enabled(&self) -> bool {
        self.prepared.is_some()
    }

    pub(super) fn apply(&self, cmd: &mut Command) {
        if let Some(prepared) = &self.prepared {
            // These override inherited/project values: they are a single
            // wrapper-generated handshake tied to this Cargo invocation.
            cmd.env(MATERIALIZE_ENV, "1")
                .env(EXPECTED_PROVENANCE_ENV, &prepared.provenance_sha256_hex)
                .env(
                    MATERIALIZER_HANDSHAKE_ENV,
                    &prepared.tool_identity_handshake_json,
                )
                .env("CUDA_OXIDE_EMIT_NVVM_IR", "1");
        }
    }
}

pub(super) fn prepare_materialization(
    ctx: &Context,
    cli_requested: bool,
    cli_arch: Option<&str>,
    emit_nvvm_ir: bool,
) -> MaterializationMode {
    prepare_materialization_result(ctx, cli_requested, cli_arch, emit_nvvm_ir).unwrap_or_else(
        |error| {
            eprintln!("Error: {error}");
            std::process::exit(2);
        },
    )
}

/// `prepare_materialization` with the ambient `CUDA_OXIDE_MATERIALIZE_CUBIN`
/// injected, so `cargo_passthrough_command_with_env` can reach
/// `materialization_requested_with_env`.
///
/// Note this still exits the process on error, which inside a unit test aborts
/// the whole test binary rather than failing one case -- a further reason tests
/// must not reach the ambient read.
pub(super) fn prepare_materialization_with_env(
    ctx: &Context,
    cli_requested: bool,
    cli_arch: Option<&str>,
    emit_nvvm_ir: bool,
    materialize_env: Option<std::ffi::OsString>,
) -> MaterializationMode {
    prepare_materialization_result_with_env(
        ctx,
        cli_requested,
        cli_arch,
        emit_nvvm_ir,
        materialize_env,
    )
    .unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(2);
    })
}

const EMIT_NVVM_IR_ENV: &str = "CUDA_OXIDE_EMIT_NVVM_IR";

pub(super) fn nvvm_ir_requested(ctx: &Context) -> Result<bool, String> {
    nvvm_ir_requested_with_env(ctx, std::env::var_os(EMIT_NVVM_IR_ENV))
}

/// `nvvm_ir_requested` with the ambient `CUDA_OXIDE_EMIT_NVVM_IR` injected.
///
/// The process value outranks project config, so resolution has to be
/// injectable for unit tests: an exported `CUDA_OXIDE_EMIT_NVVM_IR` would
/// otherwise decide the answer before the configured value is consulted.
pub(super) fn nvvm_ir_requested_with_env(
    ctx: &Context,
    env_value: Option<std::ffi::OsString>,
) -> Result<bool, String> {
    if let Some(value) = env_value {
        let value = value
            .into_string()
            .map_err(|_| format!("{EMIT_NVVM_IR_ENV} is not valid Unicode"))?;
        return parse_strict_bool(EMIT_NVVM_IR_ENV, &value);
    }

    if let Some(value) = project_config_env(ctx, EMIT_NVVM_IR_ENV) {
        return parse_strict_bool(EMIT_NVVM_IR_ENV, value);
    }

    Ok(false)
}

pub(super) fn materialization_requested(
    ctx: &Context,
    cli_requested: bool,
) -> Result<bool, String> {
    materialization_requested_with_env(ctx, cli_requested, std::env::var_os(MATERIALIZE_ENV))
}

/// `materialization_requested` with the ambient `CUDA_OXIDE_MATERIALIZE_CUBIN`
/// injected.
///
/// The process value outranks project config, so resolution has to be
/// injectable for unit tests: an exported value would otherwise turn
/// materialization on for tests that pass `materialize_cubin: false`, sending
/// them into `discover_materializer_provenance`. Same rationale as
/// `nvvm_ir_requested_with_env`.
fn materialization_requested_with_env(
    ctx: &Context,
    cli_requested: bool,
    env_value: Option<std::ffi::OsString>,
) -> Result<bool, String> {
    if cli_requested {
        return Ok(true);
    }

    if let Some(value) = env_value {
        let value = value
            .into_string()
            .map_err(|_| format!("{MATERIALIZE_ENV} is not valid Unicode"))?;
        return parse_strict_bool(MATERIALIZE_ENV, &value);
    }

    if let Some(value) = project_config_env(ctx, MATERIALIZE_ENV) {
        return parse_strict_bool(MATERIALIZE_ENV, value);
    }

    Ok(false)
}

pub(super) fn prepare_materialization_result(
    ctx: &Context,
    cli_requested: bool,
    cli_arch: Option<&str>,
    emit_nvvm_ir: bool,
) -> Result<MaterializationMode, String> {
    prepare_materialization_result_with_env(
        ctx,
        cli_requested,
        cli_arch,
        emit_nvvm_ir,
        std::env::var_os(MATERIALIZE_ENV),
    )
}

/// `prepare_materialization_result` with the ambient
/// `CUDA_OXIDE_MATERIALIZE_CUBIN` injected, forwarded to
/// `materialization_requested_with_env`.
fn prepare_materialization_result_with_env(
    ctx: &Context,
    cli_requested: bool,
    cli_arch: Option<&str>,
    emit_nvvm_ir: bool,
    materialize_env: Option<std::ffi::OsString>,
) -> Result<MaterializationMode, String> {
    let enabled = materialization_requested_with_env(ctx, cli_requested, materialize_env)?;
    if !enabled {
        return Ok(MaterializationMode::default());
    }
    if emit_nvvm_ir {
        return Err(
            "--materialize-cubin cannot be combined with --emit-nvvm-ir; one requests a final cubin and the other requests NVVM IR"
                .to_string(),
        );
    }

    let arch = configured_arch_label(ctx, cli_arch).ok_or_else(|| {
        "--materialize-cubin requires --arch, CUDA_OXIDE_TARGET, or a configured default-arch"
            .to_string()
    })?;
    let _: cuda_artifact_finalizer::CudaArch = arch
        .parse()
        .map_err(|error| format!("invalid materialization target {arch:?}: {error}"))?;

    let handshake = discover_materializer_handshake(ctx)?;
    let handshake_json = serde_json::to_string(&handshake)
        .map_err(|error| format!("could not encode materializer handshake: {error}"))?;
    Ok(MaterializationMode {
        prepared: Some(PreparedMaterialization {
            provenance_sha256_hex: digest_hex(&handshake.provenance_sha256),
            tool_identity_handshake_json: handshake_json,
        }),
    })
}

fn discover_materializer_handshake(
    ctx: &Context,
) -> Result<cuda_artifact_finalizer::MaterializerHandshakeV1, String> {
    discover_materializer_handshake_at(ctx, &materializer_handshake_cache_path(ctx))
}

fn discover_materializer_handshake_at(
    ctx: &Context,
    cache: &Path,
) -> Result<cuda_artifact_finalizer::MaterializerHandshakeV1, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("could not locate cargo-oxide executable: {error}"))?;
    let mut command = materializer_discovery_command(ctx, &executable);
    if let Some(cached) = read_materializer_handshake_cache_at(cache) {
        command.env(MATERIALIZER_HANDSHAKE_ENV, cached);
    }
    let output = command
        .output()
        .map_err(|error| format!("could not start CUDA materializer discovery: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "CUDA materializer discovery failed: {}",
            stderr.trim()
        ));
    }
    let handshake = String::from_utf8(output.stdout)
        .map_err(|_| "CUDA materializer discovery returned non-UTF-8 output".to_string())?;
    let handshake: cuda_artifact_finalizer::MaterializerHandshakeV1 =
        serde_json::from_str(handshake.trim()).map_err(|error| {
            format!("CUDA materializer discovery returned an invalid v1 handshake: {error}")
        })?;
    if !handshake.has_consistent_provenance() {
        return Err(format!(
            "CUDA materializer discovery returned an inconsistent handshake version {}",
            handshake.version
        ));
    }
    write_materializer_handshake_cache_at(cache, &handshake);
    Ok(handshake)
}

/// Native modules use the same verified compiler identity as embedded cubins.
/// Macros already track EXPECTED_PROVENANCE_ENV, so switching a linker (or
/// replacing its file) invalidates device owners even when source is unchanged.
/// Keep the descriptor hint in Cargo's selected artifact cache; it is not a
/// semantic identity and does not invalidate host-only dependencies.
pub(super) fn apply_native_tool_identity(cmd: &mut Command, ctx: &Context) -> Result<(), String> {
    if !native_modules_enabled() {
        return Ok(());
    }
    let output = cmd
        .get_envs()
        .find_map(|(key, value)| (key == "CUDA_OXIDE_PTX_DIR").then_some(value).flatten())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("CUDA_OXIDE_PTX_DIR").map(PathBuf::from))
        .ok_or("native artifact directory was not configured")?;
    let handshake = discover_materializer_handshake_at(
        ctx,
        &output.join("cache/materializer-handshake/v1.json"),
    )?;
    cmd.env(
        EXPECTED_PROVENANCE_ENV,
        digest_hex(&handshake.provenance_sha256),
    )
    .env(
        MATERIALIZER_HANDSHAKE_ENV,
        serde_json::to_string(&handshake)
            .map_err(|error| format!("could not encode native tool handshake: {error}"))?,
    );
    apply_source_compiler_identity(cmd, ctx, &output)?;
    Ok(())
}

pub(super) fn materializer_discovery_command(ctx: &Context, executable: &Path) -> Command {
    let mut command = Command::new(executable);
    command.arg("__materializer-handshake");
    apply_config_env(&mut command, ctx);
    apply_ld_library_path(&mut command, ctx);
    // Only the local cache explicitly installed by the caller may seed the
    // helper; never consume an inherited or project-provided internal value.
    command.env_remove(MATERIALIZER_HANDSHAKE_ENV);
    command
}

pub(super) fn materializer_handshake_cache_path(ctx: &Context) -> PathBuf {
    ctx.workspace_root.join(MATERIALIZER_HANDSHAKE_CACHE)
}

#[cfg(test)]
pub(super) fn read_materializer_handshake_cache(ctx: &Context) -> Option<String> {
    read_materializer_handshake_cache_at(&materializer_handshake_cache_path(ctx))
}

fn read_materializer_handshake_cache_at(path: &Path) -> Option<String> {
    let json = fs::read_to_string(path).ok()?;
    let handshake: cuda_artifact_finalizer::MaterializerHandshakeV1 =
        serde_json::from_str(json.trim()).ok()?;
    handshake.has_consistent_provenance().then_some(json)
}

#[cfg(test)]
pub(super) fn write_materializer_handshake_cache(
    ctx: &Context,
    handshake: &cuda_artifact_finalizer::MaterializerHandshakeV1,
) {
    let path = materializer_handshake_cache_path(ctx);
    write_materializer_handshake_cache_at(&path, handshake);
}

fn write_materializer_handshake_cache_at(
    path: &Path,
    handshake: &cuda_artifact_finalizer::MaterializerHandshakeV1,
) {
    let Some(parent) = path.parent() else {
        return;
    };
    if fs::create_dir_all(parent).is_err() {
        return;
    }
    let Ok(json) = serde_json::to_string(handshake) else {
        return;
    };
    let temporary = parent.join(format!("v1.{}.tmp", std::process::id()));
    if fs::write(&temporary, json).is_ok() {
        let _ = fs::rename(&temporary, path);
    }
}

pub fn print_materializer_handshake() {
    let cached = std::env::var(MATERIALIZER_HANDSHAKE_ENV)
        .ok()
        .and_then(|json| serde_json::from_str(&json).ok())
        .filter(cuda_artifact_finalizer::MaterializerHandshakeV1::has_consistent_provenance);
    let finalizer = cached
        .as_ref()
        .map_or_else(
            cuda_artifact_finalizer::Finalizer::discover,
            cuda_artifact_finalizer::Finalizer::discover_with_handshake,
        )
        .unwrap_or_else(|error| {
            eprintln!("could not discover CUDA artifact finalizer: {error}");
            std::process::exit(1);
        });
    let handshake = finalizer.materializer_handshake().unwrap_or_else(|| {
        eprintln!(
            "the loaded libNVVM or nvJitLink library cannot be tied to an exact file; refusing materialization because Cargo could not fingerprint the compiler inputs"
        );
        std::process::exit(1);
    });
    println!(
        "{}",
        serde_json::to_string(&handshake).unwrap_or_else(|error| {
            eprintln!("could not encode CUDA materializer handshake: {error}");
            std::process::exit(1);
        })
    );
}

pub(super) fn parse_strict_bool(name: &str, value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "{name} must be a boolean (accepted true values: 1, true, yes, on; false values: 0, false, no, off), got {value:?}"
        )),
    }
}

/// Prepare the source tools before Cargo checks freshness. Macros track the
/// content identity, independently of descriptor hints and output paths.
fn apply_source_compiler_identity(
    cmd: &mut Command,
    ctx: &Context,
    output: &Path,
) -> Result<(), String> {
    use reserved_oxide_symbols::{SOURCE_COMPILER_HANDSHAKE_ENV, SOURCE_COMPILER_PROVENANCE_ENV};
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let mut helper = Command::new(executable);
    helper.arg("__native-compiler-handshake");
    apply_config_env(&mut helper, ctx);
    apply_ld_library_path(&mut helper, ctx);
    helper
        .env_remove(SOURCE_COMPILER_HANDSHAKE_ENV)
        .env_remove(SOURCE_COMPILER_PROVENANCE_ENV);
    for key in [
        "CUDA_OXIDE_NATIVE_COMPILER",
        "CUDA_OXIDE_DEBUG",
        "CUDA_OXIDE_TARGET",
        MATERIALIZER_HANDSHAKE_ENV,
    ] {
        if let Some((_, value)) = cmd.get_envs().find(|(name, _)| *name == key) {
            match value {
                Some(value) => {
                    helper.env(key, value);
                }
                None => {
                    helper.env_remove(key);
                }
            }
        }
    }
    let cache = output.join("cache/source-compiler-handshake/v1.json");
    if let Ok(json) = fs::read_to_string(&cache) {
        helper.env(SOURCE_COMPILER_HANDSHAKE_ENV, json);
    }
    let discovered = helper
        .output()
        .map_err(|error| format!("could not start source compiler discovery: {error}"))?;
    if !discovered.status.success() {
        return Err(format!(
            "native source compiler discovery failed: {}",
            String::from_utf8_lossy(&discovered.stderr).trim()
        ));
    }
    let handshake: Option<cuda_artifact_finalizer::SourceCompilerHandshakeV1> =
        serde_json::from_slice(&discovered.stdout)
            .map_err(|error| format!("invalid source compiler handshake: {error}"))?;
    match handshake {
        Some(handshake) if handshake.has_consistent_provenance() => {
            let json = serde_json::to_string(&handshake).map_err(|error| error.to_string())?;
            if let Some(parent) = cache.parent() {
                let _ = fs::create_dir_all(parent);
                let temp = cache.with_extension(format!("tmp.{}", std::process::id()));
                if fs::write(&temp, &json).is_ok() {
                    let _ = fs::rename(temp, &cache);
                }
            }
            cmd.env("CUDA_OXIDE_NATIVE_COMPILER", "llvm")
                .env(
                    SOURCE_COMPILER_PROVENANCE_ENV,
                    digest_hex(&handshake.provenance_sha256),
                )
                .env(SOURCE_COMPILER_HANDSHAKE_ENV, json);
        }
        Some(_) => return Err("inconsistent source compiler handshake".into()),
        None => {
            cmd.env("CUDA_OXIDE_NATIVE_COMPILER", "nvvm")
                .env(SOURCE_COMPILER_PROVENANCE_ENV, "nvvm")
                .env_remove(SOURCE_COMPILER_HANDSHAKE_ENV);
        }
    }
    Ok(())
}

pub fn print_source_compiler_handshake() {
    let result = discover_source_compiler_handshake();
    match result {
        Ok(handshake) => println!(
            "{}",
            serde_json::to_string(&handshake).expect("serializable source compiler handshake")
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn discover_source_compiler_handshake()
-> Result<Option<cuda_artifact_finalizer::SourceCompilerHandshakeV1>, String> {
    use cuda_artifact_finalizer::{
        DebugPolicy, LlvmNvptxCompiler, NativeCompilerPreference, PtxAssembler,
        SourceCompilerHandshakeV1, rust_llvm_library,
    };
    use reserved_oxide_symbols::SOURCE_COMPILER_HANDSHAKE_ENV;
    let requested = std::env::var("CUDA_OXIDE_NATIVE_COMPILER").ok();
    let preference =
        NativeCompilerPreference::parse(requested.as_deref()).map_err(|error| error.to_string())?;
    if preference == NativeCompilerPreference::Nvvm {
        return Ok(None);
    }
    let discovered = (|| -> Result<SourceCompilerHandshakeV1, String> {
        let target = std::env::var("CUDA_OXIDE_TARGET")
            .map_err(|_| "native source compilation needs a configured target")?;
        let target = target
            .parse::<cuda_artifact_finalizer::CudaArch>()
            .map_err(|error| error.to_string())?;
        if target.uses_legacy_llvm() {
            return Err(
                "LLVM native compilation requires a modern NVVM source dialect (sm_100 or newer)"
                    .into(),
            );
        }
        let debug = std::env::var("CUDA_OXIDE_DEBUG").ok();
        if debug
            .as_deref()
            .is_some_and(|value| DebugPolicy::parse_env_override(value) != Some(DebugPolicy::None))
        {
            return Err(
                "LLVM source compilation requires optimized device code without debug info".into(),
            );
        }
        let cached = std::env::var(SOURCE_COMPILER_HANDSHAKE_ENV)
            .ok()
            .and_then(|json| serde_json::from_str::<SourceCompilerHandshakeV1>(&json).ok())
            .filter(SourceCompilerHandshakeV1::has_consistent_provenance);
        let sysroot = crate::backend::get_rustc_sysroot()
            .ok_or("could not locate the Rust toolchain sysroot")?;
        let libdevice_path =
            cuda_artifact_finalizer::find_libdevice().map_err(|error| error.to_string())?;
        let libdevice = fs::read(&libdevice_path).map_err(|error| error.to_string())?;
        let llvm = LlvmNvptxCompiler::from_path_with_provenance(
            &rust_llvm_library(Path::new(&sysroot)).map_err(|error| error.to_string())?,
            &libdevice,
            cached.as_ref().map(|hint| &hint.llvm),
        )
        .map_err(|error| error.to_string())?;
        let assembler =
            PtxAssembler::discover_with_provenance(cached.as_ref().map(|hint| &hint.ptxas))
                .map_err(|error| error.to_string())?;
        if !assembler.supports_ptx90() {
            return Err("LLVM native compilation requires CUDA 13 or newer ptxas".into());
        }
        llvm.handshake(&assembler)
            .ok_or_else(|| "source compiler provenance changed".into())
    })();
    match discovered {
        Ok(handshake) => Ok(Some(handshake)),
        Err(_) if preference == NativeCompilerPreference::Auto => Ok(None),
        Err(error) => Err(error),
    }
}
