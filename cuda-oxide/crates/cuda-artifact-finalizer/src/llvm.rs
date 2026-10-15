/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! In-process source-IR merger and NVPTX compiler.
//!
//! Each compilation owns its LLVM context. Precision reflection is resolved in
//! that module, rather than changing LLVM's process-global command-line options
//! while rustc is also compiling host code.
use crate::provenance::{StableDigest, digest_bytes, digest_file_handle};
use crate::{
    DebugPolicy, FinalizationOptions, NamedInput, PinnedToolProvenance, PtxAssembler,
    ToolFileIdentity,
};
use libloading::Library;
use std::collections::BTreeSet;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

type Ref = *mut c_void;
type Message = *mut c_char;

#[derive(Debug, thiserror::Error)]
#[error("LLVM NVPTX: {0}")]
pub struct LlvmNvptxError(String);

fn failure(message: impl Into<String>) -> LlvmNvptxError {
    LlvmNvptxError(message.into())
}

macro_rules! api {
    ($($field:ident: $signature:ty = $name:literal;)*) => {
        struct Api {
            $($field: $signature,)*
            _library: Library,
            file: File,
            identity: ToolFileIdentity,
            digest: [u8; 32],
        }
        impl Api {
            unsafe fn open(path: &Path, expected: Option<&PinnedToolProvenance>) -> Result<Self, LlvmNvptxError> {
                let file = File::open(path).map_err(|e| failure(e.to_string()))?;
                let identity = ToolFileIdentity::capture(&file)
                    .ok_or_else(|| failure("cannot identify LLVM library"))?;
                let digest = match expected.filter(|hint| hint.file.has_unix_identity() && hint.file == identity) {
                    Some(hint) => hint.sha256,
                    None => digest_file_handle(&file).map_err(|e| failure(e.to_string()))?,
                };
                if ToolFileIdentity::capture(&file) != Some(identity) { return Err(failure("LLVM library changed while hashing")); }
                // Pin the opened inode on Linux, including across path replacement.
                #[cfg(target_os = "linux")]
                let library = {
                    use std::os::fd::AsRawFd;
                    unsafe { Library::new(format!("/proc/self/fd/{}", file.as_raw_fd())) }
                };
                #[cfg(not(target_os = "linux"))]
                let library = unsafe { Library::new(path) };
                let library = library.map_err(|e| failure(e.to_string()))?;
                Ok(Self {
                    $($field: unsafe { symbol::<$signature>(&library, $name)? },)*
                    _library: library, file, identity, digest,
                })
            }
        }
    }
}

unsafe fn symbol<T: Copy>(library: &Library, name: &str) -> Result<T, LlvmNvptxError> {
    for spelling in [
        name.to_string(),
        format!("__{name}_13_0"),
        format!("__{name}_12_0"),
    ] {
        if let Ok(value) = unsafe { library.get::<T>(spelling.as_bytes()) } {
            return Ok(*value);
        }
    }
    Err(failure(format!("missing C API symbol {name}")))
}

api! {
    version: unsafe extern "C" fn(*mut c_uint, *mut c_uint, *mut c_uint) = "LLVMGetVersion";
    init_info: unsafe extern "C" fn() = "LLVMInitializeNVPTXTargetInfo";
    init_target: unsafe extern "C" fn() = "LLVMInitializeNVPTXTarget";
    init_mc: unsafe extern "C" fn() = "LLVMInitializeNVPTXTargetMC";
    init_printer: unsafe extern "C" fn() = "LLVMInitializeNVPTXAsmPrinter";
    context: unsafe extern "C" fn() -> Ref = "LLVMContextCreate";
    dispose_context: unsafe extern "C" fn(Ref) = "LLVMContextDispose";
    buffer: unsafe extern "C" fn(*const c_char, usize, *const c_char) -> Ref = "LLVMCreateMemoryBufferWithMemoryRangeCopy";
    dispose_buffer: unsafe extern "C" fn(Ref) = "LLVMDisposeMemoryBuffer";
    buffer_start: unsafe extern "C" fn(Ref) -> *const u8 = "LLVMGetBufferStart";
    buffer_size: unsafe extern "C" fn(Ref) -> usize = "LLVMGetBufferSize";
    parse_ir: unsafe extern "C" fn(Ref, Ref, *mut Ref, *mut Message) -> c_int = "LLVMParseIRInContext";
    parse_bc: unsafe extern "C" fn(Ref, Ref, *mut Ref) -> c_int = "LLVMParseBitcodeInContext2";
    dispose_module: unsafe extern "C" fn(Ref) = "LLVMDisposeModule";
    link: unsafe extern "C" fn(Ref, Ref) -> c_int = "LLVMLinkModules2";
    verify: unsafe extern "C" fn(Ref, c_int, *mut Message) -> c_int = "LLVMVerifyModule";
    dispose_message: unsafe extern "C" fn(Message) = "LLVMDisposeMessage";
    first_function: unsafe extern "C" fn(Ref) -> Ref = "LLVMGetFirstFunction";
    next_function: unsafe extern "C" fn(Ref) -> Ref = "LLVMGetNextFunction";
    name: unsafe extern "C" fn(Ref, *mut usize) -> *const u8 = "LLVMGetValueName2";
    blocks: unsafe extern "C" fn(Ref) -> c_uint = "LLVMCountBasicBlocks";
    linkage: unsafe extern "C" fn(Ref, c_int) = "LLVMSetLinkage";
    first_block: unsafe extern "C" fn(Ref) -> Ref = "LLVMGetFirstBasicBlock";
    next_block: unsafe extern "C" fn(Ref) -> Ref = "LLVMGetNextBasicBlock";
    first_instruction: unsafe extern "C" fn(Ref) -> Ref = "LLVMGetFirstInstruction";
    next_instruction: unsafe extern "C" fn(Ref) -> Ref = "LLVMGetNextInstruction";
    opcode: unsafe extern "C" fn(Ref) -> c_uint = "LLVMGetInstructionOpcode";
    called: unsafe extern "C" fn(Ref) -> Ref = "LLVMGetCalledValue";
    operand: unsafe extern "C" fn(Ref, c_uint) -> Ref = "LLVMGetOperand";
    constant_expression: unsafe extern "C" fn(Ref) -> Ref = "LLVMIsAConstantExpr";
    global: unsafe extern "C" fn(Ref) -> Ref = "LLVMIsAGlobalVariable";
    initializer: unsafe extern "C" fn(Ref) -> Ref = "LLVMGetInitializer";
    constant_string: unsafe extern "C" fn(Ref) -> c_int = "LLVMIsConstantString";
    string: unsafe extern "C" fn(Ref, *mut usize) -> *const u8 = "LLVMGetAsString";
    type_of: unsafe extern "C" fn(Ref) -> Ref = "LLVMTypeOf";
    integer: unsafe extern "C" fn(Ref, u64, c_int) -> Ref = "LLVMConstInt";
    replace: unsafe extern "C" fn(Ref, Ref) = "LLVMReplaceAllUsesWith";
    erase: unsafe extern "C" fn(Ref) = "LLVMInstructionEraseFromParent";
    get_target: unsafe extern "C" fn(*const c_char, *mut Ref, *mut Message) -> c_int = "LLVMGetTargetFromTriple";
    machine: unsafe extern "C" fn(Ref, *const c_char, *const c_char, *const c_char, c_int, c_int, c_int) -> Ref = "LLVMCreateTargetMachine";
    dispose_machine: unsafe extern "C" fn(Ref) = "LLVMDisposeTargetMachine";
    set_target: unsafe extern "C" fn(Ref, *const c_char) = "LLVMSetTarget";
    pass_options: unsafe extern "C" fn() -> Ref = "LLVMCreatePassBuilderOptions";
    dispose_options: unsafe extern "C" fn(Ref) = "LLVMDisposePassBuilderOptions";
    vectorize: unsafe extern "C" fn(Ref, c_int) = "LLVMPassBuilderOptionsSetLoopVectorization";
    slp: unsafe extern "C" fn(Ref, c_int) = "LLVMPassBuilderOptionsSetSLPVectorization";
    passes: unsafe extern "C" fn(Ref, *const c_char, Ref, Ref) -> Ref = "LLVMRunPasses";
    error_message: unsafe extern "C" fn(Ref) -> Message = "LLVMGetErrorMessage";
    dispose_error: unsafe extern "C" fn(Message) = "LLVMDisposeErrorMessage";
    emit: unsafe extern "C" fn(Ref, Ref, c_int, *mut Message, *mut Ref) -> c_int = "LLVMTargetMachineEmitToMemoryBuffer";
}

struct Owned<'a> {
    _api: &'a Api,
    value: Ref,
    dispose: unsafe extern "C" fn(Ref),
}
impl Drop for Owned<'_> {
    fn drop(&mut self) {
        if !self.value.is_null() {
            unsafe { (self.dispose)(self.value) }
        }
    }
}
impl Owned<'_> {
    fn release(mut self) -> Ref {
        std::mem::replace(&mut self.value, std::ptr::null_mut())
    }
}

impl Api {
    fn own(&self, value: Ref, dispose: unsafe extern "C" fn(Ref)) -> Owned<'_> {
        Owned {
            _api: self,
            value,
            dispose,
        }
    }
    unsafe fn message(&self, message: Message) -> String {
        if message.is_null() {
            return "unknown LLVM error".into();
        }
        let text = unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        unsafe { (self.dispose_message)(message) };
        text
    }
    unsafe fn value_name(&self, value: Ref) -> Vec<u8> {
        let mut length = 0;
        let pointer = unsafe { (self.name)(value, &mut length) };
        if length == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(pointer, length) }.to_vec()
        }
    }
    unsafe fn parse(
        &self,
        context: Ref,
        input: NamedInput<'_>,
    ) -> Result<Owned<'_>, LlvmNvptxError> {
        let name = CString::new(input.name).map_err(|e| failure(e.to_string()))?;
        let buffer = self.own(
            unsafe {
                (self.buffer)(
                    input.bytes.as_ptr().cast(),
                    input.bytes.len(),
                    name.as_ptr(),
                )
            },
            self.dispose_buffer,
        );
        let mut module = std::ptr::null_mut();
        let mut message = std::ptr::null_mut();
        let status = if input.bytes.starts_with(b"BC\xc0\xde") {
            unsafe { (self.parse_bc)(context, buffer.value, &mut module) }
        } else {
            // The textual parser consumes its buffer on both success and failure.
            unsafe { (self.parse_ir)(context, buffer.release(), &mut module, &mut message) }
        };
        let module = self.own(module, self.dispose_module);
        if status != 0 {
            return Err(failure(format!("{}: {}", input.name, unsafe {
                self.message(message)
            })));
        }
        if !message.is_null() {
            unsafe { (self.dispose_message)(message) };
        }
        Ok(module)
    }

    unsafe fn reflect_precision(&self, module: Ref) {
        let mut function = unsafe { (self.first_function)(module) };
        while !function.is_null() {
            let mut block = unsafe { (self.first_block)(function) };
            while !block.is_null() {
                let mut instruction = unsafe { (self.first_instruction)(block) };
                while !instruction.is_null() {
                    let next = unsafe { (self.next_instruction)(instruction) };
                    if unsafe { (self.opcode)(instruction) } == 45 {
                        let called = unsafe { (self.called)(instruction) };
                        let name = unsafe { self.value_name(called) };
                        if name == b"__nvvm_reflect" || name == b"llvm.nvvm.reflect" {
                            let mut argument = unsafe { (self.operand)(instruction, 0) };
                            while !unsafe { (self.constant_expression)(argument) }.is_null() {
                                argument = unsafe { (self.operand)(argument, 0) };
                            }
                            if !unsafe { (self.global)(argument) }.is_null() {
                                let initializer = unsafe { (self.initializer)(argument) };
                                if !initializer.is_null()
                                    && unsafe { (self.constant_string)(initializer) } != 0
                                {
                                    let mut length = 0;
                                    let pointer =
                                        unsafe { (self.string)(initializer, &mut length) };
                                    let key =
                                        unsafe { std::slice::from_raw_parts(pointer, length) };
                                    if key == b"__CUDA_PREC_DIV\0" || key == b"__CUDA_PREC_SQRT\0" {
                                        let one = unsafe {
                                            (self.integer)((self.type_of)(instruction), 1, 0)
                                        };
                                        unsafe {
                                            (self.replace)(instruction, one);
                                            (self.erase)(instruction);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    instruction = next;
                }
                block = unsafe { (self.next_block)(block) };
            }
            function = unsafe { (self.next_function)(function) };
        }
    }
}

#[derive(Clone)]
pub struct LlvmNvptxCompiler {
    api: Arc<Api>,
    libdevice: Arc<[u8]>,
}

/// Wall times for one fresh source compilation, separate from native assembly.
pub struct LlvmPtxReport {
    pub ptx: Vec<u8>,
    pub parse_merge_seconds: f64,
    pub optimize_seconds: f64,
    pub emit_ptx_seconds: f64,
}

/// Default native source compiler selection, with an explicit NVIDIA opt-out.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeCompilerPreference {
    Auto,
    Llvm,
    Nvvm,
}

impl NativeCompilerPreference {
    pub fn parse(value: Option<&str>) -> Result<Self, LlvmNvptxError> {
        match value {
            None | Some("auto") => Ok(Self::Auto),
            Some("llvm") => Ok(Self::Llvm),
            Some("nvvm") => Ok(Self::Nvvm),
            _ => Err(failure(
                "CUDA_OXIDE_NATIVE_COMPILER must be auto, llvm, or nvvm",
            )),
        }
    }
}

/// Exact native tools passed from cargo-oxide to the backend. Descriptor
/// identities accelerate verification; semantic identity uses content hashes.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceCompilerHandshakeV1 {
    pub version: u32,
    pub llvm: PinnedToolProvenance,
    pub ptxas: PinnedToolProvenance,
    pub libdevice_sha256: [u8; 32],
    pub provenance_sha256: [u8; 32],
}

impl SourceCompilerHandshakeV1 {
    pub fn has_consistent_provenance(&self) -> bool {
        self.version == 1
            && self.provenance_sha256
                == source_tool_digest(
                    &self.llvm.sha256,
                    &self.ptxas.sha256,
                    &self.libdevice_sha256,
                )
    }
}

fn source_tool_digest(llvm: &[u8; 32], ptxas: &[u8; 32], libdevice: &[u8; 32]) -> [u8; 32] {
    StableDigest::new()
        .field("route", b"llvm-source-nvptx-v1")
        .field("llvm", llvm)
        .field("ptxas", ptxas)
        .field("libdevice", libdevice)
        .finish()
}

/// Find the Rust toolchain's LLVM DSO rather than depending on a system LLVM
/// installation. Rust also ships a differently named copy; pick its soname.
pub fn rust_llvm_library(sysroot: &Path) -> Result<PathBuf, LlvmNvptxError> {
    let directory = sysroot.join("lib");
    let mut libraries = BTreeSet::new();
    for entry in std::fs::read_dir(&directory).map_err(|e| failure(e.to_string()))? {
        let entry = entry.map_err(|e| failure(e.to_string()))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if (name.starts_with("libLLVM.so.") && name.contains("-rust-")) || name == "libLLVM.dylib" {
            libraries
                .insert(std::fs::canonicalize(entry.path()).map_err(|e| failure(e.to_string()))?);
        }
    }
    if libraries.len() != 1 {
        return Err(failure(format!(
            "expected one Rust LLVM library in {}",
            directory.display()
        )));
    }
    Ok(libraries.pop_first().unwrap())
}

impl LlvmNvptxCompiler {
    pub fn from_path(path: &Path, libdevice: &[u8]) -> Result<Self, LlvmNvptxError> {
        Self::from_path_with_provenance(path, libdevice, None)
    }

    pub fn from_path_with_provenance(
        path: &Path,
        libdevice: &[u8],
        expected: Option<&PinnedToolProvenance>,
    ) -> Result<Self, LlvmNvptxError> {
        static INITIALIZE: Mutex<()> = Mutex::new(());
        let _guard = INITIALIZE.lock().map_err(|e| failure(e.to_string()))?;
        let api = unsafe { Api::open(path, expected)? };
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        unsafe {
            (api.version)(&mut major, &mut minor, &mut patch);
        }
        if major < 23 {
            return Err(failure(format!(
                "LLVM {major}.{minor}.{patch} is too old for this route"
            )));
        }
        unsafe {
            (api.init_info)();
            (api.init_target)();
            (api.init_mc)();
            (api.init_printer)();
        }
        Ok(Self {
            api: Arc::new(api),
            libdevice: libdevice.into(),
        })
    }

    pub fn handshake(&self, assembler: &PtxAssembler) -> Option<SourceCompilerHandshakeV1> {
        if ToolFileIdentity::capture(&self.api.file)? != self.api.identity {
            return None;
        }
        let llvm = PinnedToolProvenance {
            sha256: self.api.digest,
            file: self.api.identity,
        };
        let ptxas = assembler.pinned_tool_provenance()?;
        let libdevice_sha256 = digest_bytes(&self.libdevice);
        Some(SourceCompilerHandshakeV1 {
            version: 1,
            llvm,
            ptxas,
            libdevice_sha256,
            provenance_sha256: source_tool_digest(&llvm.sha256, &ptxas.sha256, &libdevice_sha256),
        })
    }

    pub fn artifact_digest(
        &self,
        inputs: &[NamedInput<'_>],
        kernels: &BTreeSet<String>,
        options: &FinalizationOptions,
    ) -> Option<[u8; 32]> {
        if ToolFileIdentity::capture(&self.api.file)? != self.api.identity {
            return None;
        }
        let mut digest = StableDigest::new().field("route", b"llvm-source-nvptx-v1");
        digest = digest
            .field("llvm", self.api.digest)
            .field("libdevice", digest_bytes(&self.libdevice));
        digest = digest.field("options", format!("{:?}", options).as_bytes());
        for input in inputs {
            digest = digest
                .field("name", input.name.as_bytes())
                .field("source", digest_bytes(input.bytes));
        }
        for kernel in kernels {
            digest = digest.field("kernel", kernel.as_bytes());
        }
        Some(digest.finish())
    }

    pub fn compile(
        &self,
        inputs: &[NamedInput<'_>],
        kernels: &BTreeSet<String>,
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, LlvmNvptxError> {
        Ok(self.compile_with_report(inputs, kernels, options)?.ptx)
    }

    pub fn compile_with_report(
        &self,
        inputs: &[NamedInput<'_>],
        kernels: &BTreeSet<String>,
        options: &FinalizationOptions,
    ) -> Result<LlvmPtxReport, LlvmNvptxError> {
        if inputs.is_empty() || kernels.is_empty() {
            return Err(failure("source inputs and entry points are required"));
        }
        if options.debug_policy() != DebugPolicy::None {
            return Err(failure(
                "LLVM source compilation requires optimized device code without debug info",
            ));
        }
        if self.artifact_digest(inputs, kernels, options).is_none() {
            return Err(failure("LLVM library changed during compilation"));
        }
        let api = &self.api;
        // All module, pass, target-machine, and buffer handles below belong to
        // this call's context; no handle is passed to another worker.
        unsafe {
            let parse_started = Instant::now();
            let context = api.own((api.context)(), api.dispose_context);
            let module = api.parse(context.value, inputs[0])?;
            for input in inputs
                .iter()
                .skip(1)
                .copied()
                .chain([NamedInput::new("libdevice.10.bc", &self.libdevice)])
            {
                let part = api.parse(context.value, input)?;
                if (api.link)(module.value, part.release()) != 0 {
                    return Err(failure(format!("cannot link {}", input.name)));
                }
            }
            let mut entries = BTreeSet::new();
            let mut function = (api.first_function)(module.value);
            while !function.is_null() {
                if (api.blocks)(function) != 0 {
                    let name = String::from_utf8_lossy(&api.value_name(function)).into_owned();
                    if kernels.contains(&name) {
                        entries.insert(name);
                    }
                }
                function = (api.next_function)(function);
            }
            if entries != *kernels {
                return Err(failure("a requested kernel has no definition"));
            }
            api.reflect_precision(module.value);
            let mut function = (api.first_function)(module.value);
            while !function.is_null() {
                if (api.blocks)(function) != 0
                    && !kernels
                        .contains(&String::from_utf8_lossy(&api.value_name(function)).into_owned())
                {
                    (api.linkage)(function, 8);
                }
                function = (api.next_function)(function);
            }
            let triple = c"nvptx64-nvidia-cuda";
            let cpu = CString::new(options.target().sm()).map_err(|e| failure(e.to_string()))?;
            let mut target = std::ptr::null_mut();
            let mut message = std::ptr::null_mut();
            if (api.get_target)(triple.as_ptr(), &mut target, &mut message) != 0 {
                return Err(failure(api.message(message)));
            }
            (api.set_target)(module.value, triple.as_ptr());
            let machine = api.own(
                (api.machine)(
                    target,
                    triple.as_ptr(),
                    cpu.as_ptr(),
                    c"+ptx90".as_ptr(),
                    3,
                    0,
                    0,
                ),
                api.dispose_machine,
            );
            if machine.value.is_null() {
                return Err(failure("cannot create NVPTX target machine"));
            }
            let pass_options = api.own((api.pass_options)(), api.dispose_options);
            (api.vectorize)(pass_options.value, 0);
            (api.slp)(pass_options.value, 0);
            let parse_merge_seconds = parse_started.elapsed().as_secs_f64();
            let optimize_started = Instant::now();
            let error = (api.passes)(
                module.value,
                c"default<O3>".as_ptr(),
                machine.value,
                pass_options.value,
            );
            if !error.is_null() {
                let message = (api.error_message)(error);
                let text = CStr::from_ptr(message).to_string_lossy().into_owned();
                (api.dispose_error)(message);
                return Err(failure(text));
            }
            let optimize_seconds = optimize_started.elapsed().as_secs_f64();
            let emit_started = Instant::now();
            if (api.verify)(module.value, 2, &mut message) != 0 {
                return Err(failure(api.message(message)));
            }
            if !message.is_null() {
                (api.dispose_message)(message);
                message = std::ptr::null_mut();
            }
            let mut generated = std::ptr::null_mut();
            let status = (api.emit)(machine.value, module.value, 0, &mut message, &mut generated);
            let generated = api.own(generated, api.dispose_buffer);
            if status != 0 {
                return Err(failure(api.message(message)));
            }
            if !message.is_null() {
                (api.dispose_message)(message);
            }
            let output = std::slice::from_raw_parts(
                (api.buffer_start)(generated.value),
                (api.buffer_size)(generated.value),
            )
            .to_vec();
            if self.artifact_digest(inputs, kernels, options).is_none() {
                return Err(failure("LLVM library changed during compilation"));
            }
            Ok(LlvmPtxReport {
                ptx: output,
                parse_merge_seconds,
                optimize_seconds,
                emit_ptx_seconds: emit_started.elapsed().as_secs_f64(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_compiler_preference_has_default_and_nvidia_optout() {
        assert_eq!(
            NativeCompilerPreference::parse(None).unwrap(),
            NativeCompilerPreference::Auto
        );
        assert_eq!(
            NativeCompilerPreference::parse(Some("auto")).unwrap(),
            NativeCompilerPreference::Auto
        );
        assert_eq!(
            NativeCompilerPreference::parse(Some("nvvm")).unwrap(),
            NativeCompilerPreference::Nvvm
        );
        assert_eq!(
            NativeCompilerPreference::parse(Some("llvm")).unwrap(),
            NativeCompilerPreference::Llvm
        );
        assert!(NativeCompilerPreference::parse(Some("other")).is_err());
    }

    #[test]
    fn source_handshake_rejects_changed_content_hashes_and_wire_version() {
        let file = ToolFileIdentity {
            length: 1,
            modified_seconds: 2,
            modified_nanoseconds: 3,
            device: Some(4),
            inode: Some(5),
            change_time_seconds: Some(6),
            change_time_nanoseconds: Some(7),
        };
        let base = SourceCompilerHandshakeV1 {
            version: 1,
            llvm: PinnedToolProvenance {
                sha256: [1; 32],
                file,
            },
            ptxas: PinnedToolProvenance {
                sha256: [2; 32],
                file,
            },
            libdevice_sha256: [3; 32],
            provenance_sha256: source_tool_digest(&[1; 32], &[2; 32], &[3; 32]),
        };
        assert!(base.has_consistent_provenance());
        for field in 0..4 {
            let mut changed = base;
            match field {
                0 => changed.version += 1,
                1 => changed.llvm.sha256[0] ^= 1,
                2 => changed.ptxas.sha256[0] ^= 1,
                _ => changed.libdevice_sha256[0] ^= 1,
            }
            assert!(!changed.has_consistent_provenance());
        }
        let mut descriptor_only = base;
        descriptor_only.llvm.file.inode = Some(8);
        assert!(descriptor_only.has_consistent_provenance());
    }
}
