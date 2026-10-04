// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Publish the GPU manifest paired with Cargo's exact cached Rust library.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

type Error = Box<dyn std::error::Error>;

pub(super) fn build(command: &Command, root: &Path) -> Result<ExitStatus, Error> {
    let (mut command, print_json) = artifact_command(command)?;
    let mut child = command.stdout(Stdio::piped()).spawn()?;
    let mut artifacts = BTreeMap::<String, BTreeSet<PathBuf>>::new();
    // Always drain stdout and reap Cargo before reporting a protocol error.
    let mut protocol_error = None;
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                protocol_error = Some(error.to_string());
                break;
            }
        };
        if print_json {
            println!("{line}");
        }
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            if !print_json {
                println!("{line}");
            }
            continue;
        };
        match message["reason"].as_str() {
            Some("compiler-message") if !print_json => {
                if let Some(rendered) = message["message"]["rendered"].as_str() {
                    eprint!("{rendered}");
                }
            }
            Some("compiler-artifact") => {
                if let Some((name, path)) = library_artifact(&message) {
                    artifacts.entry(name).or_default().insert(path);
                }
            }
            _ => {}
        }
    }
    let status = child.wait()?;
    if !status.success() {
        return Ok(status);
    }
    if let Some(error) = protocol_error {
        return Err(error.into());
    }
    // Validate the entire selection before replacing any manifest aliases.
    let manifests = artifacts
        .into_iter()
        .map(|(name, metadata)| selection(root, &name, &metadata))
        .collect::<Result<Vec<_>, _>>()?;
    for manifest in manifests.into_iter().flatten() {
        manifest.publish()?;
    }
    Ok(status)
}

fn library_artifact(message: &Value) -> Option<(String, PathBuf)> {
    if message["profile"]["test"].as_bool() == Some(true) {
        return None;
    }
    let name = message["target"]["name"].as_str()?.replace('-', "_");
    let metadata = message["filenames"]
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .map(PathBuf::from)
        .find(|path| path.extension().is_some_and(|ext| ext == "rmeta"))?;
    Some((name, metadata))
}

fn artifact_command(original: &Command) -> Result<(Command, bool), Error> {
    let args: Vec<_> = original.get_args().map(OsString::from).collect();
    let mut output = Vec::new();
    let mut format = None;
    let mut index = 0;
    while index < args.len() && args[index] != "--" {
        if args[index] == "--message-format" {
            index += 1;
            format = Some(
                args.get(index)
                    .ok_or("missing Cargo message format")?
                    .to_string_lossy()
                    .into_owned(),
            );
        } else if let Some(value) = args[index]
            .to_str()
            .and_then(|arg| arg.strip_prefix("--message-format="))
        {
            format = Some(value.to_string());
        } else {
            output.push(args[index].clone());
        }
        index += 1;
    }
    let print_json = format
        .as_ref()
        .is_some_and(|format| format.split(',').any(|value| value.starts_with("json")));
    output.push(
        format!(
            "--message-format={}",
            if print_json {
                format.as_deref().unwrap()
            } else if format.as_deref() == Some("short") {
                "json-render-diagnostics,json-diagnostic-short"
            } else {
                "json-render-diagnostics"
            }
        )
        .into(),
    );
    output.extend_from_slice(&args[index..]);
    let mut command = Command::new(original.get_program());
    command.args(output);
    if let Some(cwd) = original.get_current_dir() {
        command.current_dir(cwd);
    }
    for (key, value) in original.get_envs() {
        if let Some(value) = value {
            command.env(key, value);
        } else {
            command.env_remove(key);
        }
    }
    Ok((command, print_json))
}

struct Publication {
    alias: PathBuf,
    bytes: Option<Vec<u8>>,
    changed: bool,
}

impl Publication {
    fn publish(self) -> Result<(), Error> {
        if !self.changed {
            return Ok(());
        }
        if let Some(bytes) = self.bytes {
            static NEXT_WRITE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let id = NEXT_WRITE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let temporary = self
                .alias
                .with_extension(format!("tmp.{}.{id}", std::process::id()));
            std::fs::write(&temporary, bytes)?;
            std::fs::rename(temporary, self.alias)?;
        } else if let Err(error) = std::fs::remove_file(self.alias)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            return Err(error.into());
        }
        Ok(())
    }
}

// Cargo can compile a library twice for build/proc-macro dependencies and
// the target. CPU-only copies do not retire a selected GPU-bearing copy.
fn selection(
    root: &Path,
    name: &str,
    metadata: &BTreeSet<PathBuf>,
) -> Result<Option<Publication>, Error> {
    let mut selected: Option<Publication> = None;
    for path in metadata {
        let Some(candidate) = prepare(root, name, path)? else {
            continue;
        };
        if let Some(current) = &selected {
            match (&current.bytes, &candidate.bytes) {
                (Some(first), Some(second)) if first != second => {
                    return Err(format!(
                        "multiple Rust libraries publish conflicting CUDA manifests for `{name}`"
                    )
                    .into());
                }
                (Some(_), _) => continue,
                _ => {}
            }
        }
        selected = Some(candidate);
    }
    Ok(selected)
}

fn prepare(root: &Path, name: &str, metadata: &Path) -> Result<Option<Publication>, Error> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err("invalid CUDA crate name in Cargo artifact".into());
    }
    let alias = root.join(format!("{name}.modules.json"));
    let path = metadata.with_extension("cuda-modules");
    let snapshot = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !alias.exists() => {
            return Ok(None);
        }
        Err(error) => {
            return Err(format!(
                "cannot read CUDA manifest snapshot {}: {error}; rebuild `{name}`",
                path.display()
            )
            .into());
        }
    };
    if snapshot.len() < 32 || Sha256::digest(&snapshot[32..]).as_slice() != &snapshot[..32] {
        return Err(format!(
            "corrupt CUDA manifest snapshot {}; rebuild `{name}`",
            path.display()
        )
        .into());
    }
    let snapshot: Value = serde_json::from_slice(&snapshot[32..])?;
    if snapshot["version"] != 1 || snapshot["crate"] != name {
        return Err("CUDA manifest snapshot does not match Cargo artifact".into());
    }
    let bytes = match snapshot
        .get("manifest")
        .ok_or("missing snapshot manifest")?
    {
        Value::Null => None,
        Value::String(text) => Some(text.as_bytes().to_vec()),
        _ => return Err("invalid snapshot manifest".into()),
    };
    if let Some(bytes) = &bytes {
        let manifest: Value = serde_json::from_slice(bytes)?;
        if manifest["version"] != 1 || manifest["crate"] != name {
            return Err("CUDA module manifest does not match Cargo artifact".into());
        }
        let modules = manifest["modules"]
            .as_object()
            .ok_or("missing CUDA modules")?;
        if std::fs::read(&alias).ok().as_deref() == Some(bytes.as_slice()) {
            return Ok(Some(Publication {
                alias,
                bytes: Some(bytes.clone()),
                changed: false,
            }));
        }
        for entry in modules.values() {
            let path = Path::new(entry["path"].as_str().ok_or("missing CUDA module path")?);
            if path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
                || path.as_os_str().is_empty()
            {
                return Err("CUDA module path escapes artifact directory".into());
            }
            let payload = root.join(path);
            if !payload.canonicalize()?.starts_with(root.canonicalize()?) {
                return Err("CUDA module symlink escapes artifact directory".into());
            }
            let digest = format!("{:x}", Sha256::digest(std::fs::read(payload)?));
            if entry["sha256"] != digest {
                return Err("corrupt CUDA module payload; rebuild the library".into());
            }
        }
    }
    let changed = bytes.is_some() || alias.exists();
    Ok(Some(Publication {
        alias,
        bytes,
        changed,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    struct Fixture {
        root: PathBuf,
        metadata: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "oxide-snapshot-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&root).unwrap();
            let metadata = root.join("libtoy-a.rmeta");
            Self { root, metadata }
        }
        fn snapshot(&self, manifest: Value) {
            let bytes = serde_json::to_vec(&json!({"version":1,"crate":"toy","manifest":manifest}))
                .unwrap();
            let mut checked = Sha256::digest(&bytes).to_vec();
            checked.extend(bytes);
            std::fs::write(self.metadata.with_extension("cuda-modules"), checked).unwrap();
        }
        fn manifest(&self, payload: &[u8]) -> String {
            let digest = format!("{:x}", Sha256::digest(payload));
            std::fs::write(self.root.join(format!("{digest}.cubin")), payload).unwrap();
            json!({"version":1,"crate":"toy","modules":{"k":{"path":format!("{digest}.cubin"),"sha256":digest}}}).to_string()
        }
        fn alias(&self) -> PathBuf {
            self.root.join("toy.modules.json")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
    #[test]
    fn restores_exact_cached_configuration_and_retires_native_alias() {
        let f = Fixture::new();
        let first = f.manifest(b"first");
        let second = f.manifest(b"second");
        f.snapshot(json!(first));
        std::fs::write(f.alias(), second).unwrap();
        prepare(&f.root, "toy", &f.metadata)
            .unwrap()
            .unwrap()
            .publish()
            .unwrap();
        assert_eq!(std::fs::read_to_string(f.alias()).unwrap(), first);
        assert!(
            !prepare(&f.root, "toy", &f.metadata)
                .unwrap()
                .unwrap()
                .changed
        );
        f.snapshot(Value::Null);
        prepare(&f.root, "toy", &f.metadata)
            .unwrap()
            .unwrap()
            .publish()
            .unwrap();
        assert!(!f.alias().exists());
    }
    #[test]
    fn corrupt_or_missing_snapshot_never_replaces_existing_manifest() {
        let f = Fixture::new();
        let first = f.manifest(b"first");
        std::fs::write(f.alias(), &first).unwrap();
        assert!(prepare(&f.root, "toy", &f.metadata).is_err());
        f.snapshot(json!(first));
        let path = f.metadata.with_extension("cuda-modules");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[32] ^= 1;
        std::fs::write(path, bytes).unwrap();
        assert!(prepare(&f.root, "toy", &f.metadata).is_err());
        assert_eq!(std::fs::read_to_string(f.alias()).unwrap(), first);
    }
    #[test]
    fn missing_corrupt_or_escaping_payload_is_rejected_before_publication() {
        let f = Fixture::new();
        let first = f.manifest(b"first");
        let second = f.manifest(b"second");
        std::fs::write(f.alias(), &first).unwrap();
        f.snapshot(json!(second));
        let m: Value = serde_json::from_str(&second).unwrap();
        let path = f.root.join(m["modules"]["k"]["path"].as_str().unwrap());
        std::fs::write(&path, b"bad").unwrap();
        assert!(prepare(&f.root, "toy", &f.metadata).is_err());
        std::fs::remove_file(path).unwrap();
        assert!(prepare(&f.root, "toy", &f.metadata).is_err());
        let mut m = m;
        m["modules"]["k"]["path"] = json!("../outside.cubin");
        f.snapshot(json!(m.to_string()));
        assert!(prepare(&f.root, "toy", &f.metadata).is_err());
        assert_eq!(std::fs::read_to_string(f.alias()).unwrap(), first);
    }
    #[test]
    fn cpu_only_duplicate_does_not_retire_a_fresh_native_copy() {
        let mut f = Fixture::new();
        let first = f.manifest(b"first");
        f.snapshot(json!(first));
        std::fs::write(f.alias(), &first).unwrap();
        let native = f.metadata.clone();
        f.metadata = f.root.join("libtoy-host.rmeta");
        f.snapshot(Value::Null);
        let mut paths = BTreeSet::from([native, f.metadata.clone()]);
        let selected = selection(&f.root, "toy", &paths).unwrap().unwrap();
        assert_eq!(selected.bytes.as_deref(), Some(first.as_bytes()));
        selected.publish().unwrap();
        assert_eq!(std::fs::read_to_string(f.alias()).unwrap(), first);
        f.metadata = f.root.join("libtoy-conflict.rmeta");
        let other = f.manifest(b"other");
        f.snapshot(json!(other));
        paths.insert(f.metadata.clone());
        assert!(selection(&f.root, "toy", &paths).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn failed_cargo_does_not_publish_and_successful_fresh_cargo_does() {
        let f = Fixture::new();
        let first = f.manifest(b"first");
        let second = f.manifest(b"second");
        f.snapshot(json!(first));
        std::fs::write(f.alias(), &second).unwrap();
        let event = json!({"reason":"compiler-artifact", "target":{"name":"toy"},
            "profile":{"test":false}, "filenames":[f.metadata], "fresh":true})
        .to_string();
        for code in [1, 0] {
            let mut command = Command::new("sh");
            command
                .args([
                    "-c",
                    &format!("printf '%s\\n' \"$SNAPSHOT_EVENT\"; exit {code}"),
                ])
                .env("SNAPSHOT_EVENT", &event);
            let status = build(&command, &f.root).unwrap();
            assert_eq!(status.code(), Some(code));
            assert_eq!(
                std::fs::read_to_string(f.alias()).unwrap(),
                if code == 0 {
                    first.as_str()
                } else {
                    second.as_str()
                }
            );
        }
    }

    #[test]
    fn cargo_artifact_uses_hashed_metadata_and_preserves_json_requests() {
        let message = json!({"target":{"name":"toy-lib"},"profile":{"test":false},"filenames":["target/libtoy_lib.rlib","target/build/toy-lib/abc/out/libtoy_lib-abc.rmeta"],"fresh":true});
        assert_eq!(
            library_artifact(&message),
            Some((
                "toy_lib".into(),
                PathBuf::from("target/build/toy-lib/abc/out/libtoy_lib-abc.rmeta")
            ))
        );
        let mut original = Command::new("cargo");
        original
            .args(["build", "--release"])
            .env("OXIDE_TEST", "kept")
            .env_remove("OXIDE_REMOVED")
            .current_dir("/tmp");
        let (command, print_json) = artifact_command(&original).unwrap();
        assert!(!print_json);
        assert!(
            command
                .get_args()
                .any(|arg| arg == "--message-format=json-render-diagnostics")
        );
        assert_eq!(
            command.get_envs().collect::<Vec<_>>(),
            original.get_envs().collect::<Vec<_>>()
        );
        assert_eq!(command.get_current_dir(), original.get_current_dir());
        original.args(["--message-format", "json"]);
        let (command, print_json) = artifact_command(&original).unwrap();
        assert!(print_json);
        assert!(command.get_args().any(|arg| arg == "--message-format=json"));
        let mut short = Command::new("cargo");
        short.args(["build", "--message-format=short"]);
        let (command, print_json) = artifact_command(&short).unwrap();
        assert!(!print_json);
        assert!(
            command
                .get_args()
                .any(|arg| arg == "--message-format=json-render-diagnostics,json-diagnostic-short")
        );
    }
}
