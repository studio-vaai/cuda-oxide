#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU regression for device-file builds paired with a separate host build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cargo-oxide', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / 'crates/rustc-codegen-cuda/examples/incremental_modules'
source = root / 'src/lib.rs'
original = source.read_text()
args.output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_OXIDE_MODULE_FILES_ONLY='1',
           CUDA_OXIDE_REUSE_HOST_FOR_KERNEL_EDITS='1', CARGO_PROFILE_RELEASE_DEBUG='0')
for key in ('CUDA_OXIDE_PTX_DIR', 'CUDA_OXIDE_DEVICE_ONLY', 'CUDA_OXIDE_KERNELS_ONLY'):
    env.pop(key, None)
metadata = json.loads(subprocess.check_output([
    'cargo', 'metadata', '--format-version=1', '--no-deps',
    '--manifest-path', str(root / 'Cargo.toml')], cwd=root, env=env))
target = Path(metadata['target_directory'])
artifact_root = target / 'release/oxide'
binary = target / 'release/incremental_modules'
records = []


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(label, device_only=False, success=True):
    command = [args.cargo_oxide]
    if device_only:
        command += ['--device-only', 'incremental_modules']
    command += ['build', '--', '--release', '--manifest-path', str(root / 'Cargo.toml')]
    if device_only:
        command += ['--lib']
    log = args.output / (label + '.log')
    start = time.perf_counter()
    with log.open('w') as stream:
        code = subprocess.call(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(label=label, seconds=time.perf_counter() - start, exit_code=code)
    records.append(record)
    print(json.dumps(record), flush=True)
    if (code == 0) != success:
        raise RuntimeError(log.read_text()[-8000:])
    if not success:
        return None
    if device_only:
        assert 'device-only files: CPU codegen omitted' in log.read_text()
        assert 'host proof' not in log.read_text()
    manifest = json.loads((artifact_root / 'incremental_modules.modules.json').read_text())
    (args.output / (label + '.manifest.json')).write_text(json.dumps(manifest, indent=2))
    return manifest


def run(delta=0, host_value=7):
    subprocess.run([str(binary), '--files', f'--first-delta={delta}',
                    f'--second-delta={delta}', f'--host-value={host_value}'],
                   cwd=root, env=env, check=True)


def images(manifest):
    return {name: entry['sha256'] for name, entry in manifest['modules'].items()}


try:
    # No host seed is required: this mode deliberately creates no host library.
    device_baseline = build('device_baseline', device_only=True)
    baseline = build('normal_baseline')
    run()
    assert images(device_baseline) == images(baseline)
    host_artifacts = {path: digest(path) for path in (target / 'release').rglob('*')
                      if path.is_file() and (path.suffix in ('.rlib', '.rmeta') or path == binary)}
    assert binary in host_artifacts
    delta = 100 + time.time_ns() % 1000000
    source.write_text(original.replace('x: value + 2', f'x: value + {2 + delta}', 1))
    changed = build('device_helper_edit', device_only=True)
    run(delta)
    assert all(not entry['link_hit'] for entry in changed['modules'].values())
    assert all(entry['sha256'] != baseline['modules'][name]['sha256']
               for name, entry in changed['modules'].items())
    assert {path: digest(path) for path in host_artifacts} == host_artifacts

    # Host edits do not approve reuse of an old host binary: the caller must
    # build that host separately. Device-only outputs cannot replace its rlib.
    edited = source.read_text().replace('fn host_value() -> u32 {\n    7\n}',
                                      'fn host_value() -> u32 {\n    8\n}')
    assert edited != source.read_text()
    source.write_text(edited)
    host_edit = build('device_host_edit', device_only=True)
    assert images(host_edit) == images(changed)
    assert {path: digest(path) for path in host_artifacts} == host_artifacts
    run(delta)
    build('separate_host_rebuild')
    run(delta, host_value=8)

    manifest_path = artifact_root / 'incremental_modules.modules.json'
    published = manifest_path.read_bytes()
    source.write_text(source.read_text().replace(f'x: value + {2 + delta}',
                                               'x: missing_device_symbol', 1))
    build('reject_invalid_device', device_only=True, success=False)
    assert manifest_path.read_bytes() == published
    print('PASS: identical native code, helper invalidation, preserved host outputs, '
          'separate host rebuild and failure-safe publication', flush=True)
finally:
    source.write_text(original)
    build('final_restore')
    run()
    (args.output / 'measurements.json').write_text(json.dumps(records, indent=2))
