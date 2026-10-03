#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that GPU-only kernel edits reuse host objects and CPU edits cannot."""
import argparse,json,os,subprocess,time
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cargo-oxide',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/'crates/rustc-codegen-cuda/examples/incremental_modules'

source=root/'src/lib.rs';original=source.read_text();args.output.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CUDA_OXIDE_MODULE_FILES_ONLY='1',CUDA_OXIDE_REUSE_HOST_FOR_KERNEL_EDITS='1')
env.pop('CUDA_OXIDE_INCREMENTAL_MODULES', None)
env.pop('CUDA_OXIDE_PTX_DIR', None)
metadata=json.loads(subprocess.check_output(['cargo','metadata','--format-version=1','--no-deps','--manifest-path',str(root/'Cargo.toml')],env=env,cwd=root))
artifact_root=Path(metadata['target_directory'])/'release/oxide'

records=[]
def build(label):
    start=time.perf_counter();log=args.output/(label+'.log')
    with log.open('w') as stream:
        code=subprocess.call([args.cargo_oxide,'build','--emit-nvvm-ir','--','--release','--manifest-path',str(root/'Cargo.toml')],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
    record=dict(label=label,seconds=time.perf_counter()-start,exit_code=code,host_reused='host objects reused:' in log.read_text())
    records.append(record);print(json.dumps(record),flush=True)
    if code:raise RuntimeError(log.read_text()[-6000:])
    return record
def run(*flags):
    meta=json.loads(subprocess.check_output(['cargo','metadata','--format-version=1','--no-deps','--manifest-path',str(root/'Cargo.toml')],cwd=root,env=env))
    result=subprocess.run([str(Path(meta['target_directory'])/'release/incremental_modules'),'--files',*flags],env=env,check=True,capture_output=True,text=True)
    print(result.stdout,flush=True)
    line=source.read_text().splitlines().index('    caller_line()')+1
    assert f'host_caller_line={line}\n' in result.stdout
try:
    build('seed');run()
    delta=100+time.time_ns()%1000000
    source.write_text(original.replace('kernel_bias = 10u32',f'kernel_bias = {10+delta}u32',1))
    record=build('kernel_only_edit');assert record['host_reused'];run(f'--first-delta={delta}')
    source.write_text(source.read_text().replace(f'kernel_bias = {10+delta}u32',f'kernel_bias = {11+delta}u32',1))
    record=build('successive_kernel_edit');assert record['host_reused'];run(f'--first-delta={delta+1}')
    source.write_text(source.read_text().replace(f'kernel_bias = {11+delta}u32',f'kernel_bias = {10+delta}u32',1))
    source.write_text(source.read_text().replace('fn host_value() -> u32 {\n    7\n}', f'fn host_value() -> u32 {{\n    {8+delta}\n}}'))
    assert source.read_text()!=original.replace('kernel_bias = 10u32',f'kernel_bias = {10+delta}u32',1)
    record=build('host_function_edit');assert not record['host_reused'];run(f'--first-delta={delta}',f'--host-value={8+delta}')
    source.write_text(source.read_text().replace('let kernel_bias =', '\n        let kernel_bias =',1))
    record=build('host_caller_location_edit');assert not record['host_reused'];run(f'--first-delta={delta}',f'--host-value={8+delta}')
    # Cache contents cannot silently replace a changed CPU implementation.
    cache=artifact_root/'cache/host/v2/incremental_modules'
    entries=[p for p in cache.glob('*.json') if p.name!='latest-contract.json'];assert entries
    entry=json.loads(max(entries,key=lambda p:p.stat().st_mtime).read_text())
    module=next(m for m in entry['modules'] if m['object'])
    artifact=cache/'objects'/module['object']['file'];saved=artifact.read_bytes()
    artifact.write_bytes(saved[:-1]+bytes([saved[-1]^1]))
    source.write_text(source.read_text().replace(f'kernel_bias = {10+delta}u32',f'kernel_bias = {11+delta}u32',1))
    record=build('corrupted_host_cache');assert not record['host_reused'];run(f'--first-delta={delta+1}',f'--host-value={8+delta}')
finally:
    source.write_text(original);build('final_restore')
    (args.output/'measurements.json').write_text(json.dumps(records,indent=2))
