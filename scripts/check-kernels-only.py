#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU regression for metadata-free kernel edits against an existing binary."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cargo-oxide',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/'crates/rustc-codegen-cuda/examples/incremental_modules'

source=root/'src/lib.rs';original=source.read_text();args.output.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CUDA_OXIDE_MODULE_FILES_ONLY='1',CUDA_OXIDE_REUSE_HOST_FOR_KERNEL_EDITS='1',CARGO_PROFILE_RELEASE_DEBUG='0')
env.pop('CUDA_OXIDE_INCREMENTAL_MODULES', None)
env.pop('CUDA_OXIDE_PTX_DIR', None)
metadata=json.loads(subprocess.check_output(['cargo','metadata','--format-version=1','--no-deps','--manifest-path',str(root/'Cargo.toml')],env=env,cwd=root))
artifact_root=Path(metadata['target_directory'])/'release/oxide'

records=[]
def build(label,only=False,success=True):
    command=[args.cargo_oxide]
    if only:command+=['--kernels-only','incremental_modules']
    command+=['build','--emit-nvvm-ir','--','--release','--manifest-path',str(root/'Cargo.toml')]
    if only:command+=['--lib']
    log=args.output/(label+'.log');start=time.perf_counter()
    with log.open('w') as stream:code=subprocess.call(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
    record=dict(label=label,seconds=time.perf_counter()-start,exit_code=code);records.append(record);print(json.dumps(record),flush=True)
    if (code==0)!=success:raise RuntimeError(log.read_text()[-8000:])
    return log.read_text()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(delta=0):
    subprocess.run([str(binary),'--files',f'--first-delta={delta}'],env=env,check=True)
try:
    build('baseline')
    meta=json.loads(subprocess.check_output(['cargo','metadata','--format-version=1','--no-deps','--manifest-path',str(root/'Cargo.toml')],cwd=root,env=env))
    target=Path(meta['target_directory']);binary=target/'release/incremental_modules'
    libraries=list((target/'release/build/incremental_modules').glob('*/out/libincremental_modules-*.rlib'))
    library=max(libraries,key=lambda p:p.stat().st_mtime)
    artifacts={p:digest(p) for p in [library,library.with_suffix('.rmeta'),binary]}
    run()
    delta=100+time.time_ns()%1000000
    for increment in range(2):
        source.write_text(original.replace('kernel_bias = 10u32',f'kernel_bias = {10+delta+increment}u32',1))
        build(f'kernel_edit_{increment}',only=True)
        assert {p:digest(p) for p in artifacts}==artifacts
        run(delta+increment)
    manifest=artifact_root/'incremental_modules.modules.json';before=digest(manifest)
    source.write_text(source.read_text().replace('fn host_value() -> u32 {\n    7\n}', 'fn host_value() -> u32 {\n    8\n}'))
    log=build('reject_host_change',only=True,success=False)
    assert 'cannot prove unchanged host MIR/ABI' in log
    assert digest(manifest)==before and {p:digest(p) for p in artifacts}==artifacts
    source.write_text(original.replace('let kernel_bias =', '\n        let kernel_bias =',1))
    build('reject_host_location_change',only=True,success=False)
    assert digest(manifest)==before and {p:digest(p) for p in artifacts}==artifacts
    # Local type/ABI and helper changes require refreshing the host too.
    for label,changed in [
        ('reject_cpu_closure_change',original.replace('let thunk = || 7u32','let thunk = || 8u32',1)),
        ('reject_helper_change',original.replace('x: value + 2','x: value + 3',1)),
        ('reject_abi_change',original.replace('#[derive(Clone, Copy)]','#[derive(Clone, Copy)] #[repr(C, align(16))]',1)),
    ]:
        assert changed!=original
        source.write_text(changed)
        log=build(label,only=True,success=False)
        assert 'cannot prove unchanged host MIR/ABI' in log
        assert digest(manifest)==before and {p:digest(p) for p in artifacts}==artifacts
    # A damaged proof cache never approves use of an unknown CPU image.
    cache=artifact_root/'cache/host/v2/incremental_modules'
    latest=json.loads((cache/'latest-contract.json').read_text())
    record=json.loads((cache/(latest['key']+'.json')).read_text())
    module=next(m for m in record['modules'] if m['object'])
    proof=cache/'objects'/module['object']['file'];saved=proof.read_bytes()
    try:
        proof.write_bytes(saved[:-1]+bytes([saved[-1]^1]))
        source.write_text(original.replace('kernel_bias = 10u32',f'kernel_bias = {11+delta}u32',1))
        build('reject_corrupted_host_cache',only=True,success=False)
        assert digest(manifest)==before and {p:digest(p) for p in artifacts}==artifacts
    finally:proof.write_bytes(saved)
    print('PASS: successive native updates execute in the unchanged binary; CPU, location, helper, ABI and corrupted-proof edits rejected before publication',flush=True)
finally:
    source.write_text(original);build('final_restore')
    if 'binary' in globals():run()
    (args.output/'measurements.json').write_text(json.dumps(records,indent=2))
