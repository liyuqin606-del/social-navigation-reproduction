#!/usr/bin/env python3
"""Prepare Falcon under one authorized root. No model execution or GPU allocation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback

ROOT = Path('/mnt/home/liyuqin/USERDIR/vln_baselines_20261103')
LOGS = ROOT / 'prep_logs'
STATE = LOGS / 'falcon_prepare_status.json'
ENV = ROOT / 'env/falcon'
PY = ENV / 'bin/python'
SRC = ROOT / 'src/Falcon'
RESERVE = 4 * 2**30
PEAK = 24 * 2**30
START = time.time()
LAST_OWNED_BYTES = None
INITIAL_FREE_BYTES = None


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def disk(measure=False):
    global LAST_OWNED_BYTES, INITIAL_FREE_BYTES
    free = shutil.disk_usage(ROOT).free
    if INITIAL_FREE_BYTES is None:
        INITIAL_FREE_BYTES = free
    # Avoid repeatedly walking tens of thousands of shared-filesystem package
    # files while pip installs. Check free space every 5 s; measure owned usage
    # between stages. The filesystem delta is a conservative in-stage cap.
    if LAST_OWNED_BYTES is not None and not measure:
        used = LAST_OWNED_BYTES + max(0, INITIAL_FREE_BYTES-free)
        if free < RESERVE or used > PEAK:
            raise RuntimeError('STORAGE_RESERVE_OR_CONSERVATIVE_PEAK_LIMIT')
        return {'filesystem_free_bytes':free,'conservative_owned_upper_bytes':used,
                'reserve_bytes':RESERVE,'account_quota_verified':False}
    paths = [ROOT / n for n in ['env', 'cache', 'prep_tmp', 'src']]
    existing = [str(x) for x in paths if x.exists()]
    if existing:
        measured = subprocess.run(['du', '-sk'] + existing, capture_output=True, text=True, timeout=300)
        # pip atomically renames/removes temporary files while we measure. A vanished
        # path is benign; permission/I/O errors still stop admission.
        if measured.returncode and any('No such file or directory' not in line
                                       for line in measured.stderr.splitlines() if line):
            raise RuntimeError('Storage measurement failed: ' + measured.stderr)
        rows = measured.stdout.splitlines()
        if len(rows) != len(existing):
            raise RuntimeError('Incomplete storage measurement')
        used = sum(int(x.split()[0]) for x in rows) * 1024
    else:
        used = 0
    used += sum(p.stat().st_size for p in (ROOT / 'inputs').glob('*.pth*'))
    LAST_OWNED_BYTES = used
    INITIAL_FREE_BYTES = free
    result = {'filesystem_free_bytes': free, 'owned_preparation_bytes': used,
              'reserve_bytes': RESERVE, 'preparation_peak_limit_bytes': PEAK,
              'account_quota_verified': False}
    if free < RESERVE or used > PEAK:
        raise RuntimeError('STORAGE_LIMIT: ' + json.dumps(result))
    return result


def status(stage, **extra):
    value = {'kind': 'ENVIRONMENT_PREPARATION_NO_GPU', 'stage': stage,
             'pid': os.getpid(), 'updated_unix': time.time(), 'elapsed_seconds': time.time()-START,
             'gpu_jobs_submitted': 0, 'model_forward_calls': 0, **extra}
    atomic(STATE, value)
    print(json.dumps(value), flush=True)


def run(name, argv, cwd=None, timeout=1800):
    before = disk(measure=True)
    logfile = LOGS / (name + '.log')
    status(name, argv=argv, log=str(logfile), storage=before)
    with logfile.open('wb') as log:
        p = subprocess.Popen(argv, cwd=cwd, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        began = time.time()
        while p.poll() is None:
            if time.time()-began > timeout:
                p.terminate(); p.wait(timeout=20)
                raise RuntimeError(name + ': timeout')
            try:
                disk()
            except Exception:
                p.terminate(); p.wait(timeout=20)
                raise
            time.sleep(5)
        if p.returncode:
            raise RuntimeError(name + ': exit ' + str(p.returncode) + '; see ' + str(logfile))
    disk(measure=True)


def main():
    for name in ['cache/pip', 'cache/xdg', 'prep_tmp', 'prep_scripts', 'inputs', 'src', 'prep_logs']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    os.environ.update({'PIP_CACHE_DIR': str(ROOT/'cache/pip'), 'XDG_CACHE_HOME': str(ROOT/'cache/xdg'),
                       'TMPDIR': str(ROOT/'prep_tmp'), 'CUDA_VISIBLE_DEVICES': '',
                       'PYTHONNOUSERSITE': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
                       'PIP_DEFAULT_TIMEOUT': '90', 'PIP_RETRIES': '5',
                       'PIP_INDEX_URL': 'https://pypi.tuna.tsinghua.edu.cn/simple',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    status('WAIT_EXISTING_CONDA', storage=disk())
    prior = json.loads((LOGS/'conda_process.json').read_text())
    deadline = time.time()+3600
    while Path('/proc/'+str(prior['pid'])).exists():
        try:
            cmd = Path('/proc/'+str(prior['pid'])+'/cmdline').read_bytes()
            if str(ENV).encode() not in cmd or b'conda' not in cmd:
                break
        except FileNotFoundError:
            break
        if time.time()>deadline:
            raise RuntimeError('Existing conda create not finished within 3600 seconds; not cancelled')
        disk(); time.sleep(5)
    if not PY.is_file():
        raise RuntimeError('Existing conda create ended without environment Python; inspect conda_create.log')

    archive = ROOT/'inputs/falcon_source.tar.gz'
    expected = '7a607cb5db56cea30f7d351ce0daff76142c3156f8ddc920767615977754d4f0'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
    SRC.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            path = Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts and member.isfile()
        tf.extractall(SRC)
    source = json.loads((SRC/'SOURCE_MANIFEST.json').read_text())
    for entry in source['files']:
        f = SRC/entry['path']
        assert f.stat().st_size==entry['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256']
    atomic(LOGS/'source_verification.json', {'verified': True, 'files': len(source['files']),
           'commit': source['commit'], 'archive_sha256': expected, 'storage': disk()})

    constraints = ROOT/'prep_scripts/constraints.txt'
    constraints.write_text('''numpy==1.23.5
torch==2.1.2+cu118
torchvision==0.16.2+cu118
gym==0.23.0
scipy==1.10.1
numba==0.57.1
numpy-quaternion==2022.4.3
opencv-python==4.8.1.78
matplotlib==3.7.5
hydra-core==1.3.2
omegaconf==2.3.0
protobuf==3.20.1
tensorboard==2.8.0
moviepy==1.0.3
imageio==2.31.6
imageio-ffmpeg==0.4.9
''')
    run('pip_bootstrap', [str(PY), '-m', 'pip', 'install', 'pip==23.3.2', 'setuptools==65.5.0', 'wheel==0.41.3'])
    run('pip_torch', [str(PY), '-m', 'pip', 'install', 'torch==2.1.2', 'torchvision==0.16.2',
                     '--index-url', 'https://download.pytorch.org/whl/cu118'])
    run('pip_dependencies', [str(PY), '-m', 'pip', 'install', '-c', str(constraints),
                            '-e', str(SRC/'habitat-lab'), '-e', str(SRC/'habitat-baselines'),
                            '-r', str(SRC/'requirements.txt'), 'gdown==5.2.0'], cwd=str(SRC))
    run('pip_check', [str(PY), '-m', 'pip', 'check'])
    run('pip_freeze', [str(PY), '-m', 'pip', 'freeze', '--all'])
    run('conda_list', ['/mnt/home/liyuqin/.Miniconda3/bin/conda', 'list', '-p', str(ENV), '--json'])
    run('cpu_imports', [str(PY), '-c',
        "import json,torch,torchvision,numpy,habitat_sim,habitat,habitat_baselines,falcon; "
        "print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,"
        "'numpy':numpy.__version__,'torch_cuda_build':torch.version.cuda,'scope':'imports only, no GPU query or model forward'}))"], cwd=str(SRC))

    weight_code = '''import gdown, pathlib,hashlib,json,shutil,zipfile
root=pathlib.Path('/mnt/home/liyuqin/USERDIR/vln_baselines_20261103')
items=[('falcon_noaux_25.pth','1t9DGtwjAdPu7yDPBC8sHTGdXAnbai4tM',50048730),('pretrained_habitat3.pth','1QakOj1cUVPoEmTe5_aNwyMyDsBYODAyX',49961178)]
result=[]
for name,ident,size in items:
 p=root/'inputs'/name
 if not p.exists() or p.stat().st_size!=size:
  assert shutil.disk_usage(root).free>4*2**30+size
  partial=p.with_suffix(p.suffix+'.partial')
  out=gdown.download(id=ident,output=str(partial),quiet=False,use_cookies=False)
  assert out and partial.stat().st_size==size,(out,partial.stat().st_size if partial.exists() else None)
  partial.replace(p)
 assert zipfile.is_zipfile(p), 'Expected torch zip container'
 result.append({'file':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_drive_id':ident,'metadata_only':True,'pickle_executed':False})
target=root/'src/Falcon/pretrained_model';target.mkdir(exist_ok=True)
for name,_,_ in items:
 link=target/name
 if not link.exists():link.symlink_to(root/'inputs'/name)
(root/'prep_logs/weight_manifest.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result,indent=2))
'''
    run('download_weights', [str(PY), '-c', weight_code], timeout=1200)
    status('PREPARATION_COMPLETE', storage=disk(), ready_for_cpu_imports=True,
           ready_for_navigation_evaluation=False,
           remaining=['Scene assets, episode files and robot/humanoid assets must be verified separately',
                      'CUDA/renderer and model forward must be checked inside an authorized Slurm job'])


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        status('FAILED', error=str(exc))
        raise
