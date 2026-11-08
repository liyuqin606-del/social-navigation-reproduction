#!/usr/bin/env python3
"""Read-only, stdlib-only TACC admission probe. Never allocates a GPU."""
import argparse
import datetime
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
MIB = 1 << 20
GIB = 1 << 30


class Blocked(RuntimeError):
    pass


def expand(value):
    return str(Path(value).expanduser().absolute())


def run(argv, input_text=None, timeout=15):
    return subprocess.run(argv, input=input_text, capture_output=True,
                          text=True, timeout=timeout)


def budget(free_bytes, output_mib, temporary_mib, reserve_gib):
    values = (output_mib, temporary_mib, reserve_gib)
    if any(not math.isfinite(x) or x < 0 for x in values) or reserve_gib < 4:
        raise Blocked('INVALID_BUDGET_OR_RESERVE_BELOW_4_GIB')
    required = math.ceil((output_mib + temporary_mib) * MIB + reserve_gib * GIB)
    return {'free_bytes': free_bytes, 'required_bytes': required,
            'reserve_bytes': math.ceil(reserve_gib * GIB),
            'admission_ok': free_bytes >= required}


def fingerprints(text, kind=None):
    # ssh-keygen's second field is authoritative; comments can contain SHA text.
    result = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and re.fullmatch(r'SHA256:[A-Za-z0-9+/]+', fields[1]):
            if kind is None or fields[-1] == '(' + kind + ')':
                result.add(fields[1])
    return result


def verify_keys(profile, call=run):
    identity = Path(expand(profile['identity_file']))
    hosts = Path(expand(profile['known_hosts_file']))
    if not identity.is_file():
        raise Blocked('ORIGINAL_IDENTITY_FILE_MISSING')
    if not hosts.is_file():
        raise Blocked('APPROVED_KNOWN_HOST_RECORD_MISSING')
    # Derive public material only; neither private bytes nor public text is logged.
    public = call(['/usr/bin/ssh-keygen', '-y', '-f', str(identity)], input_text='')
    if public.returncode:
        raise Blocked('ORIGINAL_IDENTITY_NOT_READABLE_NONINTERACTIVELY')
    digest = call(['/usr/bin/ssh-keygen', '-l', '-f', '/dev/stdin'], input_text=public.stdout)
    if digest.returncode or profile['public_fingerprint'] not in fingerprints(digest.stdout):
        raise Blocked('ORIGINAL_PUBLIC_IDENTITY_DIFFERS')
    name = '[{}]:{}'.format(profile['hostname'], profile['port'])
    record = call(['/usr/bin/ssh-keygen', '-F', name, '-f', str(hosts)])
    if record.returncode:
        raise Blocked('APPROVED_KNOWN_HOST_RECORD_MISSING')
    digest = call(['/usr/bin/ssh-keygen', '-l', '-f', '/dev/stdin'], input_text=record.stdout)
    if digest.returncode or fingerprints(digest.stdout, 'ED25519') != {profile['approved_host_fingerprint']}:
        raise Blocked('APPROVED_HOST_FINGERPRINT_DIFFERS')
    return {'original_public_identity_matches': True, 'approved_host_fingerprint_matches': True}


def ssh_argv(profile):
    return ['/usr/bin/ssh', '-F', '/dev/null', '-o', 'BatchMode=yes',
            '-o', 'IdentitiesOnly=yes', '-o', 'IdentityAgent=none',
            '-o', 'ForwardAgent=no', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'HostKeyAlgorithms=ssh-ed25519',
            '-o', 'UserKnownHostsFile=' + expand(profile['known_hosts_file']),
            '-o', 'ConnectTimeout=12', '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=2', '-p', str(profile['port']),
            '-i', expand(profile['identity_file']),
            profile['username'] + '@' + profile['hostname']]


def remote_code(profile):
    # Python 3.6 compatible. Slurm help/query commands only; no CUDA or job action.
    code = '''import json,os,shutil,subprocess,sys,time
p = PROFILE_LITERAL
def query(argv):
 try:
  r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12)
  return r.returncode,r.stdout
 except (OSError,subprocess.TimeoutExpired):return 127,''
home=p['remote_home']
free=shutil.disk_usage(home).free
si,sinfo=query(['sinfo','-h','-o','%P|%a|%l|%D|%t'])
sq,queue=query(['squeue','-h','-u',p['username'],'-o','%i|%j|%T|%R'])
bh,help_text=query(['sbatch','--help'])
conda_exists=os.path.isfile(p['remote_conda'])
cc,conda_version=query([p['remote_conda'],'--version']) if conda_exists else (127,'')
py_exists=os.path.isfile(p['remote_python'])
pc,python_version=query([p['remote_python'],'--version']) if py_exists else (127,'')
print(json.dumps({'filesystem_free_bytes':free,'partition_query_ok':si==0,'queue_query_ok':sq==0,
 'partitions':sinfo.splitlines(),'own_jobs':queue.splitlines(),'sbatch_available':bh==0,
 'server_deadline_supported':bh==0 and '--deadline' in help_text,
 'conda_ready':cc==0,'conda_version':conda_version.strip(),'python_ready':pc==0,
 'python_version':python_version.strip(),'system_python_version':sys.version.split()[0],
 'server_time':time.strftime('%Y-%m-%d %H:%M:%S %Z'),'account_quota_verified':False,
 'gpu_allocations':0}))
'''
    allowed = {k: profile[k] for k in ('username', 'remote_home', 'remote_conda', 'remote_python')}
    return code.replace('PROFILE_LITERAL', repr(allowed))


def client_state(profile, call=run):
    binary = Path(expand(profile['tcloud_binary']))
    cfg_path = Path(expand(profile['tcloud_config']))
    cluster = Path(expand(profile['cluster_config']))
    try:
        cfg = {k.lower(): v for k, v in json.loads(cfg_path.read_text()).items()}
    except (OSError, ValueError, AttributeError):
        cfg = {}
    try:
        cluster_valid = bool(json.loads(cluster.read_text()))
    except (OSError, ValueError):
        cluster_valid = False
    version = ''
    if binary.is_file():
        try:
            result = call([str(binary), '--help'])
            if result.returncode == 0 and '[Tcloud Error]' not in result.stdout + result.stderr:
                version = next(iter(result.stdout.splitlines()), '')
        except (OSError, subprocess.TimeoutExpired):
            pass
    auth = cfg.get('authfile')
    return {'binary_available': bool(version), 'version': version,
            'username_matches': cfg.get('username') == profile['username'],
            'identity_path_matches': isinstance(auth, str) and expand(auth) == expand(profile['identity_file']),
            'cluster_config_nonempty_valid_json': cluster_valid}


def inspect(project, output_mib, temporary_mib, remote_peak_mib, reserve_gib,
            profile, call=run, usage=shutil.disk_usage):
    report = {'scope': 'READ_ONLY_TACC_PREFLIGHT',
              'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'ready_for_native_slurm': False, 'ready_for_tcloud': False,
              'gpu_jobs_submitted': 0, 'account_configuration_modified': False,
              'project_dependencies_verified': False, 'blocker': None}
    try:
        project = Path(project).expanduser().resolve()
        if not project.is_dir():
            raise Blocked('PROJECT_DIRECTORY_MISSING')
        if not math.isfinite(remote_peak_mib) or remote_peak_mib < 0:
            raise Blocked('INVALID_REMOTE_STORAGE_BUDGET')
        report['local_storage'] = budget(usage(project).free, output_mib, temporary_mib, reserve_gib)
        if not report['local_storage']['admission_ok']:
            raise Blocked('LOCAL_STORAGE_ADMISSION_FAILED')
        report['identity'] = verify_keys(profile, call)
        result = call(ssh_argv(profile) + ['python3 -c ' + shlex.quote(remote_code(profile))], timeout=75)
        if result.returncode:
            if 'Permission denied (publickey)' in result.stderr:
                raise Blocked('ORIGINAL_KEY_AUTHENTICATION_DENIED')
            if 'Host key verification failed' in result.stderr or 'HOST IDENTIFICATION HAS CHANGED' in result.stderr:
                raise Blocked('HOST_IDENTITY_REJECTED')
            raise Blocked('NETWORK_OR_SSH_SESSION_UNAVAILABLE')
        try:
            remote = json.loads(result.stdout)
        except ValueError:
            raise Blocked('REMOTE_RESPONSE_NOT_VALID_JSON')
        report['remote'] = remote
        report['remote_storage'] = budget(remote['filesystem_free_bytes'], remote_peak_mib, 0, reserve_gib)
        if not report['remote_storage']['admission_ok']:
            raise Blocked('REMOTE_STORAGE_ADMISSION_FAILED')
        if not all(remote.get(k) for k in ('partition_query_ok', 'queue_query_ok', 'sbatch_available')):
            raise Blocked('SLURM_READ_ONLY_QUERY_FAILED')
        report['client'] = client_state(profile, call)
        report['ready_for_native_slurm'] = True
        report['ready_for_tcloud'] = (all(report['client'][k] for k in
            ('binary_available', 'username_matches', 'identity_path_matches', 'cluster_config_nonempty_valid_json'))
            and remote.get('conda_ready') is True and remote.get('python_ready') is True)
    except Blocked as error:
        report['blocker'] = str(error)
    except subprocess.TimeoutExpired:
        report['blocker'] = 'READ_ONLY_PROBE_TIMED_OUT_STATE_UNKNOWN'
    except OSError:
        report['blocker'] = 'LOCAL_TOOL_OR_FILESYSTEM_UNAVAILABLE'
    except (KeyError, TypeError):
        report['blocker'] = 'PROFILE_OR_REMOTE_RESPONSE_SCHEMA_INVALID'
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--local-output-mib', type=float, required=True)
    ap.add_argument('--local-temp-mib', type=float, required=True)
    ap.add_argument('--remote-peak-mib', type=float, required=True)
    ap.add_argument('--reserve-gib', type=float, default=4)
    args = ap.parse_args()
    profile = json.loads((ROOT / 'references/profile.json').read_text())
    report = inspect(args.project, args.local_output_mib, args.local_temp_mib,
                     args.remote_peak_mib, args.reserve_gib, profile)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['ready_for_native_slurm'] else 2


if __name__ == '__main__':
    sys.exit(main())
