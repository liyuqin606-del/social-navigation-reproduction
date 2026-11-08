"""Launch the finite repair only after a separate, queue-bound user approval record."""
import argparse,json,hashlib,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True);a=p.parse_args()
 approval=json.loads(a.approval.read_text());queue=ROOT/'prep/QUEUE.json'
 if approval.get('approved') is not True or approval.get('scope')!='six_remaining_shards_including_replacement_of_28149' or approval.get('queue_sha256')!=sha(queue):
  raise RuntimeError('EXPLICIT_QUEUE_BOUND_USER_APPROVAL_REQUIRED')
 frozen=json.loads((ROOT/'prep/READY_INPUT_HASHES.json').read_text())
 for name,h in frozen['files'].items():
  if sha(name)!=h:raise RuntimeError('PREPARED_INPUT_CHANGED:'+name)
 runtime=json.loads((ROOT/'prep/RUNTIME_MANIFEST.json').read_text())
 for row in runtime['files']:
  f=ROOT/row['path']
  if 'symlink' in row and (not f.is_symlink() or str(f.readlink())!=row['symlink']):raise RuntimeError('RUNTIME_LINK_CHANGED')
 sys.path.insert(0,str(ROOT/'lifecycle'))
 from transport import Remote,read
 remote=Remote(read('/Users/yuqinli/.codex/skills/tacc-3090/references/profile.json'))
 x=remote.json("import subprocess,json;print(json.dumps({'queue':subprocess.check_output(['squeue','-h','-u','liyuqin','-o','%i|%j|%T|%b'],universal_newlines=True)}))",retry=True)
 if x['queue'].strip():raise RuntimeError('ACCOUNT_NOT_EMPTY_NO_NEW_CONTROLLER')
 directory=ROOT.parent/'runs/falcon_repair6_20261104_e'
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH='')
 cmd=[str(ROOT/'runtime/bin/python3.12'),'-B',str(ROOT/'scripts/continuous_reproduction_queue.py'),'--queue',str(queue),'--directory',str(directory)]
 # foreground ownership, no sbatch here; the controller uses the managed entry.
 return subprocess.call(cmd,env=env)
if __name__=='__main__':raise SystemExit(main())
