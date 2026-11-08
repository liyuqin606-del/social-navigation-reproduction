"""Offline tests. Fake SSH only; no Slurm/GPU operations."""
import json,sys,tempfile,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'lifecycle'))
import owner

class RepairTests(unittest.TestCase):
 def test_observation_errors_tagged_only(self):
  class Remote:
   def json(self,*a,**kw):raise RuntimeError('SSH_COMMAND_FAILED: DNS unavailable')
  with self.assertRaises(owner.ObservationUnavailable):owner.observed_json(Remote(),'READ')
  class Invalid:
   def json(self,*a,**kw):raise RuntimeError('IDENTITY_MISMATCH')
  with self.assertRaisesRegex(RuntimeError,'IDENTITY_MISMATCH'):owner.observed_json(Invalid(),'READ')
 def test_timeout_tagged(self):
  class Remote:
   def json(self,*a,**kw):raise subprocess.TimeoutExpired('ssh',1)
  with self.assertRaises(owner.ObservationUnavailable):owner.observed_json(Remote(),'READ')
 def _exercise_owner(self, uncertain_submit=False):
  calls=[]
  class Remote:
   def __init__(self,*a):pass
   def json(self,code,**kw):
    if 'disk_usage' in code:return {'free_bytes':100<<30,'required_bytes':5<<30}
    if "'account':call(" in code:
     assert 'now-1day' not in code
     return {'queue':'12345|liyuqin|offline-test-monitor','account':'12345|liyuqin|offline-test-monitor'}
    raise RuntimeError('SSH_COMMAND_FAILED: simulated transport lost')
   def upload(self,*a):pass
   def code(self,code,**kw):
    calls.append(code)
    if "'sbatch'" in code:
     if uncertain_submit:raise RuntimeError("SSH_COMMAND_FAILED: lost submit response")
     return "12345"
    return ''
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);spec={'run_id':'offline-test-monitor','remote_root':'/mnt/home/liyuqin/USERDIR/offline-test','remote_python':'/unused/python','slurm':{'partition':'tacc','gpus':1,'cpus':2,'wall_time':'00:05:00','deadline':'2099-01-01T00:00:00'},'science':{'argv':['python','-c','pass'],'cwd':'{source}','timeout_seconds':120,'validator_argv':['python','-c','pass'],'required_outputs':['result.json']},'output_budget_bytes':1<<20,'remote_peak_additional_bytes':2<<20,'monitor_timeout_seconds':600}
   (t/'spec.json').write_text(json.dumps(spec))
   subprocess.run([sys.executable,str(ROOT/'lifecycle/prepare_run.py'),'--spec',str(t/'spec.json'),'--directory',str(t/'plan')],check=True,capture_output=True)
   with patch.object(owner,'Remote',Remote),patch.object(owner.preflight,'verify_keys'),patch.object(sys,'argv',['owner','--config',str(t/'plan/CONTROL.json'),'--freeze',str(t/'plan/CONTROL_FREEZE.json')]):
    rc=owner.main()
   d=json.loads((t/'plan/run/DRIVER_RECEIPT.json').read_text())
   self.assertEqual(rc,5);self.assertEqual(d['job_id'],'12345');self.assertFalse(d['released']);self.assertTrue(d['monitoring_interrupted']);self.assertTrue(d['recovery_required'])
   self.assertFalse(any('scancel' in x for x in calls));self.assertEqual(sum("'sbatch'" in x for x in calls),1)
 def test_owner_does_not_cancel_on_lost_observation(self):self._exercise_owner()
 def test_uncertain_submit_queries_original_job_without_resubmit(self):self._exercise_owner(True)
 def test_approval_missing_stops_before_network(self):
  p=subprocess.run([sys.executable,str(ROOT/'scripts/launch_after_approval.py'),'--approval','/nonexistent/approval.json'],capture_output=True,text=True)
  self.assertNotEqual(p.returncode,0);self.assertIn('FileNotFoundError',p.stderr)

if __name__=='__main__':unittest.main()
