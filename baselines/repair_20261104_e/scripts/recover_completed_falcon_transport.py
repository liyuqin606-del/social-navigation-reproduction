#!/usr/bin/env python3
"""Recover transport-failed ownership without rerunning a successful remote shard.

No allocation or cancellation. Original owner/manager failures remain failed.
Only a normal, identity-verified remote completion plus the original independent
scientific validator can checkpoint a shard and clear transport-only ATTENTION.
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import recover_failed_falcon_owner as recovery

def load_controller(directory):
    state=json.loads((directory/'CHECKPOINT.json').read_text())
    for name,digest in state['input_files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'FROZEN_INPUT_CHANGED'
    paths=[Path(p) for p in state['input_files'] if Path(p).name=='continuous_reproduction_queue.py']
    assert len(paths)==1
    spec=importlib.util.spec_from_file_location('bound_recovery_controller',paths[0]);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    recovery.queue=q
    return q,state

def require_remote_success(observed):
    worker=observed.get('worker') or {}
    assert observed['scheduler_state']=='COMPLETED' and observed['scheduler_exit_code']=='0:0','REMOTE_NOT_NORMALLY_COMPLETED'
    assert observed.get('output_error') is None,'OUTPUT_COLLECTION_INVALID'
    assert worker.get('scientific_success') is True and worker.get('scientific_exit_code')==0 and worker.get('validation_exit_code')==0,'WORKER_SCIENCE_NOT_VERIFIED'

def main():
    p=argparse.ArgumentParser();p.add_argument('--queue-directory',type=Path,required=True);p.add_argument('--recovery-directory',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    directory=a.queue_directory.absolute();destination=a.recovery_directory.absolute();q,state=load_controller(directory)
    assert directory in destination.parents and destination!=directory
    with q.ControllerLock(directory):
        assert state['status']=='ATTENTION' and state.get('error') in ['OWNER_RELEASE_NOT_CONFIRMED','FAILED_RUN_LOG_NOT_VERIFIED'],'NOT_TRANSPORT_RECOVERY_ATTENTION'
        active=[(uid,u['attempts'][-1]) for uid,u in state['units'].items() if u['status']!='COMPLETE' and u['attempts']]
        assert len(active)==1;uid,attempt=active[0];plan=Path(attempt['plan']);driver=q.read(plan/'run/DRIVER_RECEIPT.json')
        assert driver.get('status')=='FAILED_RETAINED' and str(driver.get('error','')).startswith(('SSH_COMMAND_FAILED:', 'READ_TIMEOUT:')),'NOT_A_TRANSPORT_FAILED_OWNER'
        if a.check:
            if not driver.get('released'):recovery.admit(directory,plan)
            print(json.dumps({'status':'LOCAL_RECOVERY_ADMISSION_PASS_NO_REMOTE_CALL','job_id':driver['job_id']}));return
        if not driver.get('released'):
            ctx=recovery.admit(directory,plan)
            sys.path.insert(0,str(plan));from transport import Remote
            import preflight
            preflight.verify_keys(ctx['profile']);remote=Remote(ctx['profile'])
            # This gate refuses RUNNING/PENDING/other account jobs BEFORE recovery
            # directories or receipts are created. It never cancels an active job.
            recovery.query(remote,str(driver['job_id']),driver['session'],ctx['profile']['username'])
            recovery.recover(ctx,remote,destination)
        evidence=driver.get('recovery_evidence') or q.read(plan/'run/DRIVER_RECEIPT.json').get('recovery_evidence')
        assert evidence and q.sha(evidence['file'])==evidence['sha256'],'RECOVERY_EVIDENCE_MISSING'
        observed=q.inspect_attempt(attempt);require_remote_success(observed)
        controller=q.QueueController(state['contract'],directory);controller.verify_inputs();controller.verify_completions()
        unit=next(u for u in controller.cfg['units'] if u['unit_id']==uid);variant=unit['variants'][attempt['variant_index']]
        worker,proof=q.verify_outputs(plan,q.read(plan/'run/DRIVER_RECEIPT.json'),complete=True)
        ok,validator_proof=controller.validator(unit,attempt,variant['local_validator_argv'],'recovered_remote_complete')
        assert ok,'INDEPENDENT_SCIENTIFIC_VALIDATION_FAILED'
        proof.update(observed['proof_files']);proof.update(validator_proof);proof[evidence['file']]=evidence['sha256']
        controller.record_completion(variant['completion_ids'],unit,attempt,proof)
        controller.state['units'][uid]['attempts'][-1]['phase']='TERMINAL_RELEASED'
        assert set(unit['completion_ids'])<=set(controller.state['completed']),'PARTIAL_UNIT_NOT_SUPPORTED'
        controller.state['units'][uid]['status']='COMPLETE'
        controller.state.setdefault('resolved_transport_errors',[]).append({'error':controller.state.pop('error',None),'error_type':controller.state.pop('error_type',None),'recovery_receipt':evidence})
        controller.state['status']='READY';controller.save('TRANSPORT_FAILURE_RECOVERED_REMOTE_SCIENCE_INDEPENDENTLY_VERIFIED')
        print(json.dumps({'status':'REMOTE_COMPLETION_CHECKPOINTED_NO_NEW_SUBMISSION','completed':variant['completion_ids'],'original_owner_failure_preserved':True}))

if __name__=='__main__':main()
