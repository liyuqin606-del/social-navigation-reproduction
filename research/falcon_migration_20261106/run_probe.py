import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args,remaining=p.parse_known_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for mode in ['reference','shadow','override']:
        command=[sys.executable,str(Path(__file__).with_name('run_case.py')),*remaining,'--mode',mode,'--output',str(args.output/mode)]
        print('START',mode,flush=True)
        with (args.output/(mode+'.log')).open('x') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=260)
        if proc.returncode:
            raise RuntimeError('CASE_FAILED_'+mode+': '+str(proc.returncode))
        print('END',mode,flush=True)
    (args.output/'RUN_COMPLETE.json').write_text(json.dumps({'cases':3,'maximum_total_environment_steps':48,'benchmark':False})+'\n')
    subprocess.run([sys.executable,str(Path(__file__).with_name('validate_probe.py')),
                    '--output',str(args.output)],check=True,timeout=120)


if __name__=='__main__':
    main()
