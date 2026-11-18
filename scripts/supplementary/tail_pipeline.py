from common import *
import subprocess
def main():
 pid=int((E/'supplements.pid').read_text());start=time.monotonic();emit('FOLLOWUP_WAIT',waiting_for_pid=pid)
 while not (E/'SWAP_COMPLETE.json').exists():
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('Calibration/role-swap process stopped before completion; inspect log. No automatic relaunch.')
  if time.monotonic()-start>7200:raise TimeoutError('Wait budget exceeded; no automatic relaunch.')
  time.sleep(20)
 for script in ['timing.py','subgroups.py']:
  with (E/(script+'.log')).open('w') as log:
   subprocess.run([sys.executable,'-u',str(E/script)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=E)
 dump(E/'FOLLOWUP_COMPLETE.json',dict(time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(E/'FOLLOWUP_ERROR.json',dict(traceback=traceback.format_exc()));raise
