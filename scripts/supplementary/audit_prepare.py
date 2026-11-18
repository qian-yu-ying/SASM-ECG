"""Read-only import bridge for a historical CPSC helper; no main execution."""
import importlib.util
from pathlib import Path
_path=Path('/home/wy/ECGFounder/supplement_queue_20261101_remote/cpsc9_rerun_20261117/audit_prepare.py')
_spec=importlib.util.spec_from_file_location('_archived_cpsc_audit',_path)
_m=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
O,R,Q,B,dump,sha,NAMES=_m.O,_m.R,_m.Q,_m.B,_m.dump,_m.sha,_m.NAMES
