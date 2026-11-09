"""One owned pre-eval checkpoint + SHA-bound metric journal; no optimizer/RNG work."""
from pathlib import Path
import json,hashlib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def _binding(state,latest):return dict(step=latest['step'],spec_sha256=state['spec_sha256'],data_sha256=state['data_sha256'],code_sha256=state['code_sha256'],checkpoint=latest['checkpoint'],checkpoint_sha256=latest['checkpoint_sha256'])
def pending_evaluation(output,state,latest,should_evaluate,write):
 """Merge a validated completed journal, otherwise recover pending evalbeforetraining."""
 output=Path(output);path=output/'pending_evaluation.json';expected=_binding(state,latest);step=latest['step'];done=[x for x in state['evaluations']if x['step']==step]
 if len(done)>1:raise ValueError('Duplicate own evaluation gate')
 journal=output/'evaluation_commits'/f'step-{step:09d}.json'
 if journal.exists():
  j=read(journal)
  if j.get('kind')!='NATIVE63_CHECKPOINT_BOUND_EVALUATION_V1'or any(j.get(k)!=v for k,v in expected.items())or j['result']['step']!=step:raise ValueError('Evaluation journal source/checkpoint mismatch')
  if done and done[0]!=j['result']:raise ValueError('Conflicting own evaluation result')
  if not done:state['evaluations'].append(j['result']);write(output/'metrics.json',state['evaluations'])
  if path.exists():
   if any(read(path).get(k)!=v for k,v in expected.items()):raise ValueError('Pending marker conflicts with completedjournal')
   path.unlink()
  return None
 if path.exists():
  marker=read(path)
  if any(marker.get(k)!=v for k,v in expected.items()):raise ValueError('Pendingevaluation source/checkpoint mismatch')
  if done:path.unlink();return None
  return marker
 if should_evaluate and not done:
  # Covers a crash after canonical checkpoint commit butbefore pendingmarker.
  write(path,expected);return expected
 return None

def evaluate_and_commit(output,state,step,evaluate,save,write,terminal=False,pending=None):
 """Never overwrite the canonical checkpoint after evaluation: commit metrics separately."""
 output=Path(output);path=output/'pending_evaluation.json'
 if any(r['step']==step for r in state['evaluations']):raise ValueError('Cannot appendduplicate own evaluation')
 if pending is None:
  save(step,terminal=terminal);latest=read(output/'latest.json');marker=_binding(state,latest)
  if latest['step']!=step:raise ValueError('Pre-eval checkpoint step differs')
  write(path,marker)
 else:
  latest=read(output/'latest.json');marker=_binding(state,latest)
  if pending!=marker or step!=latest['step']:raise ValueError('Pendingcheckpoint changed beforeevaluation')
 (output/'evaluation_commits').mkdir(parents=True,exist_ok=True)
 result=evaluate(step)
 if result['step']!=step:raise ValueError('Evaluation resultstep differs')
 journal=dict(kind='NATIVE63_CHECKPOINT_BOUND_EVALUATION_V1',**marker,result=result)
 write(output/'evaluation_commits'/f'step-{step:09d}.json',journal)
 state['evaluations'].append(result);write(output/'metrics.json',state['evaluations']);path.unlink()
 return result
