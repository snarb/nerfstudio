import copy
import tempfile
import unittest
from pathlib import Path
from agent import start_allowed
from common import atomic,read
from priority_scheduler import plan,recover,PRIORITY


def job(f,root='/fleet'):
 return dict(id=f'frame{f:06d}',command=['python','production.py','--root',root,'--frame',str(f)],completion=f'{root}/frames/{f:06d}/frame_complete.json',progress=f'{root}/frames/{f:06d}/train/progress.json',env={'CUDA_VISIBLE_DEVICES':'0'})

class SchedulingTests(unittest.TestCase):
 def test_gate_blocks_new_but_keeps_recovery(self):
  with tempfile.TemporaryDirectory() as d:
   j=dict(start_gate=str(Path(d)/'gate'));self.assertFalse(start_allowed(j));self.assertTrue(start_allowed(j,dict(status='failed')));atomic(j['start_gate'],{'complete':True});self.assertTrue(start_allowed(j))
 def test_steal_preserves_union_and_started(self):
  queues={'silver0':{'jobs':[job(908)]},'silver1':{'jobs':[job(f) for f in range(909,1269)]}}
  states={'silver0':{'jobs':{'frame000908':{'status':'complete'}},'active':None},'silver1':{'jobs':{'frame000913':{'status':'running'}},'active':'frame000913'}}
  out,moves=plan(queues,states,{n:Path('/fleet') for n in queues});self.assertEqual(len(moves),1);self.assertNotEqual(moves[0]['frame'],913);self.assertEqual(sum(len(q['jobs']) for q in out.values()),361)
  for q in out.values():
   for j in q['jobs']:self.assertEqual('start_gate' in j,int(j['id'][5:]) not in PRIORITY)
  self.assertEqual(len(queues['silver0']['jobs']),1)
 def test_source_half_constraint(self):
  queues={'ws07':{'jobs':[job(908)]},'silver1':{'jobs':[job(f) for f in range(909,1269)]}}
  states={'ws07':{'jobs':{'frame000908':{'status':'complete'}},'active':None},'silver1':{'jobs':{f'frame{f:06d}':{'status':'complete'} for f in range(909,1112)},'active':None}}
  out,moves=plan(queues,states,{n:Path('/fleet') for n in queues});self.assertEqual(moves,[])
 def test_recover_partial_transaction(self):
  class Fake:
   def __init__(self,p,q):self.agent_root=p;self.q=q;self.guards_cleared=0
   def get(self,p):return self.q
   def put(self,p,q):self.q=q
   def command(self,c):self.guards_cleared+=1
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);hosts={'a':Fake(root/'a',{'jobs':[1]}),'b':Fake(root/'b',{'jobs':[3]})}
   atomic(root/'priority_queue_transaction.json',dict(committed=False,changes={'a':dict(before={'jobs':[1]},after={'jobs':[2]}),'b':dict(before={'jobs':[2]},after={'jobs':[3]})}))
   recover(root,hosts);self.assertEqual(hosts['a'].q,{'jobs':[2]});self.assertTrue(read(root/'priority_queue_transaction.json')['committed']);self.assertEqual(hosts['a'].guards_cleared,1)

if __name__=='__main__':unittest.main()
