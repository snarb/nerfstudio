import unittest
from pathlib import Path
import priority_scheduler
from dense_scheduler import plan,observe
from test_priority_scheduler import job


class DenseSchedulingTests(unittest.TestCase):
    def test_dense_endpoint_can_be_stolen_without_moving_active(self):
        queues={'silver0':{'jobs':[job(908)]},'silver1':{'jobs':[job(f) for f in range(909,1269)]}}
        states={'silver0':{'jobs':{'frame000908':{'status':'complete'}},'active':None},'silver1':{'jobs':{f'frame{f:06d}':{'status':'running' if f==1267 else 'complete'} for f in range(909,1268)},'active':'frame001267'}}
        out,moves=plan(queues,states,{n:Path('/fleet') for n in queues})
        self.assertEqual(moves,[dict(frame=1268,donor='silver1',receiver='silver0')])
        self.assertIn(job(1267),out['silver1']['jobs'])
        self.assertEqual(priority_scheduler.PRIORITY,set(range(908,1268,5)))
        self.assertEqual(sorted(int(j['id'][5:]) for q in out.values() for j in q['jobs']),list(range(908,1269)))

    def test_retired_workstation_is_not_a_receiver(self):
        queues={'ws07':{'jobs':[job(908)]},'silver0':{'jobs':[job(f) for f in range(909,1269)]}}
        states={'ws07':{'jobs':{'frame000908':{'status':'complete'}},'active':None},'silver0':{'jobs':{},'active':None}}
        _,moves=plan(queues,states,{n:Path('/fleet') for n in queues},unavailable={'ws07'})
        self.assertEqual(moves,[])

    def test_stopped_executor_is_retired_after_observation(self):
        class Fake:
            enabled=True;offline=False;row={'name':'ws07'};agent_root=Path('/source')
            def __init__(self):self.paths=[]
            def get(self,path):
                self.paths.append(path)
                return {'jobs':[]} if path.name=='queue.json' else {'jobs':{},'active':None}
            def command(self,command):return 'inactive\n'
            def activate(self):self.offline=True
        host=Fake();observe(host);self.assertTrue(host.offline)
        self.assertEqual(host.paths[:2],[Path('/source/queue.json'),Path('/source/agent_state.json')])


if __name__=='__main__':unittest.main()
