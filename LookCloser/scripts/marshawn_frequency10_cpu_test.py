"""CPU ownership and field-priority regression for unified all66 preprocessing."""
from pathlib import Path
import hashlib, json, tempfile
import marshawn_frequency10 as frequency


def run():
    with tempfile.TemporaryDirectory() as directory:
        V=Path(directory);ops=V/'operations';ops.mkdir()
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        class IO:
            @staticmethod
            def read(path):return json.loads(Path(path).read_text())
            @staticmethod
            def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        plan=V/'plan.json';write(plan,{'hard_max_iterations':140000});state=V/'state.json'
        write(state,dict(frames={'1660':dict(name='f0',host='silver',gpu=0),'1675':dict(name='f1',host=None,gpu=None)},jobs={'f0':dict(status='running'),'f1':dict(status='pending')}))
        assignments=ops/'assignments';demands=ops/'demands';assignments.mkdir();demands.mkdir()
        write(ops/'fleet10_active.json',dict(hard_max_iterations=140000,plan=str(plan),plan_sha256=IO.sha(plan),state=str(state),assignments=str(assignments),demands=str(demands)))
        write(assignments/'001675.json',dict(frame=1675,host='olive',gpu=0,status='reserved'))
        write(demands/'001680.json',dict(ready_for_training=True,status='awaiting_idle_training_slot',preferred_host='dev3',preferred_gpu=0))
        reserved,global_wait=frequency.field_reservations(V,IO)
        assert reserved=={('silver',0),('olive',0),('dev3',0)}and not global_wait
        write(demands/'001680.json',dict(ready_for_training=True,status='awaiting_idle_training_slot'))
        assert frequency.field_reservations(V,IO)[1]
        current=IO.read(state);current['frames']['1675'].update(host='olive',gpu=0);current['jobs']['f1']['status']='complete';write(state,current)
        (demands/'001680.json').unlink();reserved,_=frequency.field_reservations(V,IO)
        assert reserved=={('silver',0)}
        c=dict(hosts={'silver':dict(python='/python',campaign='/canonical',root='/canonical/operations/future_rgb_sg/dispatch')})
        j=frequency.map_job(c,dict(host='silver',frame=1680,camera='AA',name='future001680_AA_a00'))
        assert j['command']==['/python','/canonical/code/LookCloser/scripts/marshawn_video_frequency.py','--spec','/canonical/operations/future_rgb_sg/001680/specs/AA.json']
        assert j['name']=='future001680_AA_a00'and j['env']=={}
    print(json.dumps(dict(passed=True,no_reserved_camera_subset=True,same_Silver_job_namespace=True,
                          explicit_field_slot_priority=True,completed_assignments_release_capacity=True)))


if __name__=='__main__':run()
