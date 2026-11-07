"""Join two CPU validators for126 fits without assembling or promoting maps."""
import argparse,json,hashlib
from pathlib import Path


def main(root,local,silver,output):
    if output.exists():raise ValueError('Preserve prior review generation')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    release=json.loads((root/'root_release.json').read_text());assert release.get('activation_sha256')==sha(root/'activation.json')and release.get('approved_map_fits')is True
    expected=json.loads((root/'jobs.json').read_text());reports=[json.loads(p.read_text())for p in [local,silver]]
    assert all(r['passed']and r['CPU_only']and r['activation_sha256']==sha(root/'activation.json')for r in reports)
    rows=[v for report in reports for v in report['rows']];by_name={v['name']:v for v in rows};assert len(rows)==len(by_name)==len(expected)==126
    for e in expected:
        r=by_name[e['name']];assert all(r[k]==e[k]for k in ['camera','variant','host']);assert r['source_sha256']==e['source_image_sha256']
    pairs=[]
    for camera in sorted({r['camera']for r in rows}):
        pair=[r for r in rows if r['camera']==camera];assert len(pair)==2 and {r['variant']for r in pair}=={'lr_decay','threshold093'}
        for key in ['host','gpu_uuid','initial_parameters','first8_random_draws','final_rng']:assert pair[0][key]==pair[1][key],(camera,key)
        pairs.append(dict(camera=camera,host=pair[0]['host'],gpu_uuid=pair[0]['gpu_uuid'],initial_parameters_equal=True,first8_draws_equal=True,final_rng_equal=True))
    output.write_text(json.dumps(dict(passed=True,jobs=126,cameras=63,source_reports_sha256={str(p):sha(p)for p in [local,silver]},pairs=pairs,rows=rows,assembly_performed=False,READY_written=False,NeRF_training_authorized=False,promotion=False,next_gate='Separate source/map visual review and explicit data-assembly/NeRF design. Lower censoring and changed labels are not3D quality metrics.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','local','silver','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();main(a.root,a.local,a.silver,a.output)
