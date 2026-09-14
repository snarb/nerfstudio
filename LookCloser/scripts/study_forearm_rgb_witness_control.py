"""Read-only RGB witness control on observed positive anchors and veto points."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
from joint_temporal_texture import read,sha,atomic_json,cameras,exr,display,ROOT
from study_confidence_depth_prior import project_integer
from forearm_rgb_witnesses import color_errors
import study_forearm_plane_transfer_v3 as prior


def summarize(chroma,rgb):
    old=(chroma<=.04).sum(0)>=3
    return dict(points=chroma.shape[1],old_qualified=int(old.sum()),
        controls=[dict(rgb_mean_abs_limit=t,qualified=int(((chroma<=.04)&(rgb<=t)).sum(0).__ge__(3).sum()),
            lost_old_qualified=int((old&(((chroma<=.04)&(rgb<=t)).sum(0)<3)).sum())) for t in [.04,.06,.08,.1,.12]])


def run(frame,output):
    prior.configure();v1=prior.v2.v1;rows,depths,hashes=v1.load_real(frame)
    actual,_,_=cameras(frame);profiles=np.load(ROOT/'parameters.npz')['log_gain'];gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(pair):
        i,r=pair
        return r['physical_camera'],np.rint(display(exr(r['file_path'])*np.exp(profiles[i]),gain)*255).clip(0,255).astype(np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:images=dict(pool.map(load,enumerate(actual)))
    anchorroot=Path('/mnt/data/dec5_forearm_multiview_anchors')/frame
    if sha(anchorroot/'anchors.npz')!=read(anchorroot/'result.json')['anchors_sha256']:raise ValueError('Changed positive anchors')
    anchors=np.load(anchorroot/'anchors.npz')['points'];records=[];arrays={}
    names=['F004_E005_1210FP','G004_E005_1211KO','B004_E005_1210VE']
    for name in names:
        ci=next(i for i,r in enumerate(rows) if r['physical_camera']==name);reference=rows[ci]
        uv,z=project_integer(reference,anchors);xy=np.rint(uv).astype(int)
        valid=(z>0)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        ids=np.flatnonzero(valid);ids=ids[np.abs(depths[ci][xy[ids,1],xy[ids,0]]-z[ids])<=.001]
        ids=ids[::max(1,int(np.ceil(len(ids)/1000)))]
        ch,rgb=color_errors(anchors[ids],reference,rows,depths,images)
        arrays[name+'_anchor_chroma']=ch;arrays[name+'_anchor_rgb']=rgb
        record=dict(camera=name,positive_anchor_control=summarize(ch,rgb))
        if frame=='001037':
            path=Path('/mnt/data/dec5_forearm_annotation_domain_only/free_space_diagnosis')/frame/(name+'.npz')
            data=np.load(path);ch,rgb=color_errors(data['observed'],reference,rows,depths,images)
            arrays[name+'_veto_chroma']=ch;arrays[name+'_veto_rgb']=rgb
            record.update(veto=summarize(ch,rgb),veto_input_sha256=sha(path))
        records.append(record)
    output=output/frame;output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'errors.npz',**arrays)
    atomic_json(output/'result.json',dict(frame=frame,records=records,depth_hashes=hashes,
        anchors_sha256=sha(anchorroot/'anchors.npz'),errors_sha256=sha(output/'errors.npz'),
        scripts={n:sha(Path(__file__).with_name(n)) for n in ['study_forearm_rgb_witness_control.py','forearm_rgb_witnesses.py','diagnose_forearm_color_witnesses.py']},
        source_rgb_sha256={r['file_path']:sha(r['file_path']) for r in actual},profiles_sha256=sha(ROOT/'parameters.npz'),
        exposure_sha256=sha(ROOT/'exposure.json'),geometry_changed=False,threshold_selected=False,
        positive_samples_not_independent=True,scope='train-only observed-anchor confidence diagnostic'))
    print(frame,records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True)
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_rgb_witness_control'))
    a=p.parse_args();run(a.frame,a.output)
