"""Independent serialized-name/matrix/forward replay for articulation mapping."""
from pathlib import Path
import argparse
import numpy as np
from build_train_hair_semantics import read,sha,write
from probe_mhr_articulation_mapping import ROOT,MODEL


def main(action):
    import torch
    torch.set_num_threads(2)
    record=read(ROOT/'mapping.json')
    assert record['model_sha256']==sha(MODEL/'mhr_model.pt')
    assert record['script_sha256']==sha(Path(__file__).with_name('probe_mhr_articulation_mapping.py'))
    assert record['definition_sha256']==sha(ROOT/'definition/assets/compact_v6_1.model')
    assert record['probe_sha256']==sha(ROOT/'pose_probe.npz')
    model=torch.jit.load(str(MODEL/'mhr_model.pt'),map_location='cpu').eval()
    names=list(model.character_torch.parameter_transform.parameter_names)
    joints=list(model.character_torch.skeleton.joint_names)
    matrix=model.character_torch.parameter_transform.parameter_transform.numpy()
    assert record['parameter_names']==names and record['joint_names']==joints
    with np.load(ROOT/'pose_probe.npz') as probe, torch.no_grad():
        np.testing.assert_array_equal(matrix,probe['parameter_transform'])
        base=model(torch.zeros(1,45),torch.zeros(1,204),torch.zeros(1,72))[0][0].numpy()
        np.testing.assert_array_equal(base,probe['base'])
        for k,item in enumerate(record['records']):
            i=names.index(item['name']);assert i==item['index'] and i<204
            rows=np.flatnonzero(matrix[:,i])
            expected=[dict(row=int(r),joint=joints[r//7],joint_slot=int(r%7),coefficient=float(matrix[r,i])) for r in rows]
            assert expected==item['matrix_effects']
            pose=torch.zeros(1,204);pose[0,i]=item['step']
            result=model(torch.zeros(1,45),pose,torch.zeros(1,72))[0][0].numpy()
            np.testing.assert_array_equal(result,probe['positive_point_one'][k])
            delta=np.linalg.norm(result-base,axis=1)
            assert float(np.sqrt(np.mean(delta[base[:,1]>=145]**2)))==item['head_rms_cm']
    if action=='seal':
        bindings={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
        for name in [Path(__file__).name,'probe_mhr_articulation_mapping.py']:
            p=Path(__file__).with_name(name).resolve();bindings[str(p)]=sha(p)
        p=Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_articulation_mapping.md'
        bindings[str(p)]=sha(p)
        write(ROOT/'artifact_manifest.json',dict(bindings=bindings,status='mapping_and_forward_replay_verified'))
        print('sealed',len(bindings),'bindings',flush=True)
    else:
        for path,digest in read(ROOT/'artifact_manifest.json')['bindings'].items():assert sha(path)==digest,path
        print('mapping and all seven forward surfaces verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['seal','check'])
    main(parser.parse_args().action)
