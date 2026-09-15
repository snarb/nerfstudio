"""Read actual serialized rig names/matrix and check bounded neck articulation."""
from pathlib import Path
import zipfile
import numpy as np
from build_train_hair_semantics import sha,write
from prepare_mhr_head_prior import ROOT as MODEL

ROOT=Path('/mnt/data/dec5_mhr_articulation_mapping')


def main():
    import torch
    torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=False)
    model=torch.jit.load(str(MODEL/'mhr_model.pt'),map_location='cpu').eval()
    transform=model.character_torch.parameter_transform
    names=list(transform.parameter_names);joints=list(model.character_torch.skeleton.joint_names)
    matrix=transform.parameter_transform.detach().numpy()
    assert matrix.shape[0]==7*len(joints) and matrix.shape[1]==len(names)
    with zipfile.ZipFile(MODEL/'assets.zip') as archive:
        archive.extract('assets/compact_v6_1.model',ROOT/'definition')
    definition=ROOT/'definition/assets/compact_v6_1.model'
    text=definition.read_text()
    selected=['neck_twist','neck_lean','neck_bend','head_twist','head_lean','head_bend']
    records=[];vertices=[]
    with torch.no_grad():
        base=model(torch.zeros(1,45),torch.zeros(1,204),torch.zeros(1,72))[0][0].numpy()
        for name in selected:
            index=names.index(name);assert index<204
            rows=np.flatnonzero(matrix[:,index])
            effects=[dict(row=int(r),joint=joints[r//7],joint_slot=int(r%7),coefficient=float(matrix[r,index])) for r in rows]
            assert effects and any(x['joint'] in ['c_neck','c_head'] for x in effects)
            pose=torch.zeros(1,204);pose[0,index]=.1
            v=model(torch.zeros(1,45),pose,torch.zeros(1,72))[0][0].numpy()
            assert np.isfinite(v).all();vertices.append(v)
            displacement=np.linalg.norm(v-base,axis=1)
            head=base[:,1]>=145;neck=(base[:,1]>=135)&(base[:,1]<145);body=base[:,1]<135
            records.append(dict(name=name,index=index,matrix_effects=effects,
                limit_definition=[line for line in text.splitlines() if line.startswith('limit '+name+' ')],
                step=.1,head_rms_cm=float(np.sqrt(np.mean(displacement[head]**2))),
                neck_rms_cm=float(np.sqrt(np.mean(displacement[neck]**2))),
                body_max_cm=float(displacement[body].max())))
    np.savez_compressed(ROOT/'pose_probe.npz',base=base,positive_point_one=np.stack(vertices),
        parameter_transform=matrix,triangles=dict(model.named_buffers())['character_torch.mesh.faces'].numpy())
    write(ROOT/'mapping.json',dict(model_sha256=sha(MODEL/'mhr_model.pt'),script_sha256=sha(__file__),
        definition_sha256=sha(definition),parameter_names=names,joint_names=joints,records=records,
        jaw_named_pose_parameters=[dict(index=i,name=n) for i,n in enumerate(names[:204]) if 'jaw' in n.lower()],
        inference='No jaw-named parameter among first204pose values. Do not guess a jaw pose index; facial expression controls need a separate official mapping.',
        dataset_used=False,geometry_changed=False,probe_sha256=sha(ROOT/'pose_probe.npz')))
    print([{k:r[k] for k in ['name','index','head_rms_cm','neck_rms_cm','body_max_cm']} for r in records],flush=True)


if __name__=='__main__':main()
