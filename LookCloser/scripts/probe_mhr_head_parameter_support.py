"""Model-only head-basis support; spatial bands are diagnostics, not scene masks."""
from pathlib import Path
import numpy as np
from build_train_hair_semantics import read, sha, write
from prepare_mhr_head_prior import ROOT as MODEL

ROOT=Path('/mnt/data/dec5_mhr_head_parameter_support')


def differences(base, positive, negative):
    """Centered unit-coefficient derivative and symmetric nonlinear residual."""
    if positive.shape != negative.shape or positive.shape[1:] != base.shape:
        raise ValueError('Mismatched shape family')
    if not all(np.isfinite(x).all() for x in [base, positive, negative]):
        raise ValueError('Nonfinite model output')
    return (positive-negative)/2, (positive+negative)/2-base


def main():
    import torch
    import open3d as o3d
    from PIL import Image, ImageDraw
    torch.set_num_threads(2)
    ROOT.mkdir(exist_ok=False)
    write(ROOT/'request.json',dict(script_sha256=sha(__file__),model_sha256=sha(MODEL/'mhr_model.pt'),
        model_request_sha256=sha(MODEL/'request.json'),head_coefficients=list(range(20,40)),
        step=1.,pose_expression_body_parameters_frozen=True,dataset_used=False,
        bands='Neutral model centimeters; head y>=145, neck135<=y<145, lower-front145<=y<=153 and z>=0. Review-only spatial bands, not anatomical GT or fitted scene masks.'))
    model=torch.jit.load(str(MODEL/'mhr_model.pt'),map_location='cpu').eval()
    outputs=[]
    with torch.no_grad():
        base=model(torch.zeros(1,45),torch.zeros(1,204),torch.zeros(1,72))[0][0].numpy()
        for sign in [1.,-1.]:
            values=[]
            for first in range(20,40,4):
                identity=torch.zeros(4,45)
                identity[torch.arange(4),torch.arange(first,first+4)]=sign
                values.append(model(identity,torch.zeros(4,204),torch.zeros(4,72))[0].numpy())
            outputs.append(np.concatenate(values))
    derivative,residual=differences(base,*outputs)
    bands=dict(head=base[:,1]>=145,neck=(base[:,1]>=135)&(base[:,1]<145),
        lower_front=(base[:,1]>=145)&(base[:,1]<=153)&(base[:,2]>=0),below_neck=base[:,1]<135)
    records={}
    for name,mask in bands.items():
        matrix=derivative[:,mask].reshape(20,-1)
        gram=matrix@matrix.T/max(int(mask.sum()),1)
        spectrum=np.sqrt(np.maximum(np.linalg.eigvalsh(gram)[::-1],0))
        rms=np.sqrt(np.mean(np.sum(derivative[:,mask]**2,axis=-1),axis=1))
        records[name]=dict(vertices=int(mask.sum()),coefficient_rms_cm=rms.tolist(),
            singular_values_cm_per_sqrt_vertex=spectrum.tolist(),
            maximum_symmetric_residual_cm=float(np.linalg.norm(residual[:,mask],axis=-1).max()))
    order=np.argsort(records['lower_front']['coefficient_rms_cm'])[::-1][:3]
    faces=dict(model.named_buffers())['character_torch.mesh.faces'].numpy()
    np.savez_compressed(ROOT/'evidence.npz',base=base,positive=outputs[0],negative=outputs[1],
        derivative=derivative,symmetric_residual=residual,faces=faces,**bands)
    # Same low-oblique camera for every source model, without fitting DEC5.
    camera=np.array([45.,110.,80.]);center=np.array([0.,154.,2.])
    forward=center-camera;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,1.,0.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    yy,xx=np.mgrid[:420,:420]
    origins=camera+((xx-209.5)/420*42)[...,None]*right+((209.5-yy)/420*42)[...,None]*up
    rays=o3d.core.Tensor(np.concatenate([origins,np.broadcast_to(forward,origins.shape)],axis=2).astype(np.float32))
    panel=Image.new('RGB',(3*420,3*444),(20,20,20));draw=ImageDraw.Draw(panel)
    for row,mode in enumerate(order):
        for col,(label,v) in enumerate([('-1',outputs[1][mode]),('neutral',base),('+1',outputs[0][mode])]):
            mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(faces));mesh.compute_triangle_normals()
            scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            hit=scene.cast_rays(rays);depth=hit['t_hit'].numpy();ids=hit['primitive_ids'].numpy();valid=np.isfinite(depth)
            rgb=np.full((420,420,3),20,np.uint8)
            shade=np.clip(65+175*np.abs(np.asarray(mesh.triangle_normals)[ids[valid]]@-forward),0,255).astype(np.uint8)
            rgb[valid]=shade[:,None];im=Image.fromarray(rgb)
            panel.paste(im,(col*420,row*444+24));draw.text((col*420+5,row*444+5),f'head coefficient {mode+20}: {label}',fill='white')
    panel.save(ROOT/'lower_jaw_parameter_variants.png')
    write(ROOT/'result.json',dict(records=records,shown_head_coefficients=(order+20).tolist(),
        evidence_sha256=sha(ROOT/'evidence.npz'),panel_sha256=sha(ROOT/'lower_jaw_parameter_variants.png'),
        model_only=True,anatomical_bands_are_approximate=True,visual_status='pending',
        actor_fit_or_repair=False))
    print('lower front RMS cm',records['lower_front']['coefficient_rms_cm'],flush=True)
    print('top shown coefficients',(order+20).tolist(),flush=True)


if __name__=='__main__':main()
