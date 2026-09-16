"""Trace actual post-pruning visible faces; the old cohort is not a quality score."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from study_multiview_face_prior import read,save,sha
from study_query_support_quorum import ROOT,FRAME,SOURCE
from diagnose_lipstick_fin_depth import POLYGON,BOX,CAMERA
from admit_mhr_local_patch_depth import Scene2
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import panel


def main():
    root=ROOT/FRAME; dest=root/'remaining_fin';assert not dest.exists()
    q=read(root/'request.json');e=np.load(root/'evidence.npz');old_e=np.load(SOURCE/'evidence.npz')
    original=o3d.io.read_triangle_mesh(q['mesh']); v,t=np.asarray(original.vertices),np.asarray(original.triangles)
    folder=root/'rgb'/CAMERA/'frames'/FRAME; camera=read(folder/'result.json')['camera']
    regions=Image.new('L',(1080,1920));ImageDraw.Draw(regions).polygon(POLYGON,fill=1);region=np.asarray(regions,bool)
    earlier=np.load('/mnt/data/dec5_lipstick_fin_depth/000995/evidence.npz')['triangle_ids']
    result={};arrays={};images=[]
    for label,removed,prediction_root in [('any_near',old_e['removed_triangle_ids'],SOURCE),('quorum3',e['removed_triangle_ids'],root)]:
        keep=np.ones(len(t),bool);keep[removed]=False;original_ids=np.flatnonzero(keep)
        d,ids,_=camera_depth(Scene2(v,t[keep]),camera)
        render_folder=prediction_root/'rgb'/CAMERA/'frames'/FRAME
        rd=np.load(render_folder/'target_depth.npz')['depth'];valid=rd>0
        np.testing.assert_allclose(d[valid],rd[valid],atol=1e-6,rtol=0)
        mapped=np.full(d.shape,-1,int);mapped[valid]=original_ids[ids[valid]];portrait=np.rot90(mapped)
        chosen=region&(portrait>=0);facets,counts=np.unique(portrait[chosen],return_counts=True)
        details=[]
        for face,count in zip(facets,counts):
            near=e['near_counts'][e['sample_indices'][face]];far=e['stable_far_counts'][e['sample_indices'][face]]
            details.append(dict(face=int(face),polygon_pixels=int(count),in_old_diagnostic=bool(face in earlier),
                near_counts=near.tolist(),stable_far_counts=far.tolist(),
                protected_by_query_quorum=bool((near>=3).any()),fails_stable_far=bool((far<6).any())))
        result[label]=dict(polygon_hits=int(chosen.sum()),visible_faces=len(facets),
            outside_old_diagnostic_faces=int((~np.isin(facets,earlier)).sum()),
            outside_old_diagnostic_pixels=int(counts[~np.isin(facets,earlier)].sum()),details=details)
        rgb=np.asarray(Image.open(render_folder/'frame.png')).copy();marked=rgb.copy()
        marked[chosen&np.isin(portrait,earlier)]=[255,100,0]
        marked[chosen&~np.isin(portrait,earlier)]=[0,220,255]
        images.extend([rgb,marked]);arrays[label+'_original_face_ids']=portrait
    dest.mkdir();panel(dest/'ancestry.png',images,['any near','orange old / cyan other','quorum3','orange old / cyan other'],BOX)
    np.savez_compressed(dest/'evidence.npz',**arrays,polygon=region)
    paths=[root/'request.json',root/'result.json',root/'evidence.npz',SOURCE/'evidence.npz',Path(__file__),
        Path('/mnt/data/dec5_lipstick_fin_depth/000995/evidence.npz')]
    save(dest/'result.json',dict(records=result,diagnostic_polygon_not_geometry_selection=True,
        input_hashes={str(p):sha(p) for p in paths},outputs={p.name:sha(p) for p in dest.iterdir() if p.is_file()},
        not_a_missing_anatomy_or_quality_metric=True,production_changed=False))
    print({k:{name:value for name,value in r.items() if name!='details'} for k,r in result.items()},flush=True)


if __name__=='__main__':main()
