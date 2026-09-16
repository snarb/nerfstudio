"""Seal the inspected subdivision controls without promoting a video candidate."""
from pathlib import Path
import numpy as np
from PIL import Image
from study_multiview_face_prior import read,save,sha
from study_train_gap_subfaces import ROOT,FRAME
from review_measured_free_surface import VIEWS


def main():
    root=ROOT/FRAME;dest=root/'visual_review.json';assert not dest.exists()
    bindings={}
    def verify(p,h):
        assert sha(p)==h,str(p)
        bindings[str(p)]=h
    q=read(root/'request.json');r=read(root/'result.json')
    verify(root/'request.json',r['request_sha256'])
    verify(Path(__file__).with_name('study_train_gap_subfaces.py'),q['subface_script_sha256'])
    for p,h in q['helper_hashes'].items():verify(p,h)
    for p,h in q['source_masks'].items():verify(p,h)
    for p,h in r['hashes'].items():verify(root/p,h)
    for arm in ['refined','carved']:
        folder=ROOT/arm/FRAME;result=read(folder/'result.json')
        verify(folder/'request.json',result['request_sha256'])
        for p,h in result['hashes'].items():verify(folder/p,h)
    review=read(root/'review/result.json')
    for p,h in review['input_hashes'].items():verify(p,h)
    for p,h in review['images'].items():verify(root/'review'/p,h)
    verify(root/'independent_audit.json',review['independent_audit_sha256'])
    for folder in [root/'review',root/'remaining']:
        result=read(folder/'result.json')
        for p,h in result['input_hashes'].items():verify(p,h)
        for p,h in result.get('outputs',{}).items():verify(folder/p,h)
        verify(folder/'result.json',sha(folder/'result.json'))
    viewed=[]
    for view in VIEWS:
        viewed += [root/'review'/view/'lipstick_native.png',root/'review'/view/'head_native.png',
                   root/'remaining'/(view+'_new_black_native.png')]
    # Newly exposed surfaces are not eligible for same-surface color reuse.
    a=ROOT/'refined'/FRAME/'rgb/moving/frames'/FRAME
    b=ROOT/'carved'/FRAME/'rgb/moving/frames'/FRAME
    old=np.array(Image.open(a/'prediction_native.png'));new=np.array(Image.open(b/'prediction_native.png'))
    d0=np.load(a/'target_depth.npz')['depth'];d1=np.load(b/'target_depth.npz')['depth']
    missing=(old.max(2)>0)&(new.max(2)==0);y,x=np.nonzero(missing)
    pixels=[dict(landscape_xy=[int(xx),int(yy)],depth_before=float(d0[yy,xx]),
        depth_after=float(d1[yy,xx]),depth_delta=float(d1[yy,xx]-d0[yy,xx])) for yy,xx in zip(y,x)]
    assert len(pixels)==5
    save(dest,dict(reviewer='root LLM actual image inspection',
        viewed_images={str(p):sha(p) for p in viewed},head_panels_viewed_as_overviews=True,
        native_lipstick_and_all_newblack_components_viewed=True,
        checked_bindings=bindings,checked_binding_count=len(bindings),
        visual_status='partial_improvement_not_artifact_free',
        notes='Refined carving further reduces the blue gap remnant and keeps the pink tip. Residual narrow blue fringe and irregular finger/barrel junction remain. Moving view has a small newly black cluster. Crown/fringe/chin issues persist. No continuous video or temporal transfer validation.',
        moving_new_black_vs_refined=pixels,
        missing_color_on_newly_exposed_surface=int(((d1-d0>1e-5)&missing).sum()),
        geometry_not_missing_at_new_black=bool((d1[missing]>0).all()),
        no_claim_same_surface_backoff_can_fill_new_surfaces=True,
        production_promoted=False,video_changed=False,cheek_hole_goal_incomplete=True,
        script_sha256=sha(__file__)))
    print('verified',len(bindings),'bindings;9 image panels reviewed;newly exposed black pixels',sum(p['depth_delta']>1e-5 for p in pixels),flush=True)


if __name__=='__main__':main()
