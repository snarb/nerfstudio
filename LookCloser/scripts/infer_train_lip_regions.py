"""Opt-in train-only lip landmarks for coherent texture-region experiments.

Run in the existing MediaPipe environment. This never changes depth/meshes,
reads held-out RGB or supplies an evaluation ROI. No downloads are performed.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from build_train_hair_semantics import read, sha, write, ROOT as STAGED

ROOT=Path('/mnt/data/dec5_train_lip_regions')
FRAMES=['001083','001123']
CAMERAS=['G004_C005_121037','H004_C005_1210SZ','I004_C005_1210BA']


def cycles(edges):
    neighbors={}
    for a,b in edges:
        neighbors.setdefault(a,set()).add(b);neighbors.setdefault(b,set()).add(a)
    assert all(len(n)==2 for n in neighbors.values())
    remaining=set(neighbors);result=[]
    while remaining:
        start=min(remaining);ring=[start];previous=None;current=start
        while True:
            nxt=min(neighbors[current]-({previous} if previous is not None else set()))
            if nxt==start:break
            assert nxt not in ring
            ring.append(nxt);previous,current=current,nxt
        remaining.difference_update(ring);result.append(ring)
    return result


def area(points):
    return .5*abs(np.sum(points[:,0]*np.roll(points[:,1],1)-points[:,1]*np.roll(points[:,0],1)))


def main():
    import mediapipe as mp
    package=Path(mp.__file__).parent
    models={str(p):sha(p) for folder in ['face_landmark','face_detection','iris_landmark']
        for p in (package/'modules'/folder).glob('*.tflite')}
    assert models
    source=read(STAGED/'request.json');ROOT.mkdir(exist_ok=False)
    request=dict(frames=FRAMES,cameras=CAMERAS,script_sha256=sha(__file__),
        staged_request_sha256=sha(STAGED/'request.json'),model_hashes=models,
        mediapipe_version=mp.__version__,model='FaceMesh static_image/refine_landmarks',
        maximum_faces=1,minimum_detection_confidence=.5,heldout_used=False,
        mesh_or_target_view_used=False,geometry_changed=False,evaluation_roi=False,
        native_crop_portrait=source['crop_portrait'])
    write(ROOT/'request.json',request);rings=cycles(mp.solutions.face_mesh.FACEMESH_LIPS)
    results=[]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,
            refine_landmarks=True,min_detection_confidence=.5) as model:
        for frame in FRAMES:
            staged=read(STAGED/frame/'stage.json');assert staged['request_sha256']==sha(STAGED/'request.json')
            dest=ROOT/frame;dest.mkdir()
            for camera in CAMERAS:
                row=next(r for r in staged['records'] if r['camera']==camera)
                assert sha(row['input_path'])==row['input_sha256']
                image=Image.open(row['input_path']).convert('RGB');rgb=np.asarray(image)
                detected=model.process(rgb);assert detected.multi_face_landmarks, (frame,camera)
                assert len(detected.multi_face_landmarks)==1
                points=np.array([(p.x*image.width,p.y*image.height) for p in detected.multi_face_landmarks[0].landmark])
                assert np.isfinite(points).all()
                outer=max(rings,key=lambda ids:area(points[ids]));polygon=points[outer]
                assert area(polygon)>25 and ((polygon>=0)&(polygon<[image.width,image.height])).all()
                mask=Image.new('L',image.size);ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon],fill=255)
                mask.save(dest/(camera+'_mask.png'))
                overlay=image.copy();ImageDraw.Draw(overlay).line([tuple(p) for p in np.vstack([polygon,polygon[0]])],fill=(0,255,0),width=2)
                box=(max(0,int(polygon[:,0].min())-45),max(0,int(polygon[:,1].min())-45),
                    min(image.width,int(polygon[:,0].max())+46),min(image.height,int(polygon[:,1].max())+46))
                overlay.crop(box).save(dest/(camera+'_review.png'))
                np.savez_compressed(dest/(camera+'_landmarks.npz'),points=points,outer_lip_indices=outer)
                result=dict(frame=frame,camera=camera,input_path=row['input_path'],input_sha256=row['input_sha256'],
                    source_sha256=row['source_sha256'],outer_polygon_area=area(polygon),review_box=box,
                    hashes={suffix:sha(dest/(camera+suffix)) for suffix in ['_mask.png','_review.png','_landmarks.npz']},
                    visual_status='pending')
                write(dest/(camera+'.json'),result);results.append(result)
    write(ROOT/'result.json',dict(request_sha256=sha(ROOT/'request.json'),records=results,production_used=False))
    print([(r['frame'],r['camera'],r['outer_polygon_area']) for r in results],flush=True)


if __name__=='__main__':main()
