"""Independent saved-output check: opacity support vs exact projected AABB hull."""
import argparse,hashlib,json
from pathlib import Path
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(manifest,output):
    payload=json.loads(Path(manifest).read_text());output=Path(output);output.mkdir(exist_ok=False)
    cameras={r['camera']:r for r in payload['cameras_actual_runtime']};rows=[]
    for variant,record in payload['variants'].items():
        bounds=np.asarray(record['exact_AABB_prepared_world'],np.float64)
        corners=np.where(np.asarray(list(product([0,1],repeat=3))),bounds[1],bounds[0])
        for name,files in record['files'].items():
            cam=cameras[name];R=np.asarray(cam['camera_to_world_nerfstudio'],np.float64)
            local=(corners-R[:3,3])@R[:3,:3];depth=-local[:,2]
            if (depth<=0).any():raise ValueError('Non-positive corner depth')
            uv=np.stack([cam['fx']*local[:,0]/depth+cam['cx']-.5,-cam['fy']*local[:,1]/depth+cam['cy']-.5],-1)
            hull=ConvexHull(uv);equations=hull.equations
            with np.load(files['raw'],allow_pickle=False)as data:alpha=data['alpha'].copy()
            result={}
            for threshold in [0.,.01,.5]:
                y,x=np.where(alpha>threshold);xy=np.stack([x,y],-1)
                distances=xy@equations[:,:2].T+equations[:,2];outside=distances.max(-1)
                ids=np.where(outside>1e-5)[0];worst=np.argsort(outside)[-min(12,len(outside)):][::-1]
                result[str(threshold)]={'nonzero_pixels':len(x),'outside_hull_pixels':len(ids),
                    'outside_more_than_one_pixel':int((outside>1).sum()),'maximum_signed_distance_pixels':float(outside.max())if len(x)else None,
                    'examples_xy_alpha_signeddistance':[[int(x[i]),int(y[i]),float(alpha[y[i],x[i]]),float(outside[i])]for i in worst if outside[i]>0]}
            im=Image.open(files['premult_black']).convert('RGB');draw=ImageDraw.Draw(im);vertices=[tuple(v)for v in uv[hull.vertices]]
            draw.line(vertices+[vertices[0]],fill=(255,50,0),width=3);im.thumbnail((960,506));im.save(output/(variant+'_'+name+'_projected_hull.jpg'))
            rows.append(dict(variant=variant,camera=name,raw_source=files['raw'],raw_sha256=sha(files['raw']),
                projected_corners_array_xy=uv.tolist(),projected_convex_hull_array_xy=uv[hull.vertices].tolist(),thresholds=result))
    (output/'audit.json').write_text(json.dumps(dict(manifest=str(manifest),manifest_sha256=sha(manifest),rows=rows,
        scope='Saved alpha support consistency with actual camera projection ONLY; NOT evidence unclipped actor fits AABB',
        NN_forward_or_updates=0,helper_sha256=sha(__file__)),indent=2)+'\n')
    for r in rows:print(r['variant'],r['camera'],r['thresholds']['0.5'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.manifest,a.output)
