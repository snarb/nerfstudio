"""Every predeclared W32/pad1+.002 missed positive pixel, existing source only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,cv2
from scipy.ndimage import label
from PIL import Image,ImageDraw
from marshawn_actor_coordinates import raw_array_indices,raw_mask_coverage_to_HD,raw_camera,RIG

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main(c,out):
 out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2);iso=c.parent/'two_actor_isolation';base=iso/'confidence_native_hair_v2';manifest=base/'manifest.json';rows={r['camera']:r for r in read(manifest)['rows']};request=iso/'raw_native_sources/request.json';sources={r['camera']:r for r in read(request)['records']};frames={r['physical_camera']:r for r in read(c/'data/transforms.json')['frames']};inputpath=c/'independent_occupancy/source_support_fourway_CPU_v3/W32_semantic_gate_pad1_positives.json';misses=read(inputpath)['remaining_ray_miss_pixels'];assert len(misses)==91
 bindings={str(p):sha(p)for p in [manifest,request,inputpath,c/'data/transforms.json',RIG/'images.txt',RIG/'cameras.txt',Path(__file__),Path(__file__).with_name('marshawn_actor_coordinates.py')]};records=[];pages=[]
 for cam in sorted({r['camera']for r in misses}):
  row=rows[cam];src=sources[cam];frame=frames[cam];assert row['split']=='TRAIN';xy=np.array([[r['x'],r['y']]for r in misses if r['camera']==cam]);rp=Path(row['raw_archive']);hp=Path(row['archive']);mp=c/'data/semantics'/(cam+'.npz');up=c/'native_semantic_grid_20000/occlusion_v2/W32/archives'/(cam+'.npz')
  for p,d in [(rp,row['raw_archive_sha256']),(hp,row['archive_sha256']),(Path(src['image']),src['image_sha256'])]:assert sha(p)==d;bindings[str(p)]=d
  for p in [mp,up]:bindings[str(p)]=sha(p)
  K,d=raw_camera(cam);np.testing.assert_allclose(K,src['raw_OpenCV_K'],atol=1e-12,rtol=0);np.testing.assert_allclose(d,src['distortion'],atol=1e-12,rtol=0)
  for k in ['w','h','fl_x','fl_y','cx','cy']:assert frame[k]==row['camera_row'][k]
  with np.load(rp)as z:raw={k:z[k]for k in z.files}
  raw['addhair']=raw['nominal']&~(raw['black']|raw['grey']);box=raw['crop_xyxy'];assert raw['addhair'].sum()==row['native_hair_added_pixels']
  with np.load(hp)as z:original={k:z[k]for k in z.files}
  with np.load(mp)as z:main={k:z[k]for k in z.files}
  with np.load(up)as z:support={k:z[k]for k in z.files}
  coverage={}
  for k in ['black','grey','nominal','addhair','source_ineligible']:
   a=np.zeros((4320,8192),np.float32);a[box[1]:box[3],box[0]:box[2]]=raw[k];coverage[k]=raw_mask_coverage_to_HD(a,frame)[xy[:,1],xy[:,0]]
  image=Image.open(src['image']).convert('RGB');hdpath=c/'data'/frame['file_path'];hd=Image.open(hdpath).convert('RGB');bindings[str(hdpath)]=sha(hdpath);mask=np.zeros((frame['h'],frame['w']),bool);mask[xy[:,1],xy[:,0]]=True;labels,n=label(mask,np.ones((3,3)))
  for i,(x,y)in enumerate(xy):
   center=raw_array_indices(np.array([[x+.5,y+.5]]),frame)[0];corners=raw_array_indices(np.array([[x,y],[x+1,y],[x+1,y+1],[x,y+1]],float),frame);local=np.rint(center-box[:2]).astype(int)
   records.append(dict(camera=cam,HD_pixel=[int(x),int(y)],cluster=int(labels[y,x]),RAW_center=center.tolist(),RAW_geometric_pixel_corners=corners.tolist(),RAW_nearest={k:bool(raw[k][local[1],local[0]])for k in coverage},RAW_calibrated_AREA_coverage={k:float(v[i])for k,v in coverage.items()},current_HD={k:float(main[k][y,x])for k in ['black','grey','nominal','W64','core','band']},original_HD={k:float(original[k][y,x])for k in ['nominal_coverage','W64','source_ineligible']},corrected_UNKNOWN=bool(support['unknown'][y,x]),corrected_W32_support=bool(support['support'][y,x])))
  for cluster in range(1,n+1):
   ids=np.flatnonzero(labels[xy[:,1],xy[:,0]]==cluster);pixels=xy[ids];x0,y0=pixels.min(0)-35;x1,y1=pixels.max(0)+36;hdcrop=hd.crop((x0,y0,x1,y1));hdo=np.array(hdcrop).copy();unk=support['unknown'][y0:y1,x0:x1];hdo[unk]=(hdo[unk]*.4+np.array([255,0,255])*.6).astype(np.uint8);hdo=Image.fromarray(hdo);draw=ImageDraw.Draw(hdo)
   for x,y in pixels:draw.rectangle((x-x0,y-y0,x-x0+1,y-y0+1),outline='red')
   centers=raw_array_indices(pixels+.5,frame)-box[:2];left,top=np.floor(centers.min(0)-70).astype(int);right,bottom=np.ceil(centers.max(0)+71).astype(int);crop=image.crop((left,top,right,bottom));over=np.array(crop).copy();classes=np.zeros_like(over)
   for key,color in [('black',[40,130,255]),('grey',[50,255,90]),('addhair',[255,210,0]),('source_ineligible',[255,0,255])]:classes[raw[key][top:bottom,left:right]]=color
   selected=classes.any(-1);over[selected]=(over[selected]*.55+classes[selected]*.45).astype(np.uint8);over=Image.fromarray(over);draw=ImageDraw.Draw(over)
   for x,y in pixels:
    poly=raw_array_indices(np.array([[x,y],[x+1,y],[x+1,y+1],[x,y+1]],float),frame)-box[:2]-[left,top];draw.polygon([tuple(v)for v in poly],outline='red')
   panel=Image.new('RGB',(1440,500),'#202020');pd=ImageDraw.Draw(panel)
   items=[('HD RGB',hdcrop),('HD misses red / corrected UNKNOWN magenta',hdo),('Native RAW RGB',crop),('RAW black blue / grey green / added hair yellow',over)]
   for j,(title,tile)in enumerate(items):
    scale=min(350/tile.width,430/tile.height);tile=tile.resize((round(tile.width*scale),round(tile.height*scale)),Image.Resampling.NEAREST);panel.paste(tile,(j*360+(360-tile.width)//2,55));pd.text((j*360+5,30),title,fill='white')
   pd.text((5,5),f'{cam} cluster{cluster}: ALL {len(pixels)} missed HD pixels; RAW red outlines geometric cells (coverage uses AREA+remap)',fill='white');name=f'{cam}_cluster{cluster:02d}.png';panel.save(out/name);pages.append(dict(camera=cam,cluster=cluster,pixels=pixels.tolist(),page=name,page_sha256=sha(out/name)))
 assert len(records)==91 and len({(r['camera'],*r['HD_pixel'])for r in records})==91
 result=dict(CPU_only=True,no_new_labels=True,no_model=True,all91_pixels_present=True,pages=pages,records=records,source_hashes=bindings,limitations=['Original existing segmentation classes are evidence, not new labels or optical alpha GT.','RAW polygon shows geometric mapped cell; actual semantic coverage uses exact AREA prefilter and OpenCV linear remap.','UNKNOWN is source ambiguity, never a physical-free-space assertion.','Parent source remains fixed126316; no training or grid mutation.'])
 (out/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
