"""Early saved-view source/physical guards; rectangular crops, no mask clipping."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
from prepare_marshawn_frozen_semantic_cpu import sha,write


def sheet(images,labels,crop=None,thumbnail=None):
    tiles=[image.crop(crop)if crop is not None else image.copy()for image in images]
    if thumbnail is not None:
        for tile in tiles:tile.thumbnail(thumbnail)
    width=max(t.width for t in tiles);height=max(t.height for t in tiles)
    panel=Image.new('RGB',(len(tiles)*width,height+35),'#171717');draw=ImageDraw.Draw(panel)
    for i,(tile,label)in enumerate(zip(tiles,labels)):
        panel.paste(tile,(i*width,35));draw.text((i*width+3,4),label,fill='white')
    return panel


def run(c,physical,prior,camera,output):
    render=physical/prior/'renders';view=render/camera
    if not(view/'physical_and_partition.npz').exists():raise FileNotFoundError('Wait for this entire view archive to be saved')
    output.mkdir(parents=True,exist_ok=False)
    ready_path=c/'data/ready.json';ready=json.loads(ready_path.read_text());row=next(r for r in ready['records']if r['camera']==camera)
    regions_path=c/'native_semantic_grid_20000/corrected_v2/review/region_crops.json';regions=json.loads(regions_path.read_text())[camera]
    target=ready_path.parent/row['image']
    if sha(target)!=row['rgb_sha256']:raise ValueError('Frozen evaluation source changed')
    variants=['native_full','hard_theta01','hard_theta05','exactzero_smoothstep01_05']
    paths=[target]+[view/(name+'.png')for name in variants]
    labels=[camera+' GT',camera+' selected126316 source',prior+' hard.1',prior+' hard.5',prior+' smooth[.1,.5]']
    images=[Image.open(p).convert('RGB')for p in paths]
    if len({image.size for image in images})!=1:raise ValueError('Common original raster required')
    sheet(images,labels,thumbnail=(430,320)).save(output/'fullframe_overview.jpg',quality=94)
    crops=list(row['crops'].values());width,height=images[0].size
    body=[max(0,min(v[0]for v in crops)-48),max(0,min(v[1]for v in crops)-32),min(width,max(v[2]for v in crops)+48),min(height,max(v[3]for v in crops)+48)]
    sheet(images,labels,body,thumbnail=(340,560)).save(output/'body_guard.jpg',quality=95)
    for actor,crop in row['crops'].items():sheet(images,labels,crop).save(output/(actor+'_fixed_actor_crop.png'))
    details=[]
    for name,crop in regions.items():
        panel=sheet(images,labels,crop);panel.save(output/(name+'_fixed_source_guard.png'));panel.thumbnail((2000,430));details.append(panel)
    if details:
        combined=Image.new('RGB',(max(p.width for p in details),sum(p.height for p in details)),'#171717');y=0
        for panel in details:combined.paste(panel,(0,y));y+=panel.height
        combined.save(output/'all_fixed_details.jpg',quality=95)
    write(output/'early_review.json',dict(CPU_only=True,CUDA_started=False,optimizer_created=False,camera=camera,prior=prior,
        source_sha256={str(p):sha(p)for p in paths+[ready_path,regions_path,view/'physical_and_partition.npz']},
        no_semantic_masks_read=True,no_image_space_erasing_or_compositing=True,fixed_actor_crops=row['crops'],fixed_detail_crops=regions,
        full_frame_also_saved=True,full_run_completion_exists=(render/'complete.json').exists(),
        caveat='Provisional review of a saved complete view; full run/source/head immutability and all3-view metric gates remain mandatory. No selector promotion.'))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--physical-root',type=Path,required=True);parser.add_argument('--prior',choices=['prior001','prior099'],required=True)
    parser.add_argument('--camera',choices=['DC','HC','KA'],required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.campaign,a.physical_root,a.prior,a.camera,a.output)
