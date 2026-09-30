"""Follow moving head and clothing with train-hull projections, before training."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from prepare_luster_video import write


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    if not (args.root/'preparation_complete.json').exists():raise ValueError('Freeze sequence preparation before ROIs')
    frames=json.loads((args.root/'manifest.json').read_text())['frames'];panels=[]
    for index,frame in enumerate(frames):
        data=args.root/'frames'/frame/'data'
        if not (data/'complete.json').exists():continue
        if (data/'video_rois_receipt.json').exists():continue
        meta=json.loads((data/'transforms.json').read_text());points=np.load(data/'hull.npz')['points']
        low,high=points[:,2].min(),points[:,2].max();height=high-low
        groups={'head':points[points[:,2]>high-.31*height],
                'upper_body':points[points[:,2]>low+.48*height],
                'clothing':points[(points[:,2]>low+.08*height)&(points[:,2]<low+.50*height)]}
        rois={};skipped=[]
        for row in meta['frames']:
            if row['camera_id'] not in [11,12,95,97,150,151]:continue
            name=Path(row['file_path']).name;pose=np.array(row['transform_matrix']);rois[name]={}
            for label,cloud in groups.items():
                xyz=(cloud-pose[:3,3])@pose[:3,:3];xyz=xyz[xyz[:,2]<-1e-6]
                if not len(xyz):
                    skipped.append(dict(image=name,region=label,reason='Region behind camera'));continue
                uv=np.column_stack([xyz[:,0]/-xyz[:,2]*row['fl_x']+row['cx'],-xyz[:,1]/-xyz[:,2]*row['fl_y']+row['cy']])
                a=uv.min(0);b=uv.max(0);padding=np.maximum((b-a)*.05,8)
                a=np.maximum(np.floor(a-padding),[0,0]).astype(int);b=np.minimum(np.ceil(b+padding),[row['w'],row['h']]).astype(int)
                if np.min(b-a)<32:skipped.append(dict(image=name,region=label,reason='Projected body region outside the camera'));continue
                rois[name][label]=[*a.tolist(),*b.tolist()]
            if index in [0,15,30,45,59]:
                im=Image.open(data/row['file_path']).convert('RGB');draw=ImageDraw.Draw(im)
                for label,box in rois[name].items():draw.rectangle(box,outline={'head':'yellow','upper_body':'cyan','clothing':'red'}[label],width=4)
                im.thumbnail((240,328));panel=Image.new('RGB',(240,350));panel.paste(im,(0,22));ImageDraw.Draw(panel).text((2,2),f'{frame} {row["physical_camera"]}',fill='white');panels.append(panel)
        write(data/'video_rois.json',rois)
        write(data/'video_rois_receipt.json',dict(protocol='Current train-only visual hull; head top31%, upper body above48%, clothing8..50% of hull height; 5% or8px padding',skipped=skipped))
    if panels:
        sheet=Image.new('RGB',(6*240,((len(panels)+5)//6)*350))
        for i,panel in enumerate(panels):sheet.paste(panel,((i%6)*240,(i//6)*350))
        destination=args.root/'preflight/video_roi_contact.jpg';sheet.save(destination,quality=92)
    write(args.root/'video_rois_complete.json',dict(frames=frames))


if __name__=='__main__':main()
