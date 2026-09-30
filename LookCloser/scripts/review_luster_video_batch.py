"""Build inspectable temporal and native-view sheets; never auto-accept a batch."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
from archive_luster_checkpoint import sha
from prepare_luster_video import write


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);args=p.parse_args()
    frames=[f'{i:06d}' for i in range(args.start,args.end+1)]
    if not 1<=len(frames)<=6:raise ValueError('Review one to six frames per sheet')
    root=args.root;out=root/'visual_reviews'/f'batch_{frames[0]}_{frames[-1]}';out.mkdir(parents=True,exist_ok=True)
    temporal=Image.new('RGB',(len(frames)*240,900));draw=ImageDraw.Draw(temporal);records=[]
    for column,frame in enumerate(frames):
        snapshot=json.loads((root/'snapshots'/f'{frame}.json').read_text())
        receipt=json.loads((root/'video_frames/receipts'/f'{frame}.json').read_text())
        if receipt['checkpoint_sha256']!=snapshot['archived_checkpoint']['sha256']:raise ValueError('Model receipt differs')
        if receipt['cameras_sha256']!=sha(root/'video_cameras.json'):raise ValueError('Camera receipt differs')
        for row,kind in enumerate(['body','detail']):
            item=receipt['images'][kind]
            if sha(item['path'])!=item['sha256']:raise ValueError('PNG receipt differs')
            im=Image.open(item['path']);im.thumbnail((240,426))
            temporal.paste(im,(column*240,row*450+24));draw.text((column*240+4,row*450+5),f'{frame} {kind}',fill='white')
        selected=json.loads(Path(snapshot['selection']).read_text());folder=Path(selected['render_dir'])
        native=Image.new('RGB',(1200,1080));labels=ImageDraw.Draw(native)
        heads=Image.new('RGB',(1200,900));head_labels=ImageDraw.Draw(heads)
        for index,view in enumerate(selected['per_view']):
            prefix=f'{view["split"]}_{view["index"]:03d}';x=index%2*600;y=index//2*360
            im=Image.open(folder/f'{prefix}_panel.jpg');im=im.crop((0,0,im.width*2//3,im.height));im.thumbnail((590,330))
            native.paste(im,(x,y+24));labels.text((x+5,y+4),f'{view["physical_camera"]} {view["split"]}: GT | render',fill='white')
            head=folder/f'{prefix}_head.png';hy=index//2*300
            if head.exists():
                im=Image.open(head);im.thumbnail((590,270));heads.paste(im,(x,hy+24))
            head_labels.text((x+5,hy+4),f'{view["physical_camera"]} head: GT | render',fill='white')
        native.save(out/f'{frame}_native.jpg',quality=94);heads.save(out/f'{frame}_heads.jpg',quality=94)
        records.append(dict(frame=frame,training=snapshot['training_selection'],export=snapshot['export_metrics'],
                            guard=receipt['guard_diagnostics'],native=str(out/f'{frame}_native.jpg'),heads=str(out/f'{frame}_heads.jpg')))
    temporal.save(out/'temporal.jpg',quality=94)
    write(out/'index.json',dict(frames=frames,records=records,temporal=str(out/'temporal.jpg'),review_status='pending_visual_inspection'))
    print(str(out))


if __name__=='__main__':main()
