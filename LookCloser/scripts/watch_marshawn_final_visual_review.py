"""Fetch naturally completed final choices and prepare CPU-only review evidence."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def main(a):
    local=a.campaign/'independent_occupancy'/a.choices;local.mkdir(exist_ok=True)
    remote=a.remote_campaign+'/independent_occupancy/'+a.choices
    output=a.campaign/'independent_occupancy'/a.output
    while True:
        try:
            check=subprocess.run(['ssh','-o','ConnectTimeout=15',a.host,'test -f '+remote+'/complete.json'],capture_output=True,timeout=30)
            if check.returncode:time.sleep(60);continue
            subprocess.run(['rsync','-a','--exclude=checkpoint.ckpt',a.host+':'+remote+'/',str(local)+'/'],check=True,timeout=300)
            complete=json.loads((local/'complete.json').read_text())
            if not complete['complete']or not complete['selector_and_plateau_unchanged']:raise ValueError('Natural final choices not ready')
            for name,row in complete['choices'].items():
                grid=local/name/f"step-{row['step']:09d}_grid.npz"
                if hashlib.sha256(grid.read_bytes()).hexdigest()!=row['grid_sha256']:raise ValueError('Final grid receipt mismatch')
            source=a.campaign/'independent_occupancy/semantic_selected_geometry/R0_W64/renders'
            labels={'R0_selected111316':source}
            for key in ['established_selector','raw_lpips_min','terminal']:
                row=complete['choices'][key];labels[f"{a.label}_{key}_{row['step']}"]=local/key/'renders'
            if not(output/'receipt.json').exists():
                command=[a.python,str(a.opacity_helper),'--data',str(a.campaign/'data'),'--output',str(output)]
                for name,path in labels.items():command.extend(['--arm',name+'='+str(path)])
                subprocess.run(command,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=180)
            from PIL import Image,ImageDraw
            ready={r['camera']:r for r in json.loads((a.campaign/'data/ready.json').read_text())['records']}
            for camera,actor in [('DC','black'),('HC','grey'),('KA','black')]:
                sources=[('GT',a.campaign/'data'/ready[camera]['image'])]+[(label,folder/(camera+'.png'))for label,folder in labels.items()];tiles=[]
                for label,path in sources:
                    im=Image.open(path).convert('RGB').crop(ready[camera]['crops'][actor]);im.thumbnail((330,520));tile=Image.new('RGB',(im.width,im.height+40),'#ddd');tile.paste(im,(0,40));ImageDraw.Draw(tile).text((3,3),label.replace('_','\n',1),fill='black');tiles.append(tile)
                canvas=Image.new('RGB',(sum(x.width for x in tiles),max(x.height for x in tiles)),'#ddd');left=0
                for tile in tiles:canvas.paste(tile,(left,0));left+=tile.width
                canvas.save(output/f'{camera}_{actor}_comparison.jpg',quality=94)
            (output/'complete.json').write_text(json.dumps(dict(passed=True,geometry_quality_approved=False,
                choices=complete['choices'],checkpoint_ages_differ_explicitly=True,model_queries=0,optimizer_updates=0,
                mask_exterior_not_physical_free_space=True,source_complete_sha256=hashlib.sha256((local/'complete.json').read_bytes()).hexdigest()),indent=2)+'\n');return
        except Exception as error:
            with(local/'visual_fetch_errors.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),error=repr(error)))+'\n')
            time.sleep(60)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--remote-campaign',required=True);p.add_argument('--host',default='ubuntu@silver-sable');p.add_argument('--choices',required=True);p.add_argument('--output',required=True);p.add_argument('--label',required=True);p.add_argument('--python',required=True);p.add_argument('--opacity-helper',type=Path,required=True);main(p.parse_args())
