"""Self-contained review bundle with verified native images and provenance."""
import argparse
import csv
import io
from pathlib import Path
import shutil
import tarfile
import numpy as np
from PIL import Image,ImageDraw
from luster_hybrid_common import TARGETS,read,sha,write


def export(root,name,status='provisional'):
    candidate=root/'candidates'/name
    provenance=read(candidate/'provenance.json')
    assert provenance['deep'], 'Deep provenance verification is required before export'
    for rel,digest in provenance['outputs'].items():assert sha(candidate/rel)==digest
    metrics=read(root/'evaluation'/f'{name}_metrics.json');base=read(root/'evaluation/baseline_metrics.json')
    repeat=read(root/'evaluation/baseline_repeat1_metrics.json')
    assert metrics['device']==repeat['device']=='cuda'
    assert repeat['rows']==base['rows'], 'Baseline CUDA repeat must match exactly'
    assert metrics['protocol_sha256']==base['protocol_sha256']
    summary=read(root/'evaluation/protocol.json')
    out=root/'exports'/name
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    def copy(src,rel):
        dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert sha(src)==sha(dest)
    for c in TARGETS:
        d=f'cam{c:03d}'
        copy(summary['cameras'][str(c)]['source'],f'{d}/GT.jpg')
        copy(root/'baseline'/f'cam{c:03d}.png',f'{d}/LookCloser.png')
        copy(candidate/f'cam{c:03d}.png',f'{d}/hybrid.png')
        copy(candidate/f'cam{c:03d}_receipt.json',f'{d}/prediction_receipt.json')
        for kind in ['transfer','sources']:
            p=candidate/f'cam{c:03d}_{kind}.png'
            if p.exists():copy(p,f'{d}/{kind}.png')
        for kind in ['sources','confidence','reasons']:
            copy(candidate/f'cam{c:03d}_{kind}.npy',f'{d}/{kind}.npy')
        for p in (root/'evaluation'/name).glob(f'cam{c:03d}_*'):
            copy(p,f'{d}/crops/{p.name}')
        labels=np.load(candidate/f'cam{c:03d}_sources.npy',mmap_mode='r')
        colors=np.asarray(Image.open(candidate/f'cam{c:03d}_sources.png').convert('RGB'))
        mask=np.asarray(Image.open(summary['cameras'][str(c)]['mask']))>0
        source_counts=[]
        used=np.unique(labels[labels>=0])
        legend=Image.new('RGB',(420,max(1,len(used))*24),(35,35,35));legend_draw=ImageDraw.Draw(legend)
        for i,source in enumerate(used):
            yy,xx=np.nonzero(labels==source);color=tuple(map(int,colors[yy[0],xx[0]]))
            pixels=int(np.sum((labels==source)&mask));source_counts.append(dict(camera=int(source),person_pixels=pixels,person_fraction=pixels/int(mask.sum()),map_rgb=color))
            legend_draw.rectangle((4,i*24+4,24,i*24+20),fill=color)
            legend_draw.text((32,i*24+5),f'cam{source:03d}: {100*pixels/int(mask.sum()):.2f}% of person',fill='white')
        legend.save(out/d/'source_legend.png');write(out/d/'source_legend.json',source_counts)
        # Native images remain separate; this contact sheet is explicitly a preview.
        tiles=[]
        for title,path in [('GT',out/d/'GT.jpg'),('LookCloser',out/d/'LookCloser.png'),('hybrid',out/d/'hybrid.png')]:
            im=Image.open(path).convert('RGB');im.thumbnail((600,820));tiles.append((title,im))
        sheet=Image.new('RGB',(1800,850),(45,45,45));draw=ImageDraw.Draw(sheet)
        for i,(title,im) in enumerate(tiles):sheet.paste(im,(600*i,30));draw.text((600*i+10,8),title,fill='white')
        sheet.save(out/d/'comparison_preview.jpg',quality=95)
    for src,rel in [(candidate/'config.json','recipe.json'),(candidate/'provenance.json','provenance.json'),
                    (root/'evaluation'/f'{name}_metrics.json','hybrid_metrics.json'),
                    (root/'evaluation/baseline_metrics.json','baseline_metrics.json'),
                    (root/'evaluation/baseline_repeat1_metrics.json','baseline_cuda_repeat.json'),
                    (root/'evaluation/protocol.json','evaluation_protocol.json'),
                    (root/'preflight/manifest.json','input_manifest.json'),
                    (Path(provenance['warp_root'])/'preflight/sources.json','source_selection.json')]:copy(src,rel)
    for filename,digest in provenance['code'].items():
        path=root/'code/LookCloser/scripts'/filename
        assert sha(path)==digest, 'Run the final provenance audit after code changes before export'
        copy(path,f'code/{filename}')
    old={(r['camera'],r['region']):r for r in base['rows']}
    with (out/'metrics.csv').open('w',newline='') as f:
        fields=['camera','region','status','metric','LookCloser','hybrid','delta'];writer=csv.DictWriter(f,fields);writer.writeheader()
        for row in metrics['rows']:
            if row['status']!='measured':continue
            b=old[row['camera'],row['region']]
            for metric in ['lpips','psnr','ssim']:
                writer.writerow(dict(camera=row['camera'],region=row['region'],status=row['status'],metric=metric,
                                     LookCloser=b[metric],hybrid=row[metric],delta=row[metric]-b[metric]))
    reduction=100*(1-metrics['validation_mean']['lpips']/base['validation_mean']['lpips'])
    description=[f'# Luster 000540: {name}',f'Status: **{status}**. Native validation person LPIPS reduction: **{reduction:.2f}%**.',
      'Validation 012/095/150 is used for recipe selection, not an independent test. Cam007 excludes its own photograph from sources.',
      'The frozen model, calibration, AABB and grids are unchanged. Full-resolution images are separate files; comparison_preview.jpg is only a preview.',
      '','| Camera | Baseline LPIPS | Hybrid LPIPS |','|---|---:|---:|']
    for row in metrics['rows']:
        if row['region']=='person':description.append(f"| {row['camera']:03d} | {old[row['camera'],'person']['lpips']:.8f} | {row['lpips']:.8f} |")
    description+=['','| Camera | Exact LookCloser preserved within person |','|---|---:|']
    for row in metrics['coverage']:
        if row['region']=='person':description.append(f"| {row['camera']:03d} | {100*row['preserved_fraction']:.2f}% |")
    description+=['','Audited source snapshots are included in code/. Original checkpoint, photos and geometry inputs remain at the paths recorded in the provenance.',
                  'Source legends identify every used camera and its person coverage. A source label identifies the only photograph supplying texture at that pixel; LookCloser can still contribute broad appearance and boundary fading.',
                  'See metrics.csv for all regions and metrics; hybrid_metrics.json includes preserved/changed fractions. Sources and transfer maps accompany every camera.',
                  'No target validation RGB or evaluation masks are used to construct predictions. Existing geometric defects can remain. This status does not claim the 40% goal is achieved unless explicitly accepted in the campaign report.']
    (out/'README.md').write_text('\n'.join(description)+'\n')
    copy(root/'code/LookCloser/experiments/luster_000540_native_hybrid.md','campaign_report.md')
    inventory={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'SHA256.json',inventory)
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:tar.add(out,arcname=name)
    with tarfile.open(archive,'r:gz') as tar:
        for rel,digest in inventory.items():
            import hashlib
            f=tar.extractfile(f'{name}/{rel}');h=hashlib.sha256()
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==digest
    receipt=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,status=status,
                 validation_lpips_reduction_percent=reduction,verified_archive_members=len(inventory),
                 download=f'scp olive-lark:{archive} .')
    write(root/'exports'/f'{name}_receipt.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--status',default='provisional');a=p.parse_args();export(a.root,a.name,a.status)
