"""Show unchanged AABB beside independent source-ROI depth envelopes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from marshawn_aabb_overlay import project_box


def main(a):
    source = json.loads(a.input.read_text())
    receipt = json.loads((a.output/'roi_frusta.json').read_text())
    rows = {r['camera']: r for r in source['cameras']}
    bounds = np.array(receipt['aabb'])
    colors = ['#00ff80', '#ffaa00', '#00dfff', '#ff60ff']
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    fig, axes = plt.subplots(3, 1, figsize=(7, 10.12), dpi=100)
    for ax, (x, y) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.add_patch(Rectangle((bounds[0, x], bounds[0, y]),
            bounds[1, x]-bounds[0, x], bounds[1, y]-bounds[0, y],
            fill=False, edgecolor='red', lw=2))
        for (name, foot), color in zip(receipt['feet'].items(), colors):
            e = np.array(foot['padding_sensitivity']['10']['extent'])
            ax.add_patch(Rectangle((e[0, x], e[0, y]), e[1, x]-e[0, x],
                e[1, y]-e[0, y], alpha=.25, facecolor=color, edgecolor='black'))
            ax.text(e[:, x].mean(), e[:, y].mean(), name, fontsize=7, ha='center')
        ax.set_xlim(bounds[0, x]-.04, bounds[1, x]+.04)
        ax.set_ylim(bounds[0, y]-.03, bounds[1, y]+.03)
        ax.set_aspect('equal'); ax.set_xlabel('XYZ'[x]); ax.set_ylabel('XYZ'[y])
        ax.grid(alpha=.2)
    axes[0].set_title('RED = unchanged common002 box\nColored = independent boot ROI depth envelopes\n+10 HD pixels each ROI side; no box filter', fontsize=10)
    fig.tight_layout(); fig.savefig(a.output/'orthographic_envelopes.png'); plt.close(fig)
    plot = Image.open(a.output/'orthographic_envelopes.png').convert('RGB')
    for camera in ['DC', 'HC', 'KA']:
        row = rows[camera]
        im = Image.open(row['source_rgb']).convert('RGB'); draw = ImageDraw.Draw(im)
        _, uv, edges = project_box(bounds, row['camera_row'])
        for i, j in edges:
            length = np.linalg.norm(uv[j]-uv[i]); direction = (uv[j]-uv[i])/length
            for off in np.arange(0, length, 24):
                draw.line([tuple(uv[i]+direction*off),
                    tuple(uv[i]+direction*min(off+13, length))], fill=(255,40,40), width=2)
        for i in [0, 2, 4, 6]:
            draw.text(tuple(uv[i]+[8,4]), 'LOWER '+str(i), font=font,
                fill=(0,220,255), stroke_width=2, stroke_fill='black')
        for (name, foot), color in zip(receipt['feet'].items(), colors):
            if camera in foot['regions']:
                rectangle = foot['regions'][camera]
                draw.rectangle(rectangle, outline=color, width=3)
                draw.text((rectangle[0],rectangle[3]+3), name, font=font,
                    fill=color, stroke_width=2, stroke_fill='black')
        draw.rectangle((0,0,1920,55), fill='black')
        draw.text((10,5), camera+' | SAME common002 bounds. RED EDGES HAVE NO DEPTH OCCLUSION.', font=font,fill='white')
        draw.text((10,29), 'Colored source ROIs -> independently bounded XYZ at right; lower corners labeled.', font=font,fill='white')
        canvas = Image.new('RGB',(2620,1012),'white'); canvas.paste(im,(0,0)); canvas.paste(plot,(1920,0))
        canvas.save(a.output/(camera+'_feet_depth_diagnostic.png'))
    contact = Image.new('RGB',(1200,1000),'#111111'); draw=ImageDraw.Draw(contact)
    for i, ((name, foot),color) in enumerate(zip(receipt['feet'].items(), colors)):
        for col, camera in enumerate(['DC','HC','KA']):
            draw.text((col*400+5,i*250+4), name+' '+camera, font=font, fill=color)
            if camera not in foot['regions']:
                draw.text((col*400+5,i*250+35),'OCCLUDED / AMBIGUOUS: UNUSED',font=font,fill='white')
                continue
            rect = foot['regions'][camera]
            im = Image.open(rows[camera]['source_rgb']).convert('RGB')
            ImageDraw.Draw(im).rectangle(rect,outline=color,width=2)
            x0,y0,x1,y1=rect; crop=im.crop((x0-12,y0-12,x1+12,y1+12))
            scale=min(380/crop.width,205/crop.height)
            crop=crop.resize((int(crop.width*scale),int(crop.height*scale)))
            contact.paste(crop,(col*400+5,i*250+32))
    contact.save(a.output/'whole_boot_ROI_manual_review.png')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
