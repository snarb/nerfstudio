"""Explicit bounded-gain control for an existing additive color proposal.

Caps each channel to 0.5..2 times its native display value; preserves every
previously nonzero channel after RGB8 quantization. No spatial/source averaging.
This does not claim unchanged source gradients or calibrated radiance.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json


def bounded_rgb(before, offset, support, maximum_ratio=2.):
    if maximum_ratio < 1 or not np.isfinite(maximum_ratio):
        raise ValueError('Invalid fixed gain bound')
    if before.dtype != np.uint8 or offset.shape != before.shape or support.shape != before.shape[:2] or not np.isfinite(offset).all():
        raise ValueError('Invalid finite RGB/offset/support')
    value = before.astype(float)/255
    proposed = value + offset
    bounded = np.clip(proposed, value/maximum_ratio, np.minimum(value*maximum_ratio, 1))
    quantized = np.rint(bounded*255).clip(0,255).astype(np.uint8)
    quantized[(before>0)&(quantized==0)] = 1
    return np.where(support[...,None], quantized, before)


def run(source, output):
    if output.exists():
        raise ValueError('Preserve existing bounded control')
    original = read(source/'result.json')
    for name,digest in original['hashes'].items():
        if sha(source/name)!=digest:
            raise ValueError('Changed additive proposal')
    before=np.asarray(Image.open(source/'baseline.png').convert('RGB'))
    data=np.load(source/'offset.npz'); offset=data['offset'].transpose(1,2,0)
    support=(data['depth']>0)&(data['selection']>=0)
    after=bounded_rgb(before,offset,support)
    output.mkdir(parents=True)
    Image.fromarray(after).save(output/'corrected.png')
    panel=Image.new('RGB',(before.shape[1]*2,before.shape[0]+25));draw=ImageDraw.Draw(panel)
    for i,(label,rgb) in enumerate([('published hard source',before),('same labels + bounded gradient color',after)]):
        panel.paste(Image.fromarray(rgb),(i*before.shape[1],25));draw.text((i*before.shape[1]+3,4),label,fill='white')
    panel.save(output/'comparison.png')
    atomic_json(output/'result.json',dict(source_result=str(source/'result.json'),source_result_sha256=sha(source/'result.json'),
        source_offset_sha256=sha(source/'offset.npz'),baseline_sha256=sha(source/'baseline.png'),script_sha256=sha(__file__),
        source_labels_unchanged=True,geometry_unchanged=True,unsupported_rgb_unchanged=bool(np.array_equal(before[~support],after[~support])),
        no_spatial_or_source_rgb_averaging=True,maximum_display_channel_ratio=2., minimum_nonzero_rgb8_channel=1,
        newly_black_supported_pixels=int((support&(before.max(-1)>0)&(after.max(-1)==0)).sum()),
        visual_status='pending',image_quality_metrics_computed=False,
        hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
