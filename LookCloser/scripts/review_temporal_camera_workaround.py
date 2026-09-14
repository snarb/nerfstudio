"""Native jaw crops for every time in an already-rendered camera workaround.

This is a review companion, not image editing, rerendering, or a quality metric.
The crop follows the declared fixed-landmark projection, never a prediction mask.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from joint_temporal_texture import atomic_json, read, sha


def crop_box(camera):
    x, y = (round(v) for v in camera['scene_landmark_portrait_xy'])
    return (x - 250, y + 25, x + 250, y + 325)


def prepare(source, output):
    request = read(source / 'request.json')
    ids = request['ordered_frame_ids']
    if len(ids) != len(set(ids)) or ids != [r['frame_id'] for r in request['inventory']]:
        raise ValueError('Unordered or duplicated frame inventory')
    inputs = []
    request_hash = sha(source / 'request.json')
    for record in request['inventory']:
        folder = source / 'frames' / record['frame_id']
        receipt = read(folder / 'complete.json')
        digest = sha(folder / 'frame.png')
        if receipt['request_sha256'] != request_hash or receipt['hashes']['frame.png'] != digest:
            raise ValueError('Unverified render')
        inputs.append(dict(frame_id=record['frame_id'], path=str(folder / 'frame.png'),
                           sha256=digest, crop=crop_box(record['camera'])))
    spec = dict(source_request=str(source / 'request.json'), source_request_sha256=request_hash,
                inputs=inputs, script_sha256=sha(__file__), native_pixels=True,
                metrics_computed=False, video_changed=False)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'request.json').exists() and read(output / 'request.json') != spec:
        raise ValueError('Review request mismatch')
    atomic_json(output / 'request.json', spec)
    sheets = []
    for first in range(0, len(inputs), 6):
        panel = Image.new('RGB', (1000, 3 * 326), (20, 20, 20))
        draw = ImageDraw.Draw(panel)
        batch = inputs[first:first + 6]
        for j, item in enumerate(batch):
            with Image.open(item['path']) as rgb:
                box = item['crop']
                if box[0] < 0 or box[1] < 0 or box[2] > rgb.width or box[3] > rgb.height:
                    raise ValueError('Crop extends beyond original render')
                crop = rgb.convert('RGB').crop(box)
            x, y = j % 2 * 500, j // 2 * 326
            panel.paste(crop, (x, y + 26))
            draw.text((x + 4, y + 5), f"{item['frame_id']}  original pixels / jaw review", fill='white')
        path = output / f'jaw_{first:03d}.png'
        panel.save(path)
        sheets.append(dict(path=str(path), sha256=sha(path), frame_ids=[r['frame_id'] for r in batch]))
    atomic_json(output / 'sheets.json', dict(request_sha256=sha(output / 'request.json'),
                sheets=sheets, visual_status='pending', scope='jaw only; not whole-actor acceptance'))
    print(f'Prepared {len(sheets)} native sheets, {len(inputs)} distinct times', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
