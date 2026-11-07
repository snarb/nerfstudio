"""Read-only TRAIN temporal source contact sheets; no background labels/plates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(campaign, sources, output):
    output.mkdir(parents=True, exist_ok=False)
    transforms = json.loads((campaign / 'data/transforms.json').read_text())
    by_camera = {f['physical_camera']: f for f in transforms['frames']
                 if f['file_path'] in transforms['train_filenames']}
    records = []
    for camera in ['GA', 'HE', 'MC', 'NA']:
        row = by_camera[camera]  # EVAL cameras are deliberately ineligible.
        current = campaign / 'data' / row['file_path']
        with np.load(campaign / 'data/semantics' / (camera + '.npz')) as masks:
            union = masks['black'] | masks['grey']
        yy, xx = np.nonzero(union)
        bounds = (max(0, int(xx.min())-70), max(0, int(yy.min())-35),
                  min(row['w'], int(xx.max())+71), min(row['h'], int(yy.max())+71))
        sheets = Image.new('RGB', (5*400, 810), '#171717')
        draw = ImageDraw.Draw(sheets)
        draw.text((8, 4), camera + ' TRAIN: fixed crop, original camera/colour; temporal images are NOT background labels', fill='white')
        camera_rows = []
        for col, frame in enumerate([1660, 1720, 1780, 1840, 1900]):
            receipt_path = sources / str(frame) / camera / 'receipt.json'
            receipt = json.loads(receipt_path.read_text())
            assert receipt['camera'] == camera and receipt['frame'] == frame
            source_image = Path(receipt['image'])
            assert sha(source_image) == receipt['image_sha256']
            image = Image.open(source_image).convert('RGB')
            assert image.size == (row['w'], row['h'])
            difference = None
            if frame == 1660:
                difference = int(np.abs(np.asarray(image).astype(int)
                                        - np.asarray(Image.open(current).convert('RGB')).astype(int)).max())
            crop = image.crop(bounds)
            crop.thumbnail((390, 740))
            sheets.paste(crop, (col*400 + 5, 55))
            draw.text((col*400 + 5, 27), f'{frame}  {(frame-1660)/120:.1f}s', fill='white')
            camera_rows.append(dict(frame=frame, receipt_sha256=sha(receipt_path),
                                    image=str(source_image), image_sha256=sha(source_image),
                                    frozen_EV=receipt['fixed_EV'],
                                    transforms_sha256=receipt['transforms_sha256'],
                                    photometry_sha256=receipt['photometry_sha256'],
                                    current_frame_RGB_max_difference=difference))
        assert len({r['frozen_EV'] for r in camera_rows}) == 1
        for key in ['transforms_sha256', 'photometry_sha256']:
            assert len({r[key] for r in camera_rows}) == 1
        page = output / (camera + '_temporal_sources.jpg')
        sheets.save(page, quality=95)
        records.append(dict(camera=camera, split='TRAIN', crop_xyxy=bounds,
                            current_image_sha256=sha(current), frames=camera_rows,
                            page=str(page), page_sha256=sha(page)))
    result = dict(kind='TRAIN_TEMPORAL_BACKGROUND_FEASIBILITY_CONTACTS_ONLY',
                  records=records, helper_sha256=sha(__file__),
                  new_training_labels=False, background_plate_created=False,
                  geometry_changed=False, optimizer_updates=0,
                  limitation='Temporal actors, crew, shadows and moving equipment prevent interpreting a median or another frame as known background without visibility qualification.')
    (output / 'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cameras=len(records), frames_per_camera=5,
                          current_RGB_max_differences={r['camera']: r['frames'][0]['current_frame_RGB_max_difference'] for r in records})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['campaign', 'sources', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    main(args.campaign, args.sources, args.output)
