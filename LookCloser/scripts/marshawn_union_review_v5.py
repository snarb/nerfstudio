"""Source photographs with exact current-V5 union additions, for manual review."""
from pathlib import Path
import argparse, json
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np
from PIL import Image, ImageDraw
import marshawn_sam3_assist as io


def run(V, frame):
    root = V / 'operations/sam3_assist/v5' / ('frame%06d' % frame)
    ready = io.read(root / 'all66_controller/all66_review_ready.json')
    frame_root = root / 'isolated_campaign/frames' / ('%06d' % frame)
    output = root / 'source_review'; output.mkdir(exist_ok=True)
    cv2.setNumThreads(1)

    def one(camera):
        path = frame_root / 'native_masks' / (camera + '.npz')
        with np.load(path, allow_pickle=False) as z:
            actor = z['black'] | z['grey']; new = z['actor_ownership_uncertain']
        source = io.read(V / 'sources' / str(frame) / camera / 'receipt.json')
        image_path = Path(source['raw_native_image'])
        if io.sha(image_path) != source['raw_native_image_sha256']:
            raise ValueError('Review source changed')
        image = np.asarray(Image.open(image_path).convert('RGB')).copy()
        yy, xx = np.where(actor)
        box = [max(0, int(xx.min()) - 192), max(0, int(yy.min()) - 192),
               min(8192, int(xx.max()) + 193), min(4320, int(yy.max()) + 193)] if len(xx) else [0, 0, 8192, 4320]
        contour = cv2.morphologyEx(actor.astype('uint8'), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(bool)
        image[contour] = [0, 255, 0]
        image[new] = (image[new].astype(np.float32) * .2 + np.array([0, 220, 255]) * .8).astype('uint8')
        crop = Image.fromarray(image[box[1]:box[3], box[0]:box[2]])
        crop.thumbnail((520, 380))
        tile = Image.new('RGB', (540, 420), (18, 18, 18)); tile.paste(crop, ((540-crop.width)//2, 32))
        ImageDraw.Draw(tile).text((10, 10), '%s  restored=%d  green=union; cyan=restored' % (camera, int(new.sum())), fill='white')
        destination = output / (camera + '.jpg'); tile.save(destination, quality=94)
        return dict(camera=camera, restored_pixels=int(new.sum()), source_sha256=source['raw_native_image_sha256'],
                    native_archive_sha256=io.sha(path), overlay=str(destination), overlay_sha256=io.sha(destination), native_crop=box)

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(one, sorted(ready['cameras'])))
    records.sort(key=lambda row: (-row['restored_pixels'], row['camera']))
    pages = []
    for index in range(0, len(records), 12):
        rows = records[index:index+12]; sheet = Image.new('RGB', (2160, 1260), (0, 0, 0))
        for i, row in enumerate(rows):
            sheet.paste(Image.open(row['overlay']), ((i % 4)*540, (i//4)*420))
        path = output / ('contact_%02d.jpg' % (index//12)); sheet.save(path, quality=93); pages.append(str(path))
    io.write(output / 'manifest.json', dict(frame=frame, records=records, contact_pages=pages,
             artifact='Actual source photograph with measured masks only; no generated scene content',
             all66_ready_sha256=io.sha(root / 'all66_controller/all66_review_ready.json'),
             all_source_review_still_required=True))
    print(json.dumps(dict(pages=pages, cameras=len(records))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign', type=Path, required=True); p.add_argument('--frame', type=int, required=True)
    a = p.parse_args(); run(a.campaign.resolve(), a.frame)
