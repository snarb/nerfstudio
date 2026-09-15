"""Train-only native wrist/hand observations before local surface repair."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import ROOT, cameras, read, sha, atomic_json, display, exr, HELD_CAMERAS

OUTPUT = Path('/mnt/data/dec5_wrist_observations')
NAMES = ['G004_A005_121071', 'G004_B005_1210FG', 'G004_C005_121037',
         'H004_A005_1210M6', 'H004_B005_1210EL', 'H004_C005_1210SZ']


def stage(output, frame):
    folder = output / frame
    folder.mkdir(parents=True, exist_ok=False)
    rows, _, _ = cameras(frame)
    by_name = {r['physical_camera']: r for r in rows}
    profile = read(ROOT / 'camera_profiles.json')
    gains = dict(zip(profile['physical_cameras'], profile['rgb_gain']))
    exposure = read(ROOT / 'exposure.json')['fixed_exposure_gain']
    request = dict(frame=frame, camera_names=NAMES, heldout_used=False,
                   profiles_sha256=sha(ROOT / 'camera_profiles.json'), exposure_sha256=sha(ROOT / 'exposure.json'),
                   fixed_camera_gains_across_time=True, script_sha256=sha(__file__))
    atomic_json(folder / 'request.json', request)

    def load(name):
        if name in HELD_CAMERAS: raise ValueError('Held-out source')
        row = by_name[name]
        rgb = np.rint(255 * display(exr(row['file_path']) * np.array(gains[name]), exposure)).clip(0, 255).astype(np.uint8)
        path = folder / (name + '.png')
        im = Image.fromarray(np.rot90(rgb))
        im.save(path)
        crop = im.crop((0, 1250, 500, 1920))
        crop.save(folder / (name + '_hand_native.png'))
        return dict(camera=row, source_sha256=sha(row['file_path']), image=str(path), image_sha256=sha(path)), crop

    with ThreadPoolExecutor(max_workers=3) as pool:
        loaded = list(pool.map(load, NAMES))
    panel = Image.new('RGB', (1500, 1400))
    draw = ImageDraw.Draw(panel)
    for i, ((record, crop), name) in enumerate(zip(loaded, NAMES)):
        x, y = (i % 3) * 500, (i // 3) * 700
        panel.paste(crop, (x, y + 30)); draw.text((x + 4, y + 5), name, fill='white')
    panel.save(folder / 'six_train_views_native.png')
    atomic_json(folder / 'result.json', dict(request_sha256=sha(folder / 'request.json'),
        records=[r for r, _ in loaded], panel_sha256=sha(folder / 'six_train_views_native.png'),
        scope='source observations, not candidate renders or quality metrics', visual_status='pending'))
    print(frame, 'six native train views saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--frame', required=True)
    args = parser.parse_args(); stage(args.output, args.frame)
