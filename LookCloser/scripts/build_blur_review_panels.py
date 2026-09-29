"""Build labeled review sheets from saved GT/RGB crop pairs; never re-render."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step', type=int, help='Otherwise use each run\'s selected checkpoint')
    parser.add_argument('--regions', nargs='+', default=['face', 'hair', 'lipstick'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in args.runs:
        folder = args.root / name
        selection = json.loads((folder / 'selection.json').read_text())
        step = args.step or int(Path(selection['path']).stem.split('_')[-1])
        request_path = folder / 'request.json'
        if not request_path.exists():
            request_path = Path(__file__).resolve().parents[1] / 'experiments/assets/blur_ablation_fresh' / f'{name}.json'
        request = json.loads(request_path.read_text())
        if Path(request['output']).resolve() != folder.resolve():
            raise ValueError(f'Request output does not match the reviewed run: {request_path}')
        sources.append((name, step, folder / f'eval_{step:06d}', request))
    manifest = []
    for split in ['train', 'eval']:
        indices = sources[0][3].get('train_review_indices', [0]) if split == 'train' else [0, 1, 2]
        for region in args.regions:
            available = [index for index in indices if any(
                (folder / f'{split}_{index:03d}_{region}.png').exists()
                for _, _, folder, _ in sources)]
            if not available:
                continue
            width, height = 400, 240
            sheet = Image.new('RGB', (width * len(sources), height * len(available)), (30, 30, 30))
            draw = ImageDraw.Draw(sheet)
            for column, (name, step, folder, _) in enumerate(sources):
                for row, index in enumerate(available):
                    path = folder / f'{split}_{index:03d}_{region}.png'
                    with Image.open(path) as raw:
                        picture = raw.convert('RGB')
                        picture.thumbnail((width - 8, height - 42))
                    x, y = column * width + 4, row * height
                    draw.text((x, y + 3), f'{name} / {step}', fill='white')
                    draw.text((x, y + 17), f'{split}{index}: {region} / GT | RGB', fill='white')
                    sheet.paste(picture, (x, y + 38))
                    manifest.append(dict(source=str(path), run=name, step=step,
                                         split=split, index=index, region=region))
            sheet.save(args.output / f'{split}_{region}.jpg', quality=92)
    (args.output / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
