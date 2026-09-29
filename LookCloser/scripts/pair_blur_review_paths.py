"""Pair two completed learned-RGB videos with exactly the same saved camera path."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from render_blur_review_path import encode
from blur_runtime import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--labels', nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folders = [args.left, args.right]
    paths = [json.loads((p / 'path.json').read_text()) for p in folders]
    if paths[0] != paths[1]:
        raise ValueError('Camera paths differ')
    receipts = [json.loads((p / 'complete.json').read_text()) for p in folders]
    if receipts[0]['frames'] != receipts[1]['frames']:
        raise ValueError('Frame counts differ')
    args.output.mkdir(parents=True, exist_ok=True)
    count = receipts[0]['frames']
    contact = []
    for index in range(count):
        pictures = [Image.open(p / f'frame_{index:03d}.png').convert('RGB') for p in folders]
        if pictures[0].size != pictures[1].size:
            raise ValueError('Frame dimensions differ')
        width, height = pictures[0].size
        frame = Image.new('RGB', (width * 2, height + 24), (25, 25, 25))
        draw = ImageDraw.Draw(frame)
        for column, (picture, label) in enumerate(zip(pictures, args.labels)):
            draw.text((column * width + 8, 6), label, fill='white')
            frame.paste(picture, (column * width, 24))
        frame.save(args.output / f'frame_{index:03d}.png')
        if index in [0, 6, 12, 18, count - 1]:
            frame.thumbnail((960, 300))
            contact.append(frame)
    sheet = Image.new('RGB', (max(p.width for p in contact), sum(p.height for p in contact)))
    y = 0
    for frame in contact:
        sheet.paste(frame, (0, y))
        y += frame.height
    sheet.save(args.output / 'contact.jpg', quality=92)
    metadata = dict(frames=count, camera_paths_equal=True,
                    sources=[dict(folder=str(p), receipt_sha256=sha(p / 'complete.json'),
                                  checkpoint_sha256=m['checkpoint_sha256'], step=m['step'])
                             for p, m in zip(folders, receipts)],
                    labels=args.labels, render='paired learned RGB; labels only added',
                    finite_rgb_checked=all(m['finite_rgb_checked'] for m in receipts))
    write(args.output / 'path.json', paths[0])
    write(args.output / 'render_complete.json', metadata)
    encode(args.output, metadata)


if __name__ == '__main__':
    main()
