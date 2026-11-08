"""CPU-only native metadata collector; no Torch import or checkpoint loading."""
from pathlib import Path
import argparse
import json
from policy import read, sha


def collect(root, frame):
    directory = Path(root) / 'frames' / ('%06d' % int(frame))
    paths = ['frame_completion.json', 'train/result.json', 'data/ready.json',
             'data/transforms.json', 'train/cold_provenance.json']
    for optional in ['train/cold_setup_audit.json', 'dispatch_ready.json']:
        if (directory / optional).exists():
            paths.append(optional)
    return dict(frame=int(frame), native_campaign=str(root),
                inputs_sha256={name: sha(directory / name) for name in paths},
                completion=read(directory / 'frame_completion.json'),
                result=read(directory / 'train/result.json'),
                dispatch=read(directory / 'dispatch_ready.json')
                if (directory / 'dispatch_ready.json').exists() else None,
                cold_audit=read(directory / 'train/cold_setup_audit.json')
                if (directory / 'train/cold_setup_audit.json').exists() else None,
                code_seal_sha256=sha(Path(root) / 'code_seal.json'),
                recipe_sha256=sha(Path(root) / 'recipe.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--frame', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.frame), separators=(',', ':')))
