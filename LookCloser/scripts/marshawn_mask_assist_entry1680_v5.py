"""1680-only producer entry retaining the original V5 native/MAT operations."""
from pathlib import Path
import argparse
import marshawn_sam3_assist as io
import marshawn_mask_assist_entry_v5 as original
from marshawn_mask_assist_canary1680_v5 import guard


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign', type=Path, required=True); p.add_argument('--frame', type=int, required=True)
    a = p.parse_args(); V = a.campaign.resolve(); guard(V, a.frame)
    actual_read = io.read
    source = V / 'operations/sam3_assist/v5/entry_release.json'
    versioned = V / 'operations/sam3_assist/v5/entry1680_release.json'

    def versioned_release(path):
        return actual_read(versioned if Path(path).resolve() == source else path)
    io.read = versioned_release
    try:
        original.run(V, a.frame)
    finally:
        io.read = actual_read


if __name__ == '__main__':
    main()
