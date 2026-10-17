"""Recursively verify receipt dependencies without touching prediction artifacts."""
from pathlib import Path
from luster_hybrid_common import read, sha


def verify_dependencies(receipt_path, inventory, bind, deep=True, visited=None):
    """Follow both cache inputs and candidate parents, including transitive ones.

    Every declared digest is verified, even if another parent already visited
    the same file. Recursion is memoized only after successful verification.
    """
    visited = set() if visited is None else visited
    receipt_path = Path(receipt_path).resolve()
    if receipt_path in visited:
        return
    receipt = read(receipt_path)
    if 'code_sha256' in receipt:
        assert receipt['code_sha256'] in inventory.values(), f'Changed generator: {receipt_path}'
    bind(receipt_path)
    # Mark before traversal so even malformed cyclic declarations terminate.
    visited.add(receipt_path)
    for key in ('inputs', 'additional_input_hashes'):
        for filename, expected in receipt.get(key, {}).items():
            path = Path(filename).resolve()
            assert sha(path) == expected, f'Changed dependency: {path}'
            bind(path)
            if path.name.endswith('receipt.json'):
                verify_dependencies(path, inventory, bind, deep, visited)
    if deep:
        for filename, expected in receipt.get('hashes', {}).items():
            path = receipt_path.parent / filename
            if not path.exists():
                path = path.with_suffix('.npy')
            assert sha(path) == expected, f'Changed cached array: {path}'
            bind(path)
    # Older source-view warp receipts predate the common hashes dictionary.
    if deep and 'rgb_sha256' in receipt:
        path = receipt_path.parent / 'rgb.npy'
        assert sha(path) == receipt['rgb_sha256'], f'Changed source model warp: {path}'
        bind(path)
