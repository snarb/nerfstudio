"""Bind each cold frame to one registered recipe, preserving legacy recovery."""
from pathlib import Path
from common import read, atomic, sha
from audit_exports import approved_recipes


def bind_recipe(root, frame_root, settings, *, legacy=False):
    root, frame_root = Path(root), Path(frame_root)
    recipes = approved_recipes(root)
    path = frame_root / 'recipe_binding.json'
    if path.exists():
        binding = read(path)
        digest = binding['recipe_sha256']
        if digest not in recipes or binding['recipe'] != recipes[digest]:
            raise ValueError('Immutable frame recipe binding changed')
        return recipes[digest], digest
    if legacy or (frame_root / 'train/spec.json').exists():
        source = root / 'selected_recipe.json'
    else:
        source = root / settings.get('production_recipe_path', 'selected_recipe.json')
    digest = sha(source)
    if digest not in recipes or read(source) != recipes[digest]:
        raise ValueError('Active recipe source is not approved')
    atomic(path, dict(recipe_sha256=digest, recipe=recipes[digest]))
    return recipes[digest], digest
