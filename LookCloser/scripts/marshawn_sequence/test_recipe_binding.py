"""Recipe promotion must not silently rebind existing frame recovery."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common import atomic, read, sha
from recipe_binding import bind_recipe
from audit_exports import approved_recipes


class RecipeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.old=dict(parameters=dict(coarse_step=.001),steps=150000,selected_inference_cap=10)
        self.new=copy.deepcopy(self.old);self.new['parameters']['admission_background_votes']=3
        atomic(self.root/'selected_recipe.json',self.old);atomic(self.root/'recipes/votes3.json',self.new)
        self.old_sha=sha(self.root/'selected_recipe.json');self.new_sha=sha(self.root/'recipes/votes3.json')
        atomic(self.root/'approved_recipes.json',dict(recipes=[dict(path='recipes/votes3.json',sha256=self.new_sha)]))
        self.settings=dict(production_recipe_path='recipes/votes3.json')

    def test_cold_frame_uses_registered_variant(self):
        recipe,digest=bind_recipe(self.root,self.root/'frames/000923',self.settings)
        self.assertEqual((recipe,digest),(self.new,self.new_sha))
        self.assertEqual(read(self.root/'selected_recipe.json'),self.old)

    def test_existing_spec_keeps_legacy(self):
        frame=self.root/'frames/000923';atomic(frame/'train/spec.json',dict(existing=True))
        self.assertEqual(bind_recipe(self.root,frame,self.settings),(self.old,self.old_sha))

    def test_active_preparation_can_be_explicitly_pinned_legacy(self):
        frame=self.root/'frames/000923'
        self.assertEqual(bind_recipe(self.root,frame,self.settings,legacy=True),(self.old,self.old_sha))
        self.assertEqual(bind_recipe(self.root,frame,self.settings),(self.old,self.old_sha))

    def test_switching_active_pointer_does_not_rebind_frame(self):
        frame=self.root/'frames/000923';bind_recipe(self.root,frame,self.settings)
        self.assertEqual(bind_recipe(self.root,frame,{}),(self.new,self.new_sha))

    def test_unregistered_active_recipe_is_rejected(self):
        atomic(self.root/'rogue.json',dict(parameters={}))
        with self.assertRaises(ValueError):bind_recipe(self.root,self.root/'frames/000923',dict(production_recipe_path='rogue.json'))

    def test_modified_approval_source_is_rejected(self):
        atomic(self.root/'recipes/votes3.json',dict(tampered=True))
        with self.assertRaises(ValueError):approved_recipes(self.root)

    def test_modified_frame_binding_is_rejected(self):
        frame=self.root/'frames/000923';bind_recipe(self.root,frame,self.settings)
        b=read(frame/'recipe_binding.json');b['recipe']['steps']=999;atomic(frame/'recipe_binding.json',b)
        with self.assertRaises(ValueError):bind_recipe(self.root,frame,self.settings)

    def test_registry_path_cannot_escape_campaign(self):
        atomic(self.root/'approved_recipes.json',dict(recipes=[dict(path='../outside.json',sha256='x')]))
        with self.assertRaises(ValueError):approved_recipes(self.root)


if __name__=='__main__':unittest.main()
