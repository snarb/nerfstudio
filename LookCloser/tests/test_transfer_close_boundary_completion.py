import inspect
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from transfer_close_boundary_completion import optional_override
import guard_poisson_jaw_completion as original


def test_optional_override_only_nests_existing_override_body():
    code=inspect.getsource(original.prepare)
    updated=optional_override(code)
    compile(updated,'<test>','exec')
    start=updated.index("    if 'mask_override' in base:\n")
    end=updated.index("    evidence=root/'admission'")
    body=updated[start:end].splitlines(keepends=True)[1:]
    restored=updated[:start]+''.join(line[4:] for line in body)+updated[end:]
    assert restored==code


def test_drifted_implementation_fails_closed():
    with pytest.raises(ValueError):
        optional_override('def prepare(root):\n    pass\n')
