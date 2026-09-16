"""Exact adapters of the independent RGB/ray audit for face-angular policies."""
import argparse
import hashlib
import inspect
from pathlib import Path
import audit_face_visibility_recovery as audit
from study_multiview_face_prior import read,save,sha


def main(roots):
    original=Path(audit.__file__).resolve();source=inspect.getsource(audit.main)
    replacements={
        "weights=quality((direction*normal).sum(-1),length,'incidence2')*angles[:,None]":"weights=np.broadcast_to(angles[:,None],length.shape)",
        "assert valid[old,j].all() and (~valid[new,j]).all() and skin[new,j].all() and (votes>=3).all()":
        "assert valid[old,j].all() and skin[new,j].all() and (votes>=3).all()\n    if request['admission_mode']=='raster':assert valid[new,j].all()",
    }
    for old,new in replacements.items():
        assert source.count(old)==1;source=source.replace(old,new)
    proof=dict(base_audit_sha256=sha(original),adapter_sha256=sha(__file__),generated_sha256=hashlib.sha256(source.encode()).hexdigest())
    audit.__dict__['__file__']=__file__;exec(compile(source,__file__+':angular_audit','exec'),audit.__dict__)
    for root in roots:
        audit.main(root);save(root/'audit_adapter.json',proof)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+');a=p.parse_args()
    audit.torch.set_num_threads(2)
    with audit.torch.inference_mode():main(a.roots)
