"""Bind independent review evidence without modifying any source study output."""
from pathlib import Path
import numpy as np
from study_multiview_face_prior import read,save,sha
from review_mhr_conformance_independent import ROOT,SOURCE

def main():
    r=read(ROOT/'result.json');s=read(ROOT/'strict_crossings.json');checked=[]
    def check(path,digest):assert sha(path)==digest,str(path);checked.append(str(path))
    check(SOURCE/'protocol.json',r['source_protocol_sha256'])
    for arm,digest in r['source_fit_hashes'].items():check(SOURCE/arm/'fit.npz',digest)
    p=read(SOURCE/'protocol.json');check(p['original_mesh'],r['original_mesh_sha256'])
    check(Path(__file__).with_name('review_mhr_conformance_independent.py'),r['script_sha256']);check(Path(__file__).with_name('check_mhr_conformance_crossings.py'),s['script_sha256']);check(ROOT/'result.json',s['review_result_sha256'])
    for item in r['files']:check(item['path'],item['sha256'])
    assert r['train_only_replay_max_vertex_difference']==0;assert np.load(ROOT/'train_only_replay.npz')['difference'].max()==0
    assert all(x['strict_pairs_touching_y135_153']==0 for x in s['records'])
    reviewed=['smooth025_G004_B005_1210FG.png','smooth100_G004_B005_1210FG.png','smooth100_M004_B005_12109O.png','smooth400_G004_B005_1210FG.png','smooth400_M004_B005_12109O.png','smooth400_E004_B005_1210I7.png']
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_conformance_independent_review.md'
    save(ROOT/'seal.json',dict(status='passed',checked_bindings=len(checked),checked_paths=checked,reviewed_native_files={x:sha(ROOT/x) for x in reviewed},report_sha256=sha(report),script_sha256=sha(__file__),
        inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='seal.json'},source_outputs_changed=False,production_approval=False))
    print('Independent review sealed;',len(checked),'bindings;',len(reviewed),'native panels')

if __name__=='__main__':main()
