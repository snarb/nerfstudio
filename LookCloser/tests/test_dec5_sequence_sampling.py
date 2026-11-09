from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_subject_sampling import SubjectSampler


def make_sampler(tmp_path,weighted):
    images=tmp_path/'images';images.mkdir(exist_ok=True)
    maps=tmp_path/'lookcloser_frequencies';maps.mkdir(exist_ok=True)
    semantics=tmp_path/'semantics';semantics.mkdir(exist_ok=True)
    filenames=[]
    for i in range(2):
        name=f'C{i}';path=images/f'frame_train_{name}.png';path.touch();filenames.append(path)
        values=np.zeros((8,12,3),np.uint16);values[...,0]=65535
        np.savez_compressed(semantics/(name+'.npz'),probability=values)
        frequency=torch.full((2,3),16.);frequency[:,-1]=8192;torch.save(frequency,maps/(path.stem+'.pt'))
    dataset=SimpleNamespace(image_filenames=filenames,cameras=SimpleNamespace(height=[16,16],width=[24,24]))
    sampler=SimpleNamespace(sample=lambda *a,**k:None)
    return SubjectSampler(sampler,dataset,semantics,.6,.3,expected_train_count=2,frequency_weighted_hair=weighted)


def test_semantic_raster_sampling_preserves_native_coordinates(tmp_path):
    sampler=make_sampler(tmp_path,False);torch.manual_seed(42);rays=sampler.draw('hair',12000,[0,1])
    assert (rays[:,1]>=0).all() and (rays[:,1]<16).all()
    assert (rays[:,2]>=0).all() and (rays[:,2]<24).all()
    assert set(rays[:,1].tolist())==set(range(16))
    assert set(rays[:,2].tolist())==set(range(24))
    assert .30<float((rays[:,2]>=16).float().mean())<.37


def test_frequency_weighting_targets_the_correct_native_camera_cells(tmp_path):
    sampler=make_sampler(tmp_path,True);torch.manual_seed(42);rays=sampler.draw('hair',12000,[0,1])
    # Last third has 3x weight: 3/(1+1+3), independent of semantic downscale.
    assert .57<float((rays[:,2]>=16).float().mean())<.63
    assert set(rays[:,0].tolist())=={0,1}
