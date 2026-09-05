#!/usr/bin/env python3
"""Render fixed-calibration novel cameras from a TSDF mesh and train RGB only.

Normalize cameras once using the mesh receipt, then run the existing hard texture
renderer with normalization disabled. Target RGB is never read. Path selection
uses calibration only, including real held-out camera positions as review anchors.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from colmap_patchmatch_tsdf_campaign_common import atomic_json, canonical_sha256, sha256

SCRIPTS = Path(__file__).resolve().parent
ANCHORS = ("F004_B005_1210O9", "J004_D005_1210TA", "L004_B005_12106A")


def normalize_frame(frame: dict, payload: dict, mesh_metadata: dict) -> dict:
    applied = np.eye(4)
    if "applied_transform" in payload:
        applied[:3] = np.asarray(payload["applied_transform"])
    transform = np.eye(4)
    transform[:3] = np.asarray(mesh_metadata["dataparser_transform"])
    matrix = transform @ np.linalg.inv(applied) @ np.asarray(frame["transform_matrix"])
    matrix[:3,3] *= float(mesh_metadata["dataparser_scale"])
    result = deepcopy(frame)
    result["transform_matrix"] = matrix.tolist()
    result.pop("depth_file_path", None)
    if any(abs(float(result.get(k,0))) > 1e-12 for k in ("k1","k2","p1","p2")):
        raise ValueError("Path renderer requires undistorted pinhole RGB")
    return result


def calibration_path(calibration: dict, anchors: list[str], samples_per_segment: int) -> list[dict]:
    by_physical = {f["physical_camera"]:f for f in calibration["frames"]}
    keyframes = [by_physical[name] for name in anchors]
    frames = []
    for segment in range(len(keyframes)-1):
        left,right = keyframes[segment:segment+2]
        a,b = np.asarray(left["transform_matrix"]),np.asarray(right["transform_matrix"])
        rotations = Slerp([0,1],Rotation.from_matrix(np.stack([a[:3,:3],b[:3,:3]])))
        for t in np.linspace(0,1,samples_per_segment,endpoint=False):
            frame = deepcopy(left)
            mat = np.eye(4)
            mat[:3,:3] = rotations(float(t)).as_matrix()
            mat[:3,3] = (1-t)*a[:3,3]+t*b[:3,3]
            frame["transform_matrix"] = mat.tolist()
            for k in ("fl_x","fl_y","cx","cy"):
                frame[k] = float((1-t)*left[k]+t*right[k])
            frame["physical_camera"] = left["physical_camera"] if t == 0 else f"synthetic_{segment}_{t:.6f}"
            frame["path_segment"],frame["path_t"] = segment,float(t)
            frames.append(frame)
    frames.append(deepcopy(keyframes[-1]))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--mesh",type=Path,required=True)
    parser.add_argument("--mesh-metadata",type=Path,required=True)
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--anchors",nargs="+",default=list(ANCHORS))
    parser.add_argument("--samples-per-segment",type=int,default=4)
    parser.add_argument("--neighbors",type=int,default=16)
    parser.add_argument("--aggregation-mode",choices=("nearest-fill","seam-cut"),default="nearest-fill")
    parser.add_argument("--seam-cut-rank-penalty",type=float,default=.0001)
    parser.add_argument("--seam-cut-bandwidth-penalty",type=float,default=0.)
    parser.add_argument("--primary-angular-camera-count",type=int,default=0)
    parser.add_argument("--camera-color-calibration",type=Path,default=None)
    parser.add_argument("--angular-surface-color",type=Path,default=None)
    parser.add_argument("--hard-source-seam-leveling",action="store_true")
    parser.add_argument("--surface-texture-registration",action="store_true")
    parser.add_argument("--camera-color-model",choices=("ingest","exposure","rgb","spatial"),default="rgb")
    parser.add_argument("--overlap-exposure-grid",type=int,nargs=2,default=None)
    parser.add_argument("--pixel-center-offset",type=float,choices=(0.,.5),default=0.)
    parser.add_argument("--exact-mesh-visibility",action="store_true")
    parser.add_argument("--source-rgb-footprint-visibility",action="store_true")
    parser.add_argument("--source-rgb-depth-aware-sampling",action="store_true")
    parser.add_argument('--source-observed-free-space-veto',action='store_true')
    parser.add_argument('--source-observed-depth-data',type=Path,default=None)
    parser.add_argument('--source-observed-mesh-metadata',type=Path,default=None)
    parser.add_argument("--disocclusion-color-match",action="store_true")
    parser.add_argument("--seam-cut-visibility-radius",type=float,default=0.)
    parser.add_argument("--surface-color-field-smoothness",type=float,default=0.)
    parser.add_argument("--depth-log-tolerance",type=float,default=.01)
    parser.add_argument("--depth-hole-fill-max-area",type=int,default=0)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if args.source_observed_free_space_veto and (args.source_observed_depth_data is None or args.source_observed_mesh_metadata is None):
        parser.error('Free-space veto requires raw depth data and matching mesh metadata')
    if len(args.anchors)<2 or args.samples_per_segment<1:
        parser.error("At least two anchors and one sample per segment required")
    payload = json.loads((args.data/"transforms.json").read_text())
    calibration = json.loads(args.calibration.read_text())
    metadata = json.loads(args.mesh_metadata.read_text())
    train_names = set(payload["train_filenames"])
    source_frames = [normalize_frame(f,payload,metadata) for f in payload["frames"] if f["file_path"] in train_names]
    if any(f.get("mask_path") for f in payload["frames"]):
        raise ValueError("Image/person masks forbidden")
    if set(args.anchors) & {f["physical_camera"] for f in source_frames}:
        raise ValueError("Review anchors must remain held out of source RGB")
    for f in source_frames:
        f["file_path"] = str((args.data/f["file_path"]).resolve(strict=True))
    targets = [normalize_frame(f,calibration,metadata) for f in calibration_path(calibration,args.anchors,args.samples_per_segment)]
    args.output.mkdir(parents=True,exist_ok=args.resume)
    raw_depth_hashes=None
    if args.source_observed_free_space_veto:
        from carve_patchmatch_mesh_free_space import train_frames

        raw_payload=json.loads((args.source_observed_depth_data/'transforms.json').read_text())
        raw_depth_hashes={f['physical_camera']:sha256(args.source_observed_depth_data/f['depth_file_path']) for f in train_frames(raw_payload)}
    request = {"schema_version":1,"mesh_sha256":sha256(args.mesh),"mesh_metadata_sha256":sha256(args.mesh_metadata),
               "data_sha256":sha256(args.data/"transforms.json"),"calibration_sha256":sha256(args.calibration),
               "source_hashes":{f["physical_camera"]:sha256(Path(f["file_path"])) for f in source_frames},
               "targets":targets,"neighbors":args.neighbors,"aggregation_mode":args.aggregation_mode,"depth_log_tolerance":args.depth_log_tolerance,
               "seam_cut_rank_penalty":args.seam_cut_rank_penalty,
               "seam_cut_bandwidth_penalty":args.seam_cut_bandwidth_penalty,
               "bandwidth_helper_sha256":sha256(SCRIPTS/'source_bandwidth_prior.py') if args.seam_cut_bandwidth_penalty else None,
               "primary_angular_camera_count":args.primary_angular_camera_count,
               "camera_color_calibration_sha256":None if args.camera_color_calibration is None else sha256(args.camera_color_calibration),
               "angular_surface_color_sha256":None if args.angular_surface_color is None else sha256(args.angular_surface_color),
               "angular_surface_color_helper_sha256":sha256(SCRIPTS/'angular_surface_color.py') if args.angular_surface_color else None,
               "hard_source_seam_leveling":args.hard_source_seam_leveling,
               "surface_texture_registration":args.surface_texture_registration,
               "texture_registration_helper_sha256":sha256(SCRIPTS/'surface_texture_registration.py') if args.surface_texture_registration else None,
               "seam_leveling_helper_sha256":sha256(SCRIPTS/'hard_source_seam_leveling.py') if args.hard_source_seam_leveling else None,
               "camera_color_model":args.camera_color_model,
               "overlap_exposure_grid":args.overlap_exposure_grid,
               "pixel_center_offset":args.pixel_center_offset,
               "exact_mesh_visibility":args.exact_mesh_visibility,
               "source_rgb_footprint_visibility":args.source_rgb_footprint_visibility,
               "source_rgb_depth_aware_sampling":args.source_rgb_depth_aware_sampling,
               'source_observed_free_space_veto':args.source_observed_free_space_veto,
               'source_observed_raw_depth_hashes':raw_depth_hashes,
               'source_observed_data_sha256':sha256(args.source_observed_depth_data/'transforms.json') if args.source_observed_free_space_veto else None,
               'source_observed_metadata_sha256':sha256(args.source_observed_mesh_metadata) if args.source_observed_free_space_veto else None,
               'free_space_helper_sha256':sha256(SCRIPTS/'carve_patchmatch_mesh_free_space.py') if args.source_observed_free_space_veto else None,
               "disocclusion_color_match":args.disocclusion_color_match,
               "seam_cut_visibility_radius":args.seam_cut_visibility_radius,
               "surface_color_field_smoothness":args.surface_color_field_smoothness,
               "surface_color_helper_sha256":sha256(SCRIPTS/'surface_color_field.py') if args.surface_color_field_smoothness else None,
               "mesh_visibility_helper_sha256":sha256(SCRIPTS/"mesh_texture_visibility.py") if args.exact_mesh_visibility else None,
               "color_helper_sha256":sha256(SCRIPTS/"patchmatch_color_calibration.py"),
               "depth_hole_fill_max_area":args.depth_hole_fill_max_area,"eval_rgb_read":False,
               "renderer_sha256":sha256(SCRIPTS/"render_mesh_image_blend.py"),
               "path_script_sha256":sha256(Path(__file__)),
               "seam_helper_sha256":sha256(SCRIPTS/"hard_texture_seam_cut.py"),
               "raycaster_sha256":sha256(SCRIPTS/"render_tsdf_mesh_depth.py")}
    request["sha256"] = canonical_sha256(request)
    request_path = args.output/"path_request.json"
    if request_path.exists() and json.loads(request_path.read_text())!=request:
        raise ValueError("Cannot resume a different path request")
    atomic_json(request_path,request)
    work = args.output/"scratch"
    work.mkdir(exist_ok=True)
    placeholder = work/"synthetic_target.png"
    if not placeholder.exists():
        Image.new("RGB",(int(targets[0]["w"]),int(targets[0]["h"]))).save(placeholder)
    normal_args = ["--orientation-method","none","--center-method","none","--no-auto-scale-poses",
                   "--scale-factor","1","--downscale-factor","1"]
    env = dict(os.environ,OMP_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8")
    def run(command,log):
        with log.open("w") as f:
            subprocess.run([sys.executable,*map(str,command)],check=True,env=env,stdout=f,stderr=subprocess.STDOUT)
    rows=[]
    for index,target in enumerate(targets):
        current=work/f"{index:04d}"
        current.mkdir(exist_ok=True)
        target=deepcopy(target)
        target["file_path"]=str(placeholder.resolve())
        local={"frames":source_frames+[target],"train_filenames":[f["file_path"] for f in source_frames],
               "val_filenames":[target["file_path"]],"test_filenames":[target["file_path"]]}
        atomic_json(current/"transforms.json",local)
        depth=current/"mesh_depth"
        if not (depth/"mesh_depth_manifest.json").exists():
            run([SCRIPTS/"render_tsdf_mesh_depth.py","--data",current,"--mesh",args.mesh,
                 "--output-dir",depth,"--no-write-color-png","--no-portable-manifest-paths",
                 *(["--split","val"] if index else []),*normal_args],current/"raycast.log")
            if index:
                # Source cameras and mesh are fixed over this path; reuse their
                # first raycasts exactly, and raycast only the new target camera.
                first=json.loads((work/"0000/mesh_depth/mesh_depth_manifest.json").read_text())
                manifest=json.loads((depth/"mesh_depth_manifest.json").read_text())
                manifest["images"]=[row for row in first["images"] if row["split"]=="train"]+manifest["images"]
                manifest["splits"]=["train","val"]
                manifest["source_cache_manifest_sha256"]=sha256(work/"0000/mesh_depth/mesh_depth_manifest.json")
                atomic_json(depth/"mesh_depth_manifest.json",manifest)
        render=current/"render"
        if not (render/"reprojection_audit.json").exists():
            if render.exists():
                raise RuntimeError(f"Incomplete renderer requires inspection: {render}")
            run([SCRIPTS/"render_mesh_image_blend.py","--data",current,"--mesh-depth-manifest",depth/"mesh_depth_manifest.json",
                 "--output-dir",render,"--neighbors",args.neighbors,"--aggregation-modes",args.aggregation_mode,"--blend-alphas","1",
                 "--depth-log-tolerance",args.depth_log_tolerance,"--depth-hole-fill-max-area",args.depth_hole_fill_max_area,
                 "--seam-cut-rank-penalty",args.seam_cut_rank_penalty,
                 "--primary-angular-camera-count",args.primary_angular_camera_count,
                 "--pixel-center-offset",args.pixel_center_offset,
                 "--seam-cut-bandwidth-penalty",args.seam_cut_bandwidth_penalty,
                 *(["--exact-mesh-visibility"] if args.exact_mesh_visibility else []),
                 *(["--source-rgb-footprint-visibility"] if args.source_rgb_footprint_visibility else []),
                 *(["--source-rgb-depth-aware-sampling"] if args.source_rgb_depth_aware_sampling else []),
                 *(['--source-observed-free-space-veto','--source-observed-depth-data',args.source_observed_depth_data,
                    '--source-observed-mesh-metadata',args.source_observed_mesh_metadata] if args.source_observed_free_space_veto else []),
                 *(["--disocclusion-color-match"] if args.disocclusion_color_match else []),
                 "--seam-cut-visibility-radius",args.seam_cut_visibility_radius,
                 "--surface-color-field-smoothness",args.surface_color_field_smoothness,
                 *([] if args.angular_surface_color is None else ['--angular-surface-color',args.angular_surface_color]),
                 *(['--hard-source-seam-leveling'] if args.hard_source_seam_leveling else []),
                 *(['--surface-texture-registration'] if args.surface_texture_registration else []),
                 *([] if args.overlap_exposure_grid is None else ["--overlap-exposure-grid",*args.overlap_exposure_grid]),
                 *([] if args.camera_color_calibration is None else ["--camera-color-calibration",args.camera_color_calibration,
                     "--camera-color-model",args.camera_color_model]),
                 "--skip-ground-truth-copy","--device","cuda",*normal_args],current/"render.log")
        source=render/f"{args.aggregation_mode.replace('-','_')}{args.neighbors}"/"eval_pred_0000.png"
        retained=args.output/f"view_{index:04d}.png"
        shutil.copyfile(source,retained)
        rows.append({"index":index,"physical_camera":target["physical_camera"],"render":str(retained),"sha256":sha256(retained)})
        atomic_json(args.output/"path_manifest.json",{"request_sha256":request["sha256"],"state":"complete" if index==len(targets)-1 else "running","views":rows})
        print(f"view={index+1}/{len(targets)} camera={target['physical_camera']} complete",flush=True)


if __name__ == "__main__":
    main()
