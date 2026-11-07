"""Prepare, but never launch, one historical-runtime native4K render resource probe."""
import argparse,hashlib,json,os,py_compile,shutil,subprocess
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def patch_renderer(source):
    edits=[]
    def replace(before,after):
        nonlocal source
        assert source.count(before)==1,before
        source=source.replace(before,after);edits.append([before,after])
    replace('    w,h = int(camera.width.item()), int(camera.height.item())', '''    if (int(camera.width.item()), int(camera.height.item())) != (1920,1012):
        raise ValueError('Probe requires unchanged FHD native evaluation cameras')
    camera.rescale_output_resolution(2.)
    diagnostic = json.loads(a.diagnostic.read_text())
    if diagnostic.get('training_eligible') is not False:
        raise ValueError('Diagnostic camera descriptor must be excluded from training')
    expected = next(row for row in diagnostic['frames'] if row['physical_camera'] == a.camera)
    native_pose = np.linalg.inv(np.asarray(diagnostic['world_to_box'])) @ np.asarray(expected['transform_matrix'])
    actual_pose = np.eye(4); actual_pose[:3] = camera.camera_to_worlds[0].cpu().numpy()
    pose_error = float(np.max(np.abs(actual_pose-native_pose)))
    if pose_error > 1e-6:
        raise ValueError('Runtime native pose differs from inverse-floor diagnostic pose')
    intrinsic_errors = {name:abs(float(getattr(camera,attribute).item())-expected[name])
                        for name,attribute in [('fl_x','fx'),('fl_y','fy'),('cx','cx'),('cy','cy')]}
    if max(intrinsic_errors.values()) > .001:
        raise ValueError('Runtime doubled intrinsics differ from diagnostic raster')
    w,h = int(camera.width.item()), int(camera.height.item())
    if (w,h) != (expected['w'],expected['h']) or (w,h)!=(3840,2024):
        raise ValueError('Wrong actual4K raster')
    diagnostic_RGB = a.diagnostic.parent / expected['file_path']''')
    replace("    source_array=ds.get_image_float32(ci).detach().cpu().numpy()[...,:3]", "    source_array=np.asarray(Image.open(diagnostic_RGB).convert('RGB'), dtype=np.float32)/255.\n    if source_array.shape != (h,w,3): raise ValueError('Wrong actual4K source shape')")
    replace("        camera_runtime=c,display_transform=a.display_transform,overlay=overlay_receipt,", "        camera_runtime=c,display_transform=a.display_transform,overlay=overlay_receipt,\n        diagnostic_descriptor_sha256=sha(a.diagnostic),diagnostic_RGB_sha256=sha(diagnostic_RGB),\n        diagnostic_only_not_primary_metric=True,resolution_scale=2.,native_pose_error=pose_error,intrinsic_errors=intrinsic_errors,")
    replace("    p.add_argument('--camera',required=True);p.add_argument('--chunk',type=int,default=512)", "    p.add_argument('--camera',required=True);p.add_argument('--chunk',type=int,default=512)\n    p.add_argument('--diagnostic',type=Path,required=True)")
    return source,edits


def main(a):
    release=json.loads(a.production_release.read_text());release_sha=sha(a.production_release)
    production=Path(release['output']);root=a.output;root.mkdir(parents=True,exist_ok=False);code=root/'code';code.mkdir()
    source=Path(release['renderer']);text,edits=patch_renderer(source.read_text());renderer=code/'render_native_4k_partition.py';renderer.write_text(text)
    for name in ['aabb_visualization.py']:
        shutil.copyfile(source.parent/name,code/name)
    shutil.copyfile(a.code/'run_marshawn_native_4k_visual_probe.py',code/'run_marshawn_native_4k_visual_probe.py')
    shutil.copyfile(a.code/'audit_marshawn_native_4k_camera_resize.py',code/'audit_marshawn_native_4k_camera_resize.py')
    shutil.copyfile(Path(__file__),code/Path(__file__).name)
    for p in code.glob('*.py'):py_compile.compile(str(p),doraise=True)
    (root/'renderer_delta.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),staged_sha256=sha(renderer),exact_replacements=edits,
        sampler_field_compositor_AABB_rules_unchanged=True),indent=2)+'\n')
    config=production/'configs'/release_sha/'HD.yml';assert config.exists()
    diagnostic=json.loads(a.diagnostic.read_text());assert diagnostic['training_eligible']is False
    camera_proof=root/'CPU_camera_resize.json'
    command=[release['python'],str(code/'audit_marshawn_native_4k_camera_resize.py'),'--diagnostic',str(a.diagnostic),
             '--native',str(a.native),'--source',str(source),'--staged',str(renderer),'--output',str(camera_proof)]
    for path in release['python_path']:command+=['--python-path',path]
    subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    bindings=dict(release['bound_files']);bindings[str(a.production_release)]=release_sha;bindings[str(config)]=sha(config);bindings[str(a.diagnostic)]=sha(a.diagnostic)
    for row in diagnostic['frames']:
        for key in ['file_path','mask_path']:
            path=a.diagnostic.parent/row[key];bindings[str(path)]=sha(path)
    for p in list(code.glob('*.py'))+[root/'renderer_delta.json',camera_proof,a.native]:bindings[str(p)]=sha(p)
    plan=dict(kind='PREPARED_NOT_RELEASED_ONE_NATIVE4K_HC_RESOURCE_PROBE',production=str(production),production_controller_pid=221219,
        output=str(root/'probe_output'),python=release['python'],python_path=release['python_path'],renderer=str(renderer),config=str(config),source=release['source'],box=release['box'],
        diagnostic=str(a.diagnostic),camera='HC',chunk=512,gpu=0,GPU_lock=release['GPU_lock'],
        minimum_available_RAM_GiB=20,minimum_disk_GiB=20,abort_available_RAM_GiB=4,maximum_GPU_MiB=12288,maximum_seconds=1200,
        runtime=dict(torch=release['torch_version'],cuda=release['cuda_version']),bound_files=bindings,
        requires_paired_completion_and_worker_cache_exit=True,optimizer_updates=0,source_mutation=False,
        primary_FHD_protocol_unchanged=True,remaining_eight_renders_authorized=False,
        following_stage='After resource and visual review only: remaining native100k cameras and both arms at latestcommoncompletedstep, total9views including thisprobe. Source checkpoints laterbound explicitly; no selection from4Kdiagnostic.')
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(root/'release.template.json').write_text(json.dumps(dict(approved_resource_probe=False,plan_sha256=sha(root/'plan.json')),indent=2)+'\n')
    print(json.dumps(dict(plan=str(root/'plan.json'),sha256=sha(root/'plan.json'),bound_files=len(bindings),GPU_launched=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['production-release','diagnostic','native','code','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
