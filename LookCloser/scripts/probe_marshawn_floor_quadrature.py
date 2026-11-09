"""Same frozen field/rays: read-only candidate/all-cells sampling-step convergence."""
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def main(a):
    for p in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',ROOT/'final_pair/code',ROOT/'native_png_linear_distortion_STAGED/code']:
        sys.path.insert(0,str(p))
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from marshawn_release_common import trace_native,authoritative_grid
    from build_marshawn_release_grid import witness_guard
    from probe_marshawn_occupancy import file_sha,digest_model
    from audit_marshawn_floor_rays import metrics
    torch.set_num_threads(2);reference=json.loads(a.reference.read_text());selection=json.loads((a.reference.parent.parent/'selection.json').read_text());checkpoint=Path(reference['checkpoint'])
    config=ROOT/'native_png_linear_distortion_STAGED/run/marshawn_001660_png_distortion_linear001/instant-ngp-bounded/cold_seed42/config.yml'
    def configure(cfg):cfg.eval_checkpoint=checkpoint;return cfg
    cfg,pipe,loaded,step=eval_setup(config,test_mode='val',eval_num_rays_per_chunk=512,update_config_callback=configure);model=pipe.model;pipe.eval()
    before=digest_model(model);saved=model.occupancy_grid.binaries.detach().clone();saved_step=model.config.render_step_size
    if saved_step!=.003:raise ValueError('Expectedoriginal.003step')
    folder=ROOT/'native_png_linear_distortion_STAGED/grids';stem=f'step-{step:09d}_grid';core,active,ema,boxes,policy=authoritative_grid(folder/(stem+'.npz'),folder/(stem+'.json'),checkpoint)
    points=np.load(ROOT/'geometry/train_witnesses.npz')['xyz'];_,_,candidate,_=witness_guard(core,boxes,points)
    receipt=json.loads((a.candidate/'receipt.json').read_text());z=np.load(a.candidate/'grid.npz')
    if file_sha(a.candidate/'grid.npz')!=receipt['grid_archive_sha256']or not np.array_equal(z['active'],candidate)or not np.array_equal(saved.cpu().numpy(),active):raise ValueError('Candidate/sourcebindingchanged')
    a.output.mkdir(exist_ok=False);rows=[];arrays={}
    try:
        for ref in reference['rows']:
            q=ref['query'];ds=getattr(pipe.datamanager,q['split']+'_dataset');ci=[Path(p).stem.split('_')[-1]for p in ds.image_filenames].index(q['camera']);camera=ds.cameras[ci:ci+1].to(model.device)
            ray=camera.generate_rays(0,coords=torch.tensor([[q['y']+.5,q['x']+.5]],device=model.device),keep_shape=False);o=ray.origins[0].cpu().numpy();v=ray.directions[0].cpu().numpy()
            if not np.array_equal(o,ref['origin'])or not np.array_equal(v,ref['direction']):raise ValueError('Actualrayschanged')
            baseline=ref['profiles']['uniform16384'];target=np.array(baseline['actual_rgb']);row=dict(query=q,uniform16384=baseline,steps={})
            for dt in [.003,.0015,.00075,.000375]:
                model.config.render_step_size=dt;parts={}
                for label,mask in [('candidate',candidate),('all_cells',np.ones_like(candidate))]:
                    with torch.no_grad():model.occupancy_grid.binaries.copy_(torch.as_tensor(mask,device=model.device))
                    colors=[]
                    def hook(_m,_args,out):colors.append(out[FieldHeadNames.RGB].detach().float().cpu().numpy().reshape(-1,3).copy())
                    handle=model.field.register_forward_hook(hook)
                    try:trace=trace_native(model,ray)
                    finally:handle.remove()
                    if len(colors)!=1:raise ValueError('Expectedoneactualfieldcall')
                    rgb=(trace['weights'][:,None]*colors[0]).sum(0);stat=metrics(trace,o,v,np.array(selection['normal']),selection['offset'],.03)
                    stat.update(actual_rgb=rgb.tolist(),RGB_error_vs_uniform16384=float(abs(rgb-target).max()),
                        opacity_error_vs_uniform16384=abs(stat['opacity']-baseline['opacity']),
                        depth_mean_error_vs_uniform16384=abs(stat['depth_mean']-baseline['depth_mean']))
                    parts[label]=stat
                    for k,val in dict(trace,rgb_samples=colors[0]).items():
                        if isinstance(val,np.ndarray):arrays[f'q{q["id"]:02d}_dt{str(dt).replace(".","p")}_{label}_{k}']=val
                row['steps'][str(dt)]=parts
            rows.append(row);(a.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(reference['rows'])))+'\n')
    finally:
        model.config.render_step_size=saved_step
        with torch.no_grad():model.occupancy_grid.binaries.copy_(saved)
        if digest_model(model)!=before or model.config.render_step_size!=saved_step:raise ValueError('Exactstate/configrestorationfailed')
    summary={dt:{label:dict(max_RGB_error=max(r['steps'][dt][label]['RGB_error_vs_uniform16384']for r in rows),mean_RGB_error=float(np.mean([r['steps'][dt][label]['RGB_error_vs_uniform16384']for r in rows])),max_opacity_error=max(r['steps'][dt][label]['opacity_error_vs_uniform16384']for r in rows),max_mean_depth_error=max(r['steps'][dt][label]['depth_mean_error_vs_uniform16384']for r in rows))for label in ['candidate','all_cells']}for dt in rows[0]['steps']}
    np.savez_compressed(a.output/'traces.npz',**arrays);result=dict(rows=rows,summary=summary,checkpoint_sha256=file_sha(checkpoint),source_reference_sha256=file_sha(a.reference),script_sha256=file_sha(__file__),original_state_sha256=before,original_step_size=saved_step,full_original_state_and_config_restored=True,optimizer_updates=0,black_background=True)
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','candidate','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
