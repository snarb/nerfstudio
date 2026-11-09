"""Conservative multi-view mask support -> adaptive volume -> reproducible AABB.

This constructs an OBSERVED actor-volume hypothesis, not a proof that unknown
space is empty. Person identity is supplied explicitly; unknown instances never
become clear background. No NN, image-color fit or occupancy-grid mutation.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import argparse,json,math,time,hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt

BITS=np.asarray(list(product([0,1],repeat=3)),np.float64)

def select_person_instances(masks,instance_global_ids,selected_global_ids):
    """Explicit stable identity, never score/rank; all unselected people UNKNOWN."""
    masks=np.asarray(masks)
    if masks.ndim!=3 or masks.dtype!=np.bool_:raise ValueError('Boolean instance rasters required')
    selected=set(selected_global_ids)
    if len(selected)!=2:raise ValueError('Exactly two reviewed global fighter IDs required')
    mapping={int(k):v for k,v in instance_global_ids.items()}
    if any(i<0 or i>=len(masks)for i in mapping):raise ValueError('Instance index outside source masks')
    ids=[i for i,label in mapping.items()if label in selected]
    if {mapping[i]for i in ids}!=selected:raise ValueError('Both stable actor identities need explicit per-camera mapping')
    other=[i for i in range(len(masks))if i not in ids]
    fg=masks[ids].any(0);unknown=masks[other].any(0)if other else np.zeros_like(fg)
    return fg,unknown

def integral(mask):
    return np.pad(np.asarray(mask,np.int32).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32),((1,0),(1,0)))

def rectangle_sum(table,x0,y0,x1,y1):
    return table[y1,x1]-table[y0,x1]-table[y1,x0]+table[y0,x0]

@dataclass
class MaskCamera:
    name:str
    camera_to_world:np.ndarray
    fx:float
    fy:float
    cx:float
    cy:float
    foreground:np.ndarray
    unknown:np.ndarray
    valid:np.ndarray
    native_scale_yx:tuple=(1.,1.)

    def prepare(self,padding_native=64):
        self.camera_to_world=np.asarray(self.camera_to_world,np.float64)
        if self.camera_to_world.shape not in [(3,4),(4,4)]:raise ValueError('OpenGL camera-to-world required')
        shape=self.foreground.shape
        if len(shape)!=2 or any(np.asarray(x).dtype!=np.bool_ or x.shape!=shape for x in [self.foreground,self.unknown,self.valid]):
            raise ValueError('Equal boolean foreground/unknown/valid rasters required')
        if not np.isfinite(self.camera_to_world).all()or not np.isfinite([self.fx,self.fy,self.cx,self.cy,padding_native,*self.native_scale_yx]).all()or min(self.fx,self.fy)<=0 or min(self.native_scale_yx)<=0 or padding_native<0:
            raise ValueError('Finite calibrated camera/positive focal/padding>=0 required')
        rotation=self.camera_to_world[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-5)or np.linalg.det(rotation)<0:
            raise ValueError('Camera-to-world must contain a proper orthonormal OpenGL rotation')
        self.h,self.w=shape
        fg=self.foreground&self.valid
        if fg.any()and padding_native:
            fg=distance_transform_edt(~fg,sampling=self.native_scale_yx)<=padding_native
        self.foreground_integral=integral(fg&self.valid)
        # Unknown covers possible occluding people and invalid source; never BG.
        self.unknown_integral=integral(self.unknown|~self.valid)
        self.padded_foreground=fg&self.valid
        return self

def classify_cells(cameras,lower,upper,*,near=1e-5):
    """All eight corners; outward pixel footprint; no center-only rejection.

    Positive support means at least one selected-FG pixel intersects the entire
    projected-cell bounding footprint. A clear BG contradiction requires the
    WHOLE footprint in-raster/valid, with neither padded FG nor unknown pixels.
    Near-plane crossing is UNKNOWN and only an optimistic refinement support.
    """
    lower=np.asarray(lower,np.float64);upper=np.asarray(upper,np.float64)
    if lower.shape!=upper.shape or lower.ndim!=2 or lower.shape[1]!=3 or (upper<=lower).any():raise ValueError('Ordered Nx3 fullcell extents required')
    corners=lower[:,None,:]+BITS[None]*(upper-lower)[:,None,:]
    foreground=np.zeros(len(lower),np.int16);possible=foreground.copy();background=foreground.copy();unknown_views=foreground.copy()
    for camera in cameras:
        pose=camera.camera_to_world;local=(corners-pose[:3,3])@pose[:3,:3];depth=-local[...,2]
        front=(depth>near).all(1);cross=(depth>near).any(1)&~front
        # For unprojectable crossing cells, subdivision must precede any decision.
        possible[cross]+=1;unknown_views[~front]+=1
        ids=np.where(front)[0]
        if not len(ids):continue
        local=local[ids];d=depth[ids]
        u=camera.fx*local[...,0]/d+camera.cx-.5;v=-camera.fy*local[...,1]/d+camera.cy-.5
        # Outward coverage includes boundary pixels and roundoff, at most2px extra.
        left=np.floor(u.min(1)-.5);right=np.ceil(u.max(1)+.5)+1
        top=np.floor(v.min(1)-.5);bottom=np.ceil(v.max(1)+.5)+1
        full=(left>=0)&(top>=0)&(right<=camera.w)&(bottom<=camera.h)
        x0=np.clip(left,0,camera.w).astype(np.int64);x1=np.clip(right,0,camera.w).astype(np.int64)
        y0=np.clip(top,0,camera.h).astype(np.int64);y1=np.clip(bottom,0,camera.h).astype(np.int64)
        nonempty=(x1>x0)&(y1>y0)
        fg=rectangle_sum(camera.foreground_integral,x0,y0,x1,y1)>0
        uncertain=rectangle_sum(camera.unknown_integral,x0,y0,x1,y1)>0
        foreground[ids]+=fg;possible[ids]+=fg
        clear=full&nonempty&~fg&~uncertain
        background[ids]+=clear
        unknown_views[ids]+=(~full|uncertain|~nonempty)
    return foreground,possible,background,unknown_views

def outward_snap(bounds,step,origin=(0.,0.,0.)):
    b=np.asarray(bounds,np.float64);o=np.asarray(origin,np.float64)
    if not np.isfinite(step)or step<=0 or b.shape!=(2,3)or not np.isfinite(b).all()or o.shape!=(3,)or not np.isfinite(o).all()or(b[1]<=b[0]).any():raise ValueError('Finite ordered bounds and positive fixed world grid step required')
    return np.stack([o+np.floor((b[0]-o)/step)*step,o+np.ceil((b[1]-o)/step)*step])

def apply_minimum_margins(bounds,minimum_margin=0.,axis_margin=None,bottom_margin=None,*,snap_step=None,origin=(0.,0.,0.)):
    """Guarantee margins on every face BEFORE outward world-grid snapping.

    Axis/bottom arguments are absolute minimums, not additions or pixels.
    Effective XYZ=max(common,axis); effective lowerZ=max(XYZ.z,bottom).
    All quantities use the supplied prepared-world coordinate units.
    """
    b=np.asarray(bounds,np.float64)
    if b.shape!=(2,3)or not np.isfinite(b).all()or(b[1]<=b[0]).any():raise ValueError('Finite ordered source extents required')
    common=float(minimum_margin)
    axis=np.full(3,common)if axis_margin is None else np.asarray(axis_margin,np.float64)
    bottom=common if bottom_margin is None else float(bottom_margin)
    if not math.isfinite(common)or common<0 or axis.shape!=(3,)or not np.isfinite(axis).all()or(axis<0).any()or not math.isfinite(bottom)or bottom<0:
        raise ValueError('Finite nonnegative common, XYZ and lower-Z minimum margins required')
    effective=np.maximum(axis,common);lower=effective.copy();lower[2]=max(lower[2],bottom)
    expanded=np.stack([b[0]-lower,b[1]+effective]);result=outward_snap(expanded,snap_step,origin)if snap_step is not None else expanded
    actual_lower=b[0]-result[0];actual_upper=result[1]-b[1]
    if (actual_lower<lower-1e-12).any()or(actual_upper<effective-1e-12).any():raise RuntimeError('Outward snap violated requested margin')
    return result,dict(units='prepared_world; original physical unit unspecified',minimum_common_margin=common,
        requested_axis_minimums=None if axis_margin is None else axis.tolist(),requested_lower_Z_minimum=bottom_margin,
        effective_lower_face_minimums=lower.tolist(),effective_upper_face_minimums=effective.tolist(),
        applied_after_snap_lower_margins=actual_lower.tolist(),applied_after_snap_upper_margins=actual_upper.tolist(),
        verified_actual_margins_at_least_requested=True,world_snap_step=snap_step,world_grid_origin=list(origin),
        semantics='Absolute face minimums: max(common,XYZ); lowerZ additionally max(bottom). No relation to optical mask padding.')

def temporal_envelope(boxes,*,snap_step=None,origin=(0.,0.,0.),minimum_margin=0.,axis_margin=None,bottom_margin=None):
    """Union actual per-frame boxes, then optionally snap on ONE world lattice."""
    b=np.asarray(boxes,np.float64)
    if b.ndim!=3 or b.shape[1:]!=(2,3)or not np.isfinite(b).all()or(b[:,1]<=b[:,0]).any():raise ValueError('Actual ordered per-frame boxes required')
    result=np.stack([b[:,0].min(0),b[:,1].max(0)])
    return apply_minimum_margins(result,minimum_margin,axis_margin,bottom_margin,snap_step=snap_step,origin=origin)[0]

def derive_aabb(cameras,domain,*,leaf_size=.012,min_support=3,max_background=1,
                padding_native=64,margin_world=.02,axis_margin=None,bottom_margin=None,snap_step=.01,anchors=None,batch_size=8192,max_cells=4000000):
    domain=np.asarray(domain,np.float64)
    if domain.shape!=(2,3)or not np.isfinite(domain).all()or(domain[1]<=domain[0]).any():raise ValueError('Finite full calibrated search domain required')
    if not 1<=min_support<=len(cameras)or max_background<0 or leaf_size<=0 or margin_world<0:raise ValueError('Explicit valid support/size/uncertainty parameters required')
    apply_minimum_margins(domain,margin_world,axis_margin,bottom_margin,snap_step=snap_step) # Validate BEFORE any carving.
    for camera in cameras:camera.prepare(padding_native)
    lower,upper=domain[0][None],domain[1][None]
    depth=int(math.ceil(math.log2(float((domain[1]-domain[0]).max())/leaf_size)))
    depth=max(0,depth);levels=[];started=time.monotonic();last_stats=None
    for level in range(depth+1):
        input_count=len(lower)
        keep=[];fg_counts=[];bg_counts=[];unknown_counts=[];unsupported=clear_rejected=near_ambiguous=0
        for start in range(0,len(lower),batch_size):
            lo,hi=lower[start:start+batch_size],upper[start:start+batch_size]
            fg,potential,bg,unknown=classify_cells(cameras,lo,hi)
            active=(potential>=min_support)&(bg<=max_background)
            unsupported+=int(((potential<min_support)&(bg<=max_background)).sum())
            clear_rejected+=int((bg>max_background).sum())
            if level==depth:
                near_ambiguous+=int((active&(fg<min_support)).sum())
                active&=fg>=min_support
            keep.append(np.where(active)[0]+start);fg_counts.append(fg[active]);bg_counts.append(bg[active]);unknown_counts.append(unknown[active])
        ids=np.concatenate(keep)if keep else np.array([],np.int64)
        lower,upper=lower[ids],upper[ids]
        last_stats=(np.concatenate(fg_counts),np.concatenate(bg_counts),np.concatenate(unknown_counts))
        levels.append(dict(level=level,input_cells=input_count,
            retained=len(lower),clear_background_rejected=clear_rejected,unsupported_unknown_not_declared_empty=unsupported,
            unresolved_nearplane_not_declared_empty=near_ambiguous,elapsed_seconds=time.monotonic()-started))
        print(json.dumps(levels[-1]),flush=True)
        if not len(lower):raise ValueError('No multi-view supported actor volume; masks/support contract failed')
        if level<depth:
            if len(lower)*8>max_cells:raise RuntimeError('Adaptive cell budget exceeded; fail rather than silently coarsen')
            middle=(lower+upper)/2;span=(upper-lower)/2
            lower=(lower[:,None]+BITS[None]*span[:,None]).reshape(-1,3)
            upper=(middle[:,None]+BITS[None]*span[:,None]).reshape(-1,3)
    raw=np.stack([lower.min(0),upper.max(0)])
    anchor_count=0
    if anchors is not None:
        anchors=np.asarray(anchors,np.float64)
        if anchors.ndim!=2 or anchors.shape[1]!=3 or not np.isfinite(anchors).all():raise ValueError('Finite vetted prepared-world anchors required')
        if (anchors<domain[0]).any()or(anchors>domain[1]).any():raise ValueError('Vetted anchors outside searchdomain; expand domain explicitly')
        if len(anchors):raw=np.stack([np.minimum(raw[0],anchors.min(0)),np.maximum(raw[1],anchors.max(0))]);anchor_count=len(anchors)
    touches=((raw[0]<=domain[0]+1e-9)|(raw[1]>=domain[1]-1e-9))
    box,margin_receipt=apply_minimum_margins(raw,margin_world,axis_margin,bottom_margin,snap_step=snap_step)
    receipt=dict(kind='MULTIVIEW_MASK_SUPPORTED_CONSERVATIVE_AABB_NOT_HIDDEN_GEOMETRY_GT',
        domain=domain.tolist(),aabb=box.tolist(),retained_extent_aabb=raw.tolist(),leaf_size_upper_bound=leaf_size,
        actual_leaf_extent=((domain[1]-domain[0])/2**depth).tolist(),depth=depth,retained_cells=len(lower),
        min_foreground_support=min_support,max_clear_background_contradictions=max_background,
        padding_native_pixels=padding_native,margin_prepared_world=margin_world,world_snap_step=snap_step,
        margin_guarantee=margin_receipt,
        fixed_world_grid_origin=[0,0,0],search_boundary_touched_axes=np.where(touches)[0].tolist(),
        requires_expanded_search=bool(touches.any()),vetted_anchors_union=anchor_count,levels=levels,
        limitations=['Integral bounding footprints can retain false volume, especially at coarse resolution.',
            'Unobserved/occluded/out-of-FOV volume is UNKNOWN, never certified empty; only observed multi-view actor support contributes to this bounding hypothesis.',
            'Padding/contradiction allowance protect some mask false negatives, not an exact hair matte.',
            'Ray-box coverage is necessary but does not prove actor surface depth or hidden geometry.'],
        training_approved=False,elapsed_seconds=time.monotonic()-started)
    return receipt,dict(lower=lower,upper=upper,foreground_support=last_stats[0],background_contradictions=last_stats[1],unknown_views=last_stats[2])

def ray_box_hits(origins,directions,bounds):
    o=np.asarray(origins,np.float64);d=np.asarray(directions,np.float64);b=np.asarray(bounds,np.float64)
    parallel=np.abs(d)<1e-14;safe=np.where(parallel,1,d)
    lo=(b[0]-o)/safe;hi=(b[1]-o)/safe
    enter=np.where(parallel,-np.inf,np.minimum(lo,hi)).max(-1)
    leave=np.where(parallel,np.inf,np.maximum(lo,hi)).min(-1)
    return (~(parallel&((o<b[0])|(o>b[1]))).any(-1))&(leave>=np.maximum(enter,0))

def load_npz_cameras(manifest):
    p=Path(manifest);payload=json.loads(p.read_text());cameras=[]
    for row in payload['cameras']:
        archive=p.parent/row['masks']
        if row.get('masks_sha256')and hashlib.sha256(archive.read_bytes()).hexdigest()!=row['masks_sha256']:
            raise ValueError('Calibrated mask input changed')
        with np.load(archive,allow_pickle=False)as z:
            fg=z['foreground'].copy();unknown=z['unknown'].copy();valid=z['valid'].copy()
        c=row['camera_row'];cameras.append(MaskCamera(row['camera'],np.asarray(c['transform_matrix']),c['fl_x'],c['fl_y'],c['cx'],c['cy'],
            fg,unknown,valid,tuple(row.get('native_scale_yx',(1,1)))))
    return payload,cameras

def rebuild_margins(source_receipt,cells_archive,*,minimum_margin=.02,axis_margin=None,bottom_margin=None,snap_step=.01):
    """Rebuild ONLY the enclosing box from saved fullcells+bound anchors."""
    source_receipt=Path(source_receipt);cells_archive=Path(cells_archive);source=json.loads(source_receipt.read_text())
    with np.load(cells_archive,allow_pickle=False)as z:
        lo=z['lower'];hi=z['upper']
        if lo.shape!=hi.shape or lo.ndim!=2 or lo.shape[1]!=3 or not len(lo)or not np.isfinite(lo).all()or not np.isfinite(hi).all()or(hi<=lo).any():
            raise ValueError('Valid retained FULLvoxel extents required')
        raw=np.stack([lo.min(0),hi.max(0)])
    binding=source.get('source_bindings',{});anchor=binding.get('anchors')
    if anchor:
        if hashlib.sha256(Path(anchor).read_bytes()).hexdigest()!=binding['anchors_sha256']:raise ValueError('Vetted anchors changed')
        with np.load(anchor,allow_pickle=False)as z:xyz=z['witness_xyz']
        raw=np.stack([np.minimum(raw[0],xyz.min(0)),np.maximum(raw[1],xyz.max(0))])
    if not np.allclose(raw,source['retained_extent_aabb'],rtol=0,atol=1e-12):raise ValueError('Stored cells/anchors do not reproduce original source extent')
    box,guarantee=apply_minimum_margins(raw,minimum_margin,axis_margin,bottom_margin,snap_step=snap_step)
    return dict(kind='MARGIN_ONLY_REBUILD_NO_RECARVING',aabb=box.tolist(),retained_extent_aabb=raw.tolist(),
        margin_guarantee=guarantee,source_receipt=str(source_receipt),source_receipt_sha256=hashlib.sha256(source_receipt.read_bytes()).hexdigest(),
        cells_archive=str(cells_archive),cells_archive_sha256=hashlib.sha256(cells_archive.read_bytes()).hexdigest(),
        original_padding_native_pixels=source['padding_native_pixels'],source_bindings=binding,
        training_approved=False,NN_or_mask_changes=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--leaf-size',type=float,default=.012);p.add_argument('--min-support',type=int,default=3)
    p.add_argument('--max-background',type=int,default=1);p.add_argument('--padding-native',type=float,default=64)
    p.add_argument('--margin-world',type=float,default=.02);p.add_argument('--snap-step',type=float,default=.01)
    p.add_argument('--axis-margin',type=float,nargs=3);p.add_argument('--bottom-margin',type=float)
    p.add_argument('--rebuild-from',type=Path,help='Retained_cells.npz; reconstruct margins without carving')
    p.add_argument('--source-receipt',type=Path)
    p.add_argument('--max-cells',type=int,default=4000000);p.add_argument('--envelope',type=Path,nargs='*',default=[])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.rebuild_from:
        if not a.source_receipt:p.error('--source-receipt required with --rebuild-from')
        receipt=rebuild_margins(a.source_receipt,a.rebuild_from,minimum_margin=a.margin_world,axis_margin=a.axis_margin,
            bottom_margin=a.bottom_margin,snap_step=a.snap_step)
        (a.output/'aabb.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(0)
    if not a.manifest:p.error('--manifest required for carving')
    payload,cameras=load_npz_cameras(a.manifest)
    receipt,cells=derive_aabb(cameras,payload['search_domain'],leaf_size=a.leaf_size,min_support=a.min_support,
        max_background=a.max_background,padding_native=a.padding_native,margin_world=a.margin_world,axis_margin=a.axis_margin,
        bottom_margin=a.bottom_margin,snap_step=a.snap_step,max_cells=a.max_cells)
    if a.envelope:
        boxes=[receipt['aabb']]+[json.loads(f.read_text())['aabb']for f in a.envelope]
        receipt['temporal_window_envelope_aabb']=temporal_envelope(boxes,snap_step=a.snap_step).tolist()
        receipt['temporal_scope']='Union explicit perframe results; not a claim of constant actor motion'
    np.savez_compressed(a.output/'retained_cells.npz',**cells)
    (a.output/'aabb.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt['requires_expanded_search']:raise SystemExit('Searchboundary touched: expand inputdomain before using box')
