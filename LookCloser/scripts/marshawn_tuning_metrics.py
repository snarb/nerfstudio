"""Frozen-mask Full-HD metrics and restartable two-signal saturation policy.

The approved metric composites both inputs onto RGB .5 outside each GT actor
mask. It is not the historical complete-receptive-field or whole-image LPIPS.
Imports are lightweight so the controller can restore plateau state without CUDA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Optional

DOMAIN = 'fixed GT actor crop; identical RGB0.5 exterior; AlexNet LPIPS without resize'
HISTORICAL_EVAL_STEPS = (15188, 30376, 45564, 60752, 75940, 91128, 106316)


def masked_actor_metrics(prediction, target, mask, lpips_model, *, crop_xyxy):
    """Score one actor on original-size NCHW RGB, with a sealed GT-only crop.

    PSNR/Charbonnier use all actor pixels; SSIM averages centers in the actor
    mask after neutral compositing (therefore boundary windows include neutral
    padding). LPIPS averages the entire fixed crop, including its identical
    exterior. Crop and mask MUST be identical across configurations.
    """
    import torch
    from torchmetrics.functional.image import structural_similarity_index_measure

    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[:2] != (1, 3):
        raise ValueError('Expected matching single NCHW RGB images')
    if mask.dtype != torch.bool or mask.shape != (1, 1, *prediction.shape[-2:]):
        raise ValueError('Expected fixed boolean N1HW mask')
    if mask.device != prediction.device or target.device != prediction.device:
        raise ValueError('Images and mask must share device')
    if not mask.any():
        raise ValueError('Empty actor mask')
    if lpips_model.training or getattr(lpips_model, 'pnet_type', None) != 'alex' or getattr(lpips_model, 'resize', None) is not None:
        raise ValueError('Expected eval-mode AlexNet LPIPS without resize')
    for value in (prediction, target):
        if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError('Expected finite display RGB in [0,1]')
    if len(crop_xyxy) != 4 or any(type(v) is not int for v in crop_xyxy):
        raise ValueError('Crop must contain four integer coordinates')
    x0, y0, x1, y1 = crop_xyxy
    h, w = mask.shape[-2:]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or min(x1-x0, y1-y0) < 64:
        raise ValueError('Crop must be in image bounds and at least 64x64 for AlexNet')
    local = mask[:, :, y0:y1, x0:x1]
    if int(local.sum()) != int(mask.sum()):
        raise ValueError('Frozen crop excludes actor pixels')
    with torch.inference_mode():
        p = prediction.float()[:, :, y0:y1, x0:x1]
        g = target.float()[:, :, y0:y1, x0:x1]
        delta = (p-g)[local.expand_as(p)]
        mse = float(delta.square().mean())
        validation_loss = float(torch.sqrt(delta.square()+1e-4).mean())
        p = torch.where(local, p, .5)
        g = torch.where(local, g, .5)
        lpips = float(lpips_model(p, g, normalize=True).mean())
        _, ssim_map = structural_similarity_index_measure(p, g, data_range=1., sigma=1.5,
                                                          kernel_size=11, return_full_image=True)
        ssim = float(ssim_map[local.expand_as(ssim_map)].mean())
    return dict(psnr=float('inf') if mse == 0 else -10*math.log10(mse), ssim=ssim,
                lpips=lpips, validation_loss=validation_loss, actor_pixels=int(local.sum()),
                crop_xyxy=list(crop_xyxy), crop_pixels=(y1-y0)*(x1-x0), domain=DOMAIN,
                validation_loss_definition='mean RGB Charbonnier sqrt(error^2 + 1e-4) on actor pixels',
                ssim_domain='actor centers on neutral-composited crop; includes contour windows')


def aggregate_actor_metrics(records, *, expected_views=('DC', 'HC', 'KA'), expected_actors=('black', 'grey')):
    """Equal actors within each view, then equal views; reject incomplete gates."""
    records = list(records)
    expected = {(v, a) for v in expected_views for a in expected_actors}
    keyed = {(r['view'], r['actor']): r for r in records}
    if len(keyed) != len(records) or set(keyed) != expected or not expected:
        raise ValueError('Require exactly one record for every expected view/actor pair')
    keys = ('psnr', 'ssim', 'lpips', 'validation_loss')
    for r in records:
        if r.get('domain') != DOMAIN:
            raise ValueError('Metric domain mismatch')
        for key in keys:
            val = float(r[key])
            if not math.isfinite(val) and not (key == 'psnr' and val == math.inf):
                raise ValueError('Invalid metric value')
    by_view = {v: {k: sum(float(keyed[v, a][k]) for a in expected_actors)/len(expected_actors)
                   for k in keys} for v in expected_views}
    by_actor = {a: {k: sum(float(keyed[v, a][k]) for v in expected_views)/len(expected_views)
                    for k in keys} for a in expected_actors}
    return dict(metrics={k: sum(r[k] for r in by_view.values())/len(by_view) for k in keys},
                by_view=by_view, by_actor=by_actor, records=records, domain=DOMAIN,
                aggregation='equal actors per camera, then equal cameras')


@dataclass
class PlateauTracker:
    """Mandatory full ancestry followed by 4 misses of BOTH independent signals.

    Significant-record thresholds allow subthreshold gains to accumulate. The
    first post-ancestry anchor is exactly106316. Only complete 5000-step gates
    may advance counters; duplicate replay is idempotent only for identical data.
    """
    anchor_step: int = 106316
    interval: int = 5000
    patience: int = 4
    loss_relative_delta: float = .001
    lpips_absolute_delta: float = .001
    last_step: Optional[int] = None
    last_loss: Optional[float] = None
    last_lpips: Optional[float] = None
    significant_loss: Optional[float] = None
    significant_lpips: Optional[float] = None
    loss_misses: int = 0
    lpips_misses: int = 0
    stopped: bool = False

    def __post_init__(self):
        if self.anchor_step < 0 or self.interval < 1 or self.patience < 1:
            raise ValueError('Invalid plateau cadence')
        if not 0 < self.loss_relative_delta < 1 or not 0 < self.lpips_absolute_delta < 1:
            raise ValueError('Invalid improvement threshold')
        if self.loss_misses < 0 or self.lpips_misses < 0:
            raise ValueError('Negative plateau counters')
        for value in (self.last_loss, self.last_lpips, self.significant_loss, self.significant_lpips):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError('Invalid saved metric')
        if (self.last_step is None) != (self.last_loss is None) or (self.last_loss is None) != (self.last_lpips is None):
            raise ValueError('Incomplete saved observation')
        anchored = self.last_step is not None and self.last_step >= self.anchor_step
        if anchored != (self.significant_loss is not None) or anchored != (self.significant_lpips is not None):
            raise ValueError('Incomplete saved plateau anchor')
        if anchored and (self.last_step-self.anchor_step) % self.interval:
            raise ValueError('Saved observation is off cadence')
        if not anchored and (self.loss_misses or self.lpips_misses):
            raise ValueError('Counters before mandatory ancestry completed')
        if self.stopped != (anchored and min(self.loss_misses, self.lpips_misses) >= self.patience):
            raise ValueError('Saved stop flag disagrees with counters')

    def observe(self, step, validation_loss, masked_lpips):
        if type(step) is not int or step < 0:
            raise ValueError('Step must be a nonnegative integer')
        loss, lpips = float(validation_loss), float(masked_lpips)
        if not all(math.isfinite(x) and x >= 0 for x in (loss, lpips)):
            raise ValueError('Finite nonnegative complete-validation metrics required')
        if step == self.last_step:
            if (loss, lpips) != (self.last_loss, self.last_lpips):
                raise ValueError('Conflicting duplicate validation')
            return self.to_dict()
        if self.stopped:
            raise ValueError('Plateau already reached; explicit new campaign required')
        if self.last_step is not None and step < self.last_step:
            raise ValueError('Validation steps must increase')
        if step > self.anchor_step:
            if self.significant_loss is None or step != self.last_step+self.interval:
                raise ValueError('Missing anchor or consecutive post-ancestry validation gate')
            loss_gain = self.significant_loss-loss
            lpips_gain = self.significant_lpips-lpips
            if loss_gain >= self.significant_loss*self.loss_relative_delta and loss_gain > 0:
                self.significant_loss, self.loss_misses = loss, 0
            else:
                self.loss_misses += 1
            if lpips_gain+1e-12 >= self.lpips_absolute_delta:
                self.significant_lpips, self.lpips_misses = lpips, 0
            else:
                self.lpips_misses += 1
        elif step == self.anchor_step:
            self.significant_loss, self.significant_lpips = loss, lpips
            self.loss_misses = self.lpips_misses = 0
        self.last_step, self.last_loss, self.last_lpips = step, loss, lpips
        self.stopped = min(self.loss_misses, self.lpips_misses) >= self.patience
        return self.to_dict()

    def to_dict(self):
        return {'schema_version': 1, **asdict(self)}

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        if payload.pop('schema_version', None) != 1:
            raise ValueError('Unsupported plateau schema')
        return cls(**payload)

    def save(self, path):
        path = Path(path)
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        with temporary.open('w') as output:
            json.dump(self.to_dict(), output, sort_keys=True, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def actor_regression_guard(candidate, control, *, relative_tolerance=.03, actors=('black','grey')):
    """Report, never silently filter, >3% actor degradation versus explicit control.

    Accepts a completed-task result (selection.by_actor), an evaluation result
    (by_actor), or None. Missing references remain unassessed, never a pass.
    """
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError('Finite nonnegative actor tolerance required')
    def actor_scores(value):
        if not value:
            return None
        selected=value.get('selection',value)
        return selected.get('by_actor')
    current, reference=actor_scores(candidate),actor_scores(control)
    if current is None or reference is None or any(a not in current or a not in reference for a in actors):
        return dict(guard_status='unassessed',reason='explicit control and complete per-actor metrics required',comparisons=[])
    comparisons=[]
    for actor in actors:
        new, old=float(current[actor]['lpips']),float(reference[actor]['lpips'])
        if not math.isfinite(new) or not math.isfinite(old) or new < 0 or old < 0:
            raise ValueError('Finite nonnegative actor LPIPS required')
        limit=old*(1+relative_tolerance)
        comparisons.append(dict(actor=actor,candidate_lpips=new,control_lpips=old,maximum_lpips=limit,
                                relative_change=(new/old-1) if old else (0. if new==0 else None),
                                passed=new <= limit+1e-12))
    passed=all(row['passed'] for row in comparisons)
    return dict(guard_status='pass' if passed else 'regression',comparisons=comparisons,
                relative_tolerance=relative_tolerance,
                automatic_promotion_allowed=passed)
