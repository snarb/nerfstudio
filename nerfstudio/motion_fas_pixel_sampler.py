# nerfstudio/motion_fas_pixel_sampler.py
"""
Combined frequency + motion aware pixel sampler for the temporal NeRF (VECTORIZED).

Subclasses :class:`LookCloserPixelSampler` (FAS) and adds a person-mask motion
signal. Three modes:

  * ``mode="split"`` (Variant A): every batch is a 3-way hard split into
    ``uniform_frac / freq_frac / motion_frac`` of the rays. ``uniform`` rays are
    base random sampling, ``freq`` rays follow the FAS frequency pmf, and
    ``motion`` rays follow the person-mask pmf. Sub-batches are concatenated and
    shuffled. The FAS warmup/ramp governs the freq part; ``motion_warmup_steps`` /
    ``motion_ramp_steps`` govern the motion fraction.

  * ``mode="region"`` (Variant B): per image the FAS frequency pmf is split into
    in-mask vs. out-of-mask support; ``person_frac`` of rays go to the in-mask
    support (FAS within the person region) and ``1 - person_frac`` to the
    out-of-mask support (FAS within background). The person allocation ramps in
    via the motion schedule; before that it is plain FAS.

  * ``mode="off"`` (default): pure FAS / parent behaviour - nothing changes.

PERFORMANCE: all per-image weighting is precomputed ONCE into flattened per-image
pmfs over frequency-map patch cells (and over person-mask cells). Each ``sample()``
call does only vectorized ``torch.multinomial`` draws (image-level by mass, then
pixel-level by pmf) + a vectorized patch-cell -> random-full-res-pixel mapping. No
per-batch Python loops over levels/buckets. This matches the base PixelSampler's
throughput order of magnitude.

Output contract: ``[B, 3] = (local_img_idx, row, col)`` long indices, where
``local_img_idx`` indexes the images PRESENT IN THE CURRENT BATCH. The base
``collate_image_dataset_batch`` indexes the size-N batch tensors with this column
and only afterwards remaps it to the absolute/global camera id via
``image_batch["image_idx"]``.

SUBSET CACHING: the datamanager may cache only a SUBSET of images
(``train_num_images_to_sample_from`` < dataset size). The cached pmfs are keyed by
GLOBAL image id; each ``sample()`` call slices out the batch images
(``image_batch["image_idx"]``) and emits LOCAL positions. Images in the batch
without a freq map / person mask fall back gracefully (uniform / no motion bias).

Person masks live in ``<data_dir>/<motion_map_dir>/person_masks.pt`` as a dict
``{stem: float16 [Hm, Wm]}`` (1 ~ person, dilated).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.lookcloser_pixel_sampler import (
    LookCloserPixelSampler,
    LookCloserPixelSamplerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

_MOTION_FLOOR = 1e-3  # small floor on motion pmf so non-person pixels keep tiny mass


@dataclass
class MotionFASPixelSamplerConfig(LookCloserPixelSamplerConfig):
    """Configuration for the combined frequency + motion (person-mask) sampler."""

    _target: Type = field(default_factory=lambda: MotionFASPixelSampler)

    mode: Literal["split", "region", "off"] = "off"
    """Sampling mode. 'off' = pure FAS (parent behaviour). 'split' = Variant A 3-way
    hard split (uniform/freq/motion). 'region' = Variant B region-gated FAS."""

    # --- Variant A (split) fractions; normalized to sum to 1 at runtime ---
    uniform_frac: float = 0.5
    """(split) Fraction of each batch sampled uniformly (base random)."""

    freq_frac: float = 0.25
    """(split) Fraction of each batch sampled via the FAS frequency pmf."""

    motion_frac: float = 0.25
    """(split) Fraction of each batch sampled by the person-mask motion pmf."""

    # --- Variant B (region) fraction ---
    person_frac: float = 0.8
    """(region) Fraction of rays allocated to person-mask pixels; the remainder
    goes to background. FAS runs WITHIN each region."""

    motion_map_dir: str = "person_masks"
    """Directory inside data_dir holding 'person_masks.pt' ({stem: [Hm, Wm]})."""

    motion_warmup_steps: int = 0
    """Sampler calls that use no motion bias before the motion fraction ramps in."""

    motion_ramp_steps: int = 0
    """Sampler calls over which the motion fraction ramps from 0 to its target."""


class MotionFASPixelSampler(LookCloserPixelSampler):
    """Vectorized frequency + person-mask motion aware pixel sampler.

    See module docstring. Keeps the strict ``[B, 3] = (local_img_idx, row, col)``
    output contract (LOCAL to the batch)."""

    config: MotionFASPixelSamplerConfig

    def __init__(self, config: MotionFASPixelSamplerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        # ---- Precomputed per-GLOBAL-image FREQUENCY structures (built once) ----
        # For each global image with a freq map:
        #   _freq_cells[g]  : long [M, 2] patch (y_patch, x_patch)
        #   _freq_pmf[g]    : float [M]  normalized pixel pmf within the image
        #   _freq_mass[g]   : float scalar = image's share of total FAS mass
        self._freq_cells: Dict[int, Tensor] = {}
        self._freq_pmf: Dict[int, Tensor] = {}
        self._freq_mass: Dict[int, float] = {}
        # Patch geometry (set by parent _initialize_buckets).
        # self.patch_size, self.patch_stride, self.image_shapes already exist.

        # ---- Precomputed per-GLOBAL-image MOTION (person-mask) structures ----
        #   _motion_cells[g] : long [K, 2] mask (my, mx) with positive weight
        #   _motion_pmf[g]   : float [K] normalized
        #   _motion_shape[g] : (hm, wm)
        #   _motion_mass[g]  : scalar (sum of raw mask weight)
        self._motion_cells: Dict[int, Tensor] = {}
        self._motion_pmf: Dict[int, Tensor] = {}
        self._motion_shape: Dict[int, Tuple[int, int]] = {}
        self._motion_mass: Dict[int, float] = {}
        # In/out-of-mask freq pmfs per image for region mode (lazy per global id).
        self._region_split_cache: Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        self._motion_loaded = False

        # ---- Per-sample() batch context (global<->local) ----
        self._batch_globals: Optional[Tensor] = None
        self._fas_built = False

    # ===================================================================== init
    def _build_freq_pmfs(self):
        """Convert the parent's per-level buckets into per-image flattened pmfs +
        masses, reproducing the exact FAS ramp distribution. Runs once."""
        self._fas_built = True
        if not self.config.enable_fas or not self.buckets:
            return
        num_levels = int(self.config.num_levels)
        probs = np.asarray(self.probs, dtype=np.float64)  # per-level batch mass (sums ~1)
        counts = np.asarray(self.level_counts, dtype=np.float64)  # per-level total patch count

        # Per-patch weight = probs[level] / count[level] (uniform within a level).
        # Accumulate per image: cells + weights.
        per_img_cells: Dict[int, List[Tensor]] = {}
        per_img_w: Dict[int, List[Tensor]] = {}
        for l in range(num_levels):
            bucket = self.buckets.get(l)
            if bucket is None or bucket.shape[0] == 0 or counts[l] <= 0 or probs[l] <= 0:
                continue
            w_per_patch = float(probs[l] / counts[l])
            imgs = bucket[:, 0].long()
            ys = bucket[:, 1].long()
            xs = bucket[:, 2].long()
            for g in torch.unique(imgs).tolist():
                sel = imgs == g
                cells = torch.stack([ys[sel], xs[sel]], dim=1)
                w = torch.full((int(sel.sum().item()),), w_per_patch, dtype=torch.float64)
                per_img_cells.setdefault(g, []).append(cells)
                per_img_w.setdefault(g, []).append(w)

        for g in per_img_cells:
            cells = torch.cat(per_img_cells[g], dim=0)
            w = torch.cat(per_img_w[g], dim=0)
            mass = float(w.sum().item())
            self._freq_cells[g] = cells.long()
            self._freq_pmf[g] = (w / w.sum()).float() if w.sum() > 0 else torch.full_like(w, 1.0 / len(w)).float()
            self._freq_mass[g] = mass

        CONSOLE.print(
            f"[bold green]MotionFASPixelSampler:[/bold green] built vectorized FAS pmfs for "
            f"{len(self._freq_cells)} images (patch_size={self.patch_size}, stride={self.patch_stride})."
        )

    def _resolve_motion_dir(self, dataset: Dataset) -> Optional[Path]:
        if not hasattr(dataset, "image_filenames") or len(dataset.image_filenames) == 0:
            return None
        data_dir = Path(dataset.image_filenames[0]).parent
        for candidate in (data_dir / self.config.motion_map_dir, data_dir.parent / self.config.motion_map_dir):
            if candidate.exists():
                return candidate
        return None

    def _load_person_masks(self, dataset: Dataset):
        """Load person masks keyed by GLOBAL image id and precompute per-image
        motion pmfs (with a small floor). Missing file / stems are tolerated."""
        self._motion_loaded = True
        if self.config.mode == "off":
            return

        motion_dir = self._resolve_motion_dir(dataset)
        if motion_dir is None:
            CONSOLE.print(
                f"[yellow]MotionFASPixelSampler:[/yellow] motion dir '{self.config.motion_map_dir}' not found; "
                "all images get no motion bias (FAS/uniform only)."
            )
            return
        mask_file = motion_dir / "person_masks.pt"
        if not mask_file.exists():
            CONSOLE.print(
                f"[yellow]MotionFASPixelSampler:[/yellow] {mask_file} missing; "
                "all images get no motion bias (FAS/uniform only)."
            )
            return

        masks = torch.load(mask_file, map_location="cpu")
        if not isinstance(masks, dict):
            raise ValueError(f"{mask_file} must be a dict {{stem: [Hm, Wm]}}, got {type(masks)}.")

        n_found = 0
        for g, image_path in enumerate(dataset.image_filenames):
            m = masks.get(Path(image_path).stem)
            if m is None:
                continue
            m = torch.as_tensor(m).float()
            if m.ndim != 2 or not torch.isfinite(m).all():
                continue
            m = torch.clamp(m, min=0.0)
            if m.sum() <= 0:
                continue
            hm, wm = m.shape
            flat = m.flatten()
            # Motion pmf over ALL mask cells with a small uniform floor so the
            # support never collapses; the bias still concentrates on person cells.
            floor = _MOTION_FLOOR * float(flat.mean().item() + 1e-12)
            w = flat + floor
            pmf = (w / w.sum()).float()
            # Cells with positive raw mask weight (the person region), for region mode.
            self._motion_shape[g] = (hm, wm)
            self._motion_mass[g] = float(m.sum().item())
            # Store full pmf + per-cell raw mask value (to split person/background).
            self._motion_pmf[g] = pmf
            self._motion_cells[g] = flat  # raw mask weight per flat cell (for masking)
            n_found += 1

        CONSOLE.print(
            f"[bold green]MotionFASPixelSampler:[/bold green] built motion pmfs for "
            f"{n_found}/{len(dataset.image_filenames)} train images (mode={self.config.mode})."
        )

    # =============================================================== batch ctx
    def _set_batch_context(self, image_batch: Dict):
        image_idx = image_batch.get("image_idx")
        if image_idx is None:
            self._batch_globals = None
        else:
            self._batch_globals = torch.as_tensor(image_idx).flatten().long()

    def _batch_global_ids(self, num_images: int, device) -> Tensor:
        """Global id for each LOCAL batch position [N]."""
        if self._batch_globals is not None:
            return self._batch_globals.to(device)
        return torch.arange(num_images, device=device)

    # ================================================================ schedule
    def _active_motion_strength(self) -> float:
        warmup = max(int(self.config.motion_warmup_steps), 0)
        ramp = max(int(self.config.motion_ramp_steps), 0)
        if self.sample_count < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(max(self.sample_count - warmup, 0) / float(ramp), 1.0)

    # ====================================================== vectorized helpers
    def _image_level_counts(self, masses: Tensor, n_samples: int, device) -> Tuple[Tensor, Tensor]:
        """Given per-local-image masses [N] (>=0), draw how many of n_samples rays
        go to each local image (vectorized multinomial-by-ray, then bincount)."""
        total = masses.sum()
        if total <= 0:
            # Uniform over all images.
            probs = torch.full_like(masses, 1.0 / max(masses.numel(), 1))
        else:
            probs = masses / total
        picks = torch.multinomial(probs, n_samples, replacement=True)  # [n] local ids
        counts = torch.bincount(picks, minlength=masses.numel())
        return counts, probs

    def _draw_pixels_from_cell_pmf(
        self,
        local_to_cells: Dict[int, Tensor],   # local id -> [M,2] cell (y,x)
        local_to_pmf: Dict[int, Tensor],     # local id -> [M] pmf
        counts: Tensor,                      # [N] rays per local image
        cell_h: int, cell_w: int,            # footprint of one cell in full-res px
        image_height: int, image_width: int,
        device,
    ) -> Tensor:
        """For each local image draw `counts[i]` cells by its pmf, then map each cell
        to a uniform random full-res pixel within its (cell_h x cell_w) footprint.
        Returns LOCAL [sum(counts), 3]."""
        out_img_l: List[Tensor] = []
        out_r_l: List[Tensor] = []
        out_c_l: List[Tensor] = []
        nz = torch.nonzero(counts, as_tuple=False).flatten().tolist()
        for li in nz:
            k = int(counts[li].item())
            cells = local_to_cells.get(li)
            pmf = local_to_pmf.get(li)
            if cells is None or pmf is None or cells.shape[0] == 0:
                # No structure for this image -> uniform pixels.
                r = torch.randint(0, image_height, (k,), device=device)
                c = torch.randint(0, image_width, (k,), device=device)
            else:
                sel = torch.multinomial(pmf.to(device), k, replacement=True)
                cy = cells[sel, 0].to(device)
                cx = cells[sel, 1].to(device)
                r = (cy * cell_h + torch.randint(0, cell_h, (k,), device=device)).long()
                c = (cx * cell_w + torch.randint(0, cell_w, (k,), device=device)).long()
                r = torch.clamp(r, 0, image_height - 1)
                c = torch.clamp(c, 0, image_width - 1)
            out_img_l.append(torch.full((k,), li, dtype=torch.long, device=device))
            out_r_l.append(r.long())
            out_c_l.append(c.long())
        if not out_img_l:
            return torch.empty((0, 3), dtype=torch.long, device=device)
        return torch.stack([torch.cat(out_img_l), torch.cat(out_r_l), torch.cat(out_c_l)], dim=1)

    # ------- build local-keyed views of cached structures for THIS batch -------
    def _local_freq_views(self, num_images, device):
        globals_per_local = self._batch_global_ids(num_images, device).tolist()
        cells, pmf, masses = {}, {}, torch.zeros(num_images, dtype=torch.float64)
        for li, g in enumerate(globals_per_local):
            if g in self._freq_cells:
                cells[li] = self._freq_cells[g]
                pmf[li] = self._freq_pmf[g]
                masses[li] = self._freq_mass[g]
        return cells, pmf, masses

    def _freq_indices(self, n, num_images, image_height, image_width, device) -> Tensor:
        """Vectorized FAS sampling for exactly `n` rays -> LOCAL indices."""
        if n <= 0:
            return torch.empty((0, 3), dtype=torch.long, device=device)
        cells, pmf, masses = self._local_freq_views(num_images, device)
        if masses.sum() <= 0:
            # No freq structure in this batch -> uniform (local).
            return PixelSampler.sample_method(self, n, num_images, image_height, image_width, None, device).long()
        counts, _ = self._image_level_counts(masses.to(device), n, device)
        idx = self._draw_pixels_from_cell_pmf(
            cells, pmf, counts, self.patch_stride, self.patch_stride, image_height, image_width, device
        )
        # Top up if any image lacked structure (counts there produced uniform anyway,
        # so size already == n). Guard exactness:
        if idx.shape[0] < n:
            extra = PixelSampler.sample_method(self, n - idx.shape[0], num_images, image_height, image_width, None, device).long()
            idx = torch.cat([idx, extra], dim=0)
        return idx[:n]

    # ------------------ motion (person-mask) vectorized sampling ---------------
    def _local_motion_views(self, num_images, device, person_only=None):
        """Build local-keyed motion cells/pmf. person_only: None=full pmf,
        True=in-mask support, False=background support (for region mode)."""
        globals_per_local = self._batch_global_ids(num_images, device).tolist()
        cells, pmf, masses = {}, {}, torch.zeros(num_images, dtype=torch.float64)
        for li, g in enumerate(globals_per_local):
            if g not in self._motion_pmf:
                continue
            hm, wm = self._motion_shape[g]
            raw = self._motion_cells[g]  # flat raw mask weights [hm*wm]
            if person_only is None:
                p = self._motion_pmf[g]
                supp = torch.arange(raw.numel())
                weights = p
            else:
                inside = raw > 0.5
                supp_mask = inside if person_only else ~inside
                supp = torch.nonzero(supp_mask, as_tuple=False).flatten()
                if supp.numel() == 0:
                    continue
                if person_only:
                    weights = raw[supp]
                else:
                    weights = torch.ones(supp.numel(), dtype=torch.float32)
                weights = weights / weights.sum()
            ys = (supp // wm).long()
            xs = (supp % wm).long()
            cells[li] = torch.stack([ys, xs], dim=1)
            pmf[li] = weights.float()
            masses[li] = float(weights.sum().item()) if person_only is None else (
                float(self._motion_mass[g]) if person_only else 1.0
            )
            # For region background, give every masked image equal image-level mass.
        return cells, pmf, masses

    def _motion_indices(self, n, num_images, image_height, image_width, device) -> Optional[Tensor]:
        cells, pmf, masses = self._local_motion_views(num_images, device, person_only=None)
        if masses.sum() <= 0:
            return None
        counts, _ = self._image_level_counts(masses.to(device), n, device)
        # Motion cells are mask cells -> footprint = image / mask resolution.
        # Footprint differs per image; handle per-image inside the draw via shape.
        return self._draw_motion_pixels(cells, pmf, counts, num_images, image_height, image_width, device)

    def _draw_motion_pixels(self, cells, pmf, counts, num_images, image_height, image_width, device) -> Tensor:
        globals_per_local = self._batch_global_ids(num_images, device).tolist()
        out = []
        for li in torch.nonzero(counts, as_tuple=False).flatten().tolist():
            k = int(counts[li].item())
            c_cells = cells.get(li)
            c_pmf = pmf.get(li)
            if c_cells is None or c_pmf is None or c_cells.shape[0] == 0:
                r = torch.randint(0, image_height, (k,), device=device)
                cc = torch.randint(0, image_width, (k,), device=device)
            else:
                hm, wm = self._motion_shape[globals_per_local[li]]
                sh = image_height / float(hm)
                sw = image_width / float(wm)
                sel = torch.multinomial(c_pmf.to(device), k, replacement=True)
                cy = c_cells[sel, 0].to(device).float()
                cx = c_cells[sel, 1].to(device).float()
                r = (cy * sh + torch.rand(k, device=device) * sh).long()
                cc = (cx * sw + torch.rand(k, device=device) * sw).long()
                r = torch.clamp(r, 0, image_height - 1)
                cc = torch.clamp(cc, 0, image_width - 1)
            out.append(torch.stack([torch.full((k,), li, dtype=torch.long, device=device), r.long(), cc.long()], dim=1))
        if not out:
            return torch.empty((0, 3), dtype=torch.long, device=device)
        return torch.cat(out, dim=0)

    # ------------------- region mode: FAS within person/background -------------
    def _region_freq_views(self, num_images, device, person: bool):
        """Per local image, split the FAS freq pmf by whether each patch cell falls
        inside the person mask. Returns local-keyed cells/pmf/masses restricted to
        the requested region. Cached per global id."""
        globals_per_local = self._batch_global_ids(num_images, device).tolist()
        cells, pmf, masses = {}, {}, torch.zeros(num_images, dtype=torch.float64)
        for li, g in enumerate(globals_per_local):
            if g not in self._freq_cells:
                continue
            in_cells, in_pmf, out_cells, out_pmf = self._region_split_for_global(g)
            if person:
                cc, pp = in_cells, in_pmf
            else:
                cc, pp = out_cells, out_pmf
            if cc is None or cc.shape[0] == 0:
                continue
            cells[li] = cc
            pmf[li] = pp
            masses[li] = float(pp.sum().item())  # ~1 each -> equal image weight when present
        return cells, pmf, masses

    def _region_split_for_global(self, g: int):
        """Split image g's freq cells into in-mask / out-of-mask (cached)."""
        if g in self._region_split_cache:
            return self._region_split_cache[g]
        fcells = self._freq_cells[g]
        fpmf = self._freq_pmf[g]
        if g not in self._motion_shape:
            # No mask -> treat all freq cells as background.
            res = (
                torch.empty((0, 2), dtype=torch.long), torch.empty((0,)),
                fcells, (fpmf / fpmf.sum()).float() if fpmf.numel() else fpmf,
            )
            self._region_split_cache[g] = res
            return res
        hm, wm = self._motion_shape[g]
        raw = self._motion_cells[g].reshape(hm, wm)
        # Map each freq patch cell (patch grid coords) to a mask cell via the patch
        # stride and the image/mask scale. Use the cell's center pixel.
        ish = self.image_shapes.get(g)
        if ish is None:
            # Fall back: assume mask res == patch grid res mapping by ratio.
            ih = (fcells[:, 0].max().item() + 1) * self.patch_stride
            iw = (fcells[:, 1].max().item() + 1) * self.patch_stride
        else:
            ih, iw = ish
        py = (fcells[:, 0].float() * self.patch_stride + self.patch_size / 2.0)
        px = (fcells[:, 1].float() * self.patch_stride + self.patch_size / 2.0)
        my = torch.clamp((py * hm / float(ih)).long(), 0, hm - 1)
        mx = torch.clamp((px * wm / float(iw)).long(), 0, wm - 1)
        inside = raw[my, mx] > 0.5
        in_cells = fcells[inside]
        out_cells = fcells[~inside]
        in_pmf = fpmf[inside]
        out_pmf = fpmf[~inside]
        in_pmf = (in_pmf / in_pmf.sum()).float() if in_pmf.sum() > 0 else in_pmf
        out_pmf = (out_pmf / out_pmf.sum()).float() if out_pmf.sum() > 0 else out_pmf
        res = (in_cells.long(), in_pmf, out_cells.long(), out_pmf)
        self._region_split_cache[g] = res
        return res

    def _region_indices(self, n, num_images, image_height, image_width, device, person: bool) -> Tensor:
        if n <= 0:
            return torch.empty((0, 3), dtype=torch.long, device=device)
        cells, pmf, masses = self._region_freq_views(num_images, device, person)
        if masses.sum() <= 0:
            # No support for this region -> fall back to plain FAS.
            return self._freq_indices(n, num_images, image_height, image_width, device)
        counts, _ = self._image_level_counts(masses.to(device), n, device)
        idx = self._draw_pixels_from_cell_pmf(
            cells, pmf, counts, self.patch_stride, self.patch_stride, image_height, image_width, device
        )
        if idx.shape[0] < n:
            extra = self._freq_indices(n - idx.shape[0], num_images, image_height, image_width, device)
            idx = torch.cat([idx, extra], dim=0)
        return idx[:n]

    # ============================================================ main sampler
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if self.config.mode == "off":
            if not self.is_initialized or not self.config.enable_fas or not self._freq_cells:
                return super().sample_method(batch_size, num_images, image_height, image_width, mask, device)
            return self._freq_indices(batch_size, num_images, image_height, image_width, device)

        if not self.is_initialized:
            return super().sample_method(batch_size, num_images, image_height, image_width, mask, device)

        if self.config.mode == "split":
            indices = self._sample_split(batch_size, num_images, image_height, image_width, device)
        elif self.config.mode == "region":
            indices = self._sample_region(batch_size, num_images, image_height, image_width, device)
        else:
            raise ValueError(f"Unknown MotionFASPixelSampler mode: {self.config.mode!r}")

        perm = torch.randperm(indices.shape[0], device=device)
        return indices[perm]

    def _has_batch_motion(self, num_images, device) -> bool:
        for g in self._batch_global_ids(num_images, device).tolist():
            if g in self._motion_pmf:
                return True
        return False

    def _sample_split(self, batch_size, num_images, image_height, image_width, device) -> Tensor:
        u = max(self.config.uniform_frac, 0.0)
        f = max(self.config.freq_frac, 0.0)
        m = max(self.config.motion_frac, 0.0)
        total = u + f + m
        if total <= 0:
            u, f, m, total = 1.0, 0.0, 0.0, 1.0

        motion_scale = self._active_motion_strength()
        motion_target = (m / total) * motion_scale
        if not self._has_batch_motion(num_images, device):
            motion_target = 0.0
        remainder = 1.0 - motion_target
        uf_total = u + f
        uniform_p = remainder if uf_total <= 0 else remainder * (u / uf_total)

        n_motion = int(round(batch_size * motion_target))
        n_uniform = int(round(batch_size * uniform_p))
        n_freq = batch_size - n_motion - n_uniform
        if n_freq < 0:
            n_uniform = max(n_uniform + n_freq, 0)
            n_freq = batch_size - n_motion - n_uniform

        parts: List[Tensor] = []
        if n_uniform > 0:
            parts.append(
                PixelSampler.sample_method(self, n_uniform, num_images, image_height, image_width, None, device).long()
            )
        if n_freq > 0:
            parts.append(self._freq_indices(n_freq, num_images, image_height, image_width, device))
        if n_motion > 0:
            mi = self._motion_indices(n_motion, num_images, image_height, image_width, device)
            if mi is None or mi.shape[0] < n_motion:
                fill = self._freq_indices(
                    n_motion - (0 if mi is None else mi.shape[0]), num_images, image_height, image_width, device
                )
                mi = fill if mi is None else torch.cat([mi, fill], dim=0)
            parts.append(mi[:n_motion])
        return torch.cat(parts, dim=0)

    def _sample_region(self, batch_size, num_images, image_height, image_width, device) -> Tensor:
        pf = float(np.clip(self.config.person_frac, 0.0, 1.0))
        scale = self._active_motion_strength()
        if not self._has_batch_motion(num_images, device) or scale <= 0:
            return self._freq_indices(batch_size, num_images, image_height, image_width, device)

        eff_pf = pf * scale
        n_person = int(round(batch_size * eff_pf))
        n_bg = batch_size - n_person

        parts: List[Tensor] = []
        if n_person > 0:
            parts.append(self._region_indices(n_person, num_images, image_height, image_width, device, person=True))
        if n_bg > 0:
            parts.append(self._region_indices(n_bg, num_images, image_height, image_width, device, person=False))
        return torch.cat(parts, dim=0)

    # ================================================================== entry
    def sample(self, image_batch: Dict):
        if not self.is_initialized:
            ds = self.dataset if self.dataset is not None else (
                image_batch if isinstance(image_batch, Dataset) else None
            )
            if ds is None:
                CONSOLE.print(
                    "[yellow]MotionFASPixelSampler:[/yellow] dataset unavailable; falling back to uniform."
                )
                return PixelSampler.sample(self, image_batch)
            self._initialize_buckets(ds)
            self._build_freq_pmfs()
            self._load_person_masks(ds)
        else:
            if not self._fas_built:
                self._build_freq_pmfs()
            if not self._motion_loaded and self.dataset is not None:
                self._load_person_masks(self.dataset)

        self._set_batch_context(image_batch)
        self.current_fas_strength = self._active_fas_strength()
        batch = PixelSampler.sample(self, image_batch)
        self.sample_count += 1
        return batch
