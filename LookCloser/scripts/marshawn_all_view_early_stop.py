"""Read-only quiet-runner adapter for the validated three-view native RGB setup.

For unit-range RGB, 10**(-mean(image_PSNR)/10) is the geometric mean of the
full-image MSEs. It is not their arithmetic mean or a resampled-ray estimate.
Only complete all-view metric rows participate; the original CSV is untouched.
The caller must already have verified the three eval views and PSNR range1.
"""
import csv
import math
from pathlib import Path


STOP_LOSS_SOURCE = 'full_view_geometric_MSE_from_mean_eval_all_PSNR_range1'


def geometric_mse_from_mean_psnr(mean_psnr):
    value = float(mean_psnr)
    if not math.isfinite(value):
        raise ValueError('Finite mean all-view PSNR required')
    try:
        result = 10.0 ** (-value / 10.0)
    except OverflowError as error:
        raise ValueError('All-view geometric MSE overflow') from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Positive finite all-view geometric MSE required')
    return result


def all_view_eval_rows(metrics_path):
    """Expose genuine full-view loss to the unchanged >= plateau comparison.

    Preserve the sampled eval loss and its status/delta in separate returned
    fields. Repeated complete rows for one step are one checkpoint observation.
    Partial writes, batch-only rows and nonfinite all-view metrics are ignored.
    """
    path = Path(metrics_path)
    if not path.exists():
        return []
    by_step = {}
    with path.open(newline='', encoding='utf-8') as stream:
        for raw in csv.DictReader(stream):
            try:
                step = int(raw['step'])
                values = [float(raw[key]) for key in
                          ('eval_all_psnr', 'eval_all_ssim', 'eval_all_lpips')]
                if step < 0 or not all(math.isfinite(value) for value in values):
                    continue
                loss = geometric_mse_from_mean_psnr(values[0])
            except (KeyError, TypeError, ValueError):
                continue
            row = dict(raw)
            row['sampled_eval_loss'] = raw.get('eval_loss', '')
            row['sampled_eval_loss_delta'] = raw.get('eval_loss_delta', '')
            row['sampled_eval_status'] = raw.get('status', '')
            row['eval_loss'] = format(loss, '.17g')
            row['eval_loss_source'] = STOP_LOSS_SOURCE
            by_step[step] = row
    result = [by_step[step] for step in sorted(by_step)]
    previous = None
    for row in result:
        current = float(row['eval_loss'])
        row['eval_loss_delta'] = '' if previous is None else format(current - previous, '.17g')
        row['status'] = ('ok' if previous is None else
                         'improving' if current < previous else 'no_improve')
        previous = current
    return result


def print_all_view_eval_row(row):
    print('eval step={step} stop_objective={source} stop_MSE={loss} '
          'sampled_MSE={sampled} psnr={psnr} ssim={ssim} lpips={lpips} '
          'delta={delta} status={status}'.format(
              step=row.get('step'), source=row.get('eval_loss_source'),
              loss=row.get('eval_loss'), sampled=row.get('sampled_eval_loss'),
              psnr=row.get('eval_all_psnr'), ssim=row.get('eval_all_ssim'),
              lpips=row.get('eval_all_lpips'), delta=row.get('eval_loss_delta'),
              status=row.get('status')), flush=True)


def configure_quiet_all_view_stop(quiet, checkpoint_selector):
    """Install only the row reader/printer and the existing native selector.

    Does not change parse_args, stop_on_no_improve, train_command, CSV writing,
    optimizer/sampler/RNG state or evaluation cadence. No training is launched.
    """
    if not callable(checkpoint_selector):
        raise TypeError('Existing native all-view PSNR/LPIPS selector required')
    for name in ('eval_rows', 'print_eval_row', 'best_eval_checkpoint'):
        if not hasattr(quiet, name):
            raise AttributeError('Unsupported quiet runner: missing ' + name)
    quiet.eval_rows = all_view_eval_rows
    quiet.print_eval_row = print_all_view_eval_row
    quiet.best_eval_checkpoint = checkpoint_selector
