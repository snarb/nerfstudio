"""Explicit visibility adaptation; original metric math is used for present actors."""
import math


def actor_visibility(row,actor,mask):
    entry=row.get('visibility',{}).get(actor)
    status=entry.get('status') if isinstance(entry,dict) else entry
    status=str(status).lower() if status is not None else ('present' if mask.any() else 'unknown')
    if status not in ('present','absent'):raise ValueError('UNKNOWN fighter identity/visibility must be reviewed: '+row['camera']+' '+actor)
    crop=row['crops'][actor]
    if status=='absent':
        review=entry if isinstance(entry,dict) else row.get('visibility_review',{}).get(actor,{})
        reviewed=(review.get('reviewed') is True and bool(review.get('evidence')))
        if not reviewed:raise ValueError('Actor absence requires an explicit reviewed evidence reference')
        if mask.any() or crop is not None:raise ValueError('Reviewed absence must have empty mask and null crop')
    elif not mask.any() or crop is None:raise ValueError('PRESENT actor requires nonempty mask and fixed crop')
    return status


def aggregate_visible(records,ready):
    from marshawn_tuning_metrics import aggregate_actor_metrics,DOMAIN
    views=tuple(ready['validation']);actors=('black','grey')
    expected={(v,a) for v in views for a in actors}
    rows=list(records);keys={(r['view'],r['actor']) for r in rows}
    if len(keys)!=len(rows) or not keys or not keys<=expected:raise ValueError('Invalid visible monitor pairs')
    if keys==expected:
        result=aggregate_actor_metrics(rows,expected_views=views,expected_actors=actors)
    else:
        fields=('psnr','ssim','lpips','validation_loss')
        for row in rows:
            if row['domain']!=DOMAIN:raise ValueError('Metric domain changed')
            if any(not math.isfinite(float(row[k])) and not(k=='psnr' and row[k]==math.inf) for k in fields):
                raise ValueError('Nonfinite visible-pair metric')
        by_actor={}
        for actor in actors:
            selected=[r for r in rows if r['actor']==actor]
            if not selected:raise ValueError('Each fighter needs a visible monitor; qualify another TRAIN camera: '+actor)
            by_actor[actor]={k:sum(float(r[k]) for r in selected)/len(selected) for k in fields}
        by_view={}
        for view in views:
            selected=[r for r in rows if r['view']==view]
            if selected:by_view[view]={k:sum(float(r[k]) for r in selected)/len(selected) for k in fields}
        result=dict(metrics={k:sum(r[k] for r in by_actor.values())/len(actors) for k in fields},
                    by_actor=by_actor,by_view=by_view,records=rows,domain=DOMAIN,
                    aggregation='equal visible views within each actor, then equal actors; absent pairs excluded')
    result.update(monitoring='in-sample; all66 cameras train',visible_pair_count=len(rows),
        visible_cameras_by_actor={a:sorted(r['view'] for r in rows if r['actor']==a) for a in actors},
        cross_frame_metric_comparability='not comparable when visible monitor pair set changes')
    return result
