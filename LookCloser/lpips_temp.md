# Summary

Проверены 20 опубликованных кадров `000899..000937`, controlled re-score `000941`, retained EXR/source-selection audits, обе маски и immutable snapshot scorer. Кампания, GPU, controller, outputs и code/config artifacts не изменялись; производные вычисления выполнялись только на CPU в `/tmp`.

**Финальный diagnosis:** системный рост опубликованного face LPIPS — прежде всего **metric-definition bug в ROI v1**, а не деградация лица и не numerical/scorer implementation bug. Координаты polygon и растровая маска были буквально одинаковы для `000899`, `000917`, `000937` и старого control `000941`, хотя лицо/челюсть двигались. Metadata при этом декларировала `neck`, `room`, `hair`, `clothing` excluded, но старая внешняя граница включала меняющийся клин шеи/внешнего силуэта под jawline.

Контроль на `000941` причинно изолирует ROI: **те же prediction и GT hashes** получили:

| ROI | PSNR ↑ | SSIM ↑ | LPIPS ↓ | fraction кадра |
|---|---:|---:|---:|---:|
| старая статичная ROI v1 | 19.8437 | 0.807832 | 0.141004 | 5.2572% |
| GT-only ROI v2 вдоль jawline | 27.4698 | 0.868682 | 0.061396 | 4.7003% |
| изменение | **+7.6261 dB** | **+0.060850** | **−0.079608** | −0.5569 pp |

Новая маска — строгий поднабор старой. Она удаляет 11,548 px; этот удалённый neck/silhouette wedge на 75.17% состоит из invalid/black prediction и даёт **84.56% всей squared error старой ROI**. Внутри исправленной face ROI осталось только **15 invalid pixels из 97,466 (0.0154%)**. Следовательно:

- реальный geometry-support defect есть в neck/external-silhouette wedge;
- этот wedge **вне intended corrected face ROI** и не должен определять face LPIPS;
- для лица `000941` текущая prediction имеет практически полный support и хорошие метрики;
- старые `face_*` значения и regression flags нельзя использовать для выбора TSDF/config, пока весь ряд не пересчитан с ROI v2.

Это уточняет и отменяет прежний основной вывод о face mesh regression: scorer правильно посчитал неправильную population. Scorer arithmetic исправлять не нужно; нужно исправить ROI authoring/versioning и затем отдельно измерять actor/silhouette support.

# Evidence

## 1. Controlled same-prediction comparison

Обе оценки `000941` имеют одинаковые:

- prediction SHA-256: `e367813417848b5f19a886b5b342301473435cd4e7bcaab14c55b9b5945f0c54`;
- GT SHA-256: `2f4d9eecc5db4ab612f73bce81c5ed9bda8c416030342eb9579812492958f8e8`;
- scorer/protocol: display RGB `[0,1]`, exact selected-pixel PSNR, Alex-LPIPS/SSIM на tight bbox с одинаковым zero outside mask;
- source selection, mesh, render и exposure.

Единственная причинная переменная — GT-only polygon. ROI v2 declares `prediction_used_for_selection=false`, привязана к GT hash и сохранённая `face_mask.png` побитово совпадает с независимой rasterization polygon.

Проверенные control artifacts: old receipt `/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/.diagnostics/support_sweep_000941/control_w2/score/metrics.json`; ROI-v2 polygon `/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/.diagnostics/support_sweep_000941/face_polygons_000941_v2.json`; new receipt и GT-only overlay `/tmp/score_face_v2/metrics.json`, `/tmp/score_face_v2/face_mask_overlay.png`.

Независимый CPU-пересчёт дал old LPIPS `0.141003713` точно и new LPIPS `0.061391313` против receipt `0.061395712` (разница `4.40e-6`, ожидаемая CPU/GPU численная погрешность).

## 2. Непересекающееся разложение старой ROI

| region в `000941` | pixels | invalid/black | invalid fraction | MSE | PSNR | доля old-ROI squared error |
|---|---:|---:|---:|---:|---:|---:|
| corrected face ROI v2 | 97,466 | 15 | **0.0154%** | 0.001791 | 27.470 | 15.44% |
| old-only удалённый wedge | 11,548 | 8,681 | **75.17%** | 0.082747 | 10.822 | **84.56%** |
| old ROI v1 total | 109,014 | 8,696 | 7.977% | 0.010366 | 19.844 | 100% |

Новая ROI полностью содержится в старой (`new_only=0`). Средний GT RGB удалённого wedge `[0.3845, 0.2901, 0.2360]`, а mean prediction `[0.0759, 0.0598, 0.0507]`: это действительно видимая neck/skin/silhouette область с реальной нехваткой geometry support, а не numerical шум. Но `excluded_anatomy_or_objects` во всех ROI v1 прямо содержит `neck`, `room`, `hair`, `clothing`, поэтому включение этой области нарушало собственное определение face population.

## 3. Улучшение не является bbox/zero-boundary эффектом

- New ROI tight bbox: `[710,475,1141,796]`, LPIPS `0.061391`.
- Та же new mask в **старом bbox** `[710,475,1141,836]`: LPIPS `0.056953`.
- Старая mask в старом bbox, но с oracle correction только удалённого wedge: LPIPS `0.057124`.

Укороченный tight bbox фактически повышает corrected LPIPS примерно на `0.00444`, а не искусственно понижает его. Почти равные `0.056953` и `0.057124` при одинаковом старом bbox показывают, что падение `0.141→≈0.057` вызвано содержимым ошибочного wedge. PSNR, который вообще не зависит от bbox/zero context, одновременно улучшается на 7.63 dB.

## 4. Static-mask lineage нарушала семантику движущегося лица

- SHA-256 растровой mask у `000899`, `000917`, `000937` и old-control `000941` один и тот же: `e0c8d67ba0bbb000217275e3744a72fc003ed1ce15f52ed42ec9f63f8e110717`.
- Include/exclude coordinates у опубликованных `000899`, `000917`, `000937` буквально одинаковы. Менялись только GT hash и notes.
- Old `000941` receipt прямо говорит `inherited ... stable polygon`; его raster mask побитово идентична mask `000899`.
- Overlay v1 показывает клин ниже jawline до `y≈835`; ROI v2 следует видимой jawline и заканчивается на `y=796`, сохраняя GT-only exclusions hand/tube.

Техническая стабильность mask здесь была анти-признаком: population не следовала движущейся anatomical boundary. GT hash binding доказывал, к какому изображению относится JSON, но не доказывал, что polygon был заново семантически проверен на этом GT.

## 5. Retrospective attribution прежнего 20-frame тренда

Для диагностики форму ROI v2, нарисованную на GT `000941` без prediction, применили к прежним 20 EXR. Это **не официальный backfill**: из-за движения для финальной метрики каждый кадр всё равно требует собственной GT-only трассировки. Но контроль позволяет проверить spatial attribution старого тренда.

| frame | official ROI-v1 LPIPS | diagnostic v2-shape LPIPS | invalid внутри v2-shape | invalid в удалённом wedge | wedge share old error |
|---:|---:|---:|---:|---:|---:|
| 000899 | 0.05587 | 0.05219 | 0.811% | 0.069% | 25.3% |
| 000917 | 0.09162 | 0.05835 | 0.391% | 21.51% | 67.8% |
| 000927 | 0.11542 | 0.05451 | 0.000% | 50.44% | 81.7% |
| 000937 | 0.13824 | 0.06066 | 0.000% | 69.04% | 84.2% |

- Endpoint LPIPS growth сокращается с `+0.08237` до `+0.00847`: удаляется **89.7%** тренда.
- Mean(last 5) − mean(first 5) сокращается с `+0.06628` до `+0.00353`: удаляется **94.7%** тренда.
- Correlation с временем падает с Pearson `0.996` / Spearman `0.998` до `0.328` / `0.347`.
- Diagnostic v2-shape PSNR падает лишь `29.36→27.78 dB`, а не `28.58→20.26 dB` у ROI v1.
- Invalid support в intended-face-shaped region к поздним кадрам не растёт; одновременно invalid fraction ошибочного wedge растёт строго монотонно до 69.0%.

Эта ретроспектива вместе с controlled `000941` подтверждает, что прежний системный LPIPS slope создан движением неправильной ROI population. Остаточное `≈0.008` endpoint изменение может отражать настоящее appearance/reprojection изменение или несовпадение формы `000941` с ранними лицами; его нельзя окончательно интерпретировать до proper per-frame v2 backfill.

## 6. Что остаётся реальной geometry проблемой

Renderer действительно превращает invalid support в black, потому что при `base_prediction_exr=None` использует zero base. В старом wedge это реальный actor/neck silhouette gap, и его можно исследовать TSDF/support sweep. Однако:

- corrected `000941` face имеет 99.9846% valid support;
- dominant texture source и calibration не меняются;
- старое source/coverage ухудшение локализовано почти полностью вне corrected face;
- выбирать `tensor_weight_threshold`, `sdf_trunc` или fallback по ROI-v1 face LPIPS означало бы оптимизировать neck/silhouette, ошибочно названный face.

Если полнота всей фигуры важна, defect следует сохранять как отдельный `actor_silhouette_support`/`neck_support` signal и визуально проверять на actor overlay. Он не должен исчезнуть из QA, но не должен загрязнять face PSNR/SSIM/LPIPS.

## 7. Прочие гипотезы

Предыдущие проверки остаются валидны, но теперь вторичны:

- scorer code воспроизводится; normalize/sign/EXR loading корректны;
- padding, erosion и natural-context controls меняют абсолютную LPIPS шкалу, но не создавали старый trend;
- во всех опубликованных кадрах доминирует один texture source `E004_C005_1210YM / frame_train_00018` с около 99% valid-source share;
- relative source/eval exposure gain менялся менее чем на 0.5%, covered RGB bias был стабилен;
- общий integer shift ≤1 px и source-0 depth reprojection audit не объясняли slope.

# Root-cause ranking

1. **ROI v1 semantic-definition bug — установленная причина старого face-LPIPS тренда.** Статичная polygon включала заявленно excluded neck/silhouette wedge; controlled same-prediction re-score и 20-frame spatial attribution объясняют 85–95% наблюдаемого эффекта.
2. **Реальная локальная потеря TSDF/renderer support в excluded neck/external-silhouette wedge — установлена, но находится вне corrected face metric.** Она объясняет ошибку старой population и должна стать отдельным QA signal.
3. **Небольшое настоящее face appearance/reprojection изменение — возможно.** Corrected `000941` LPIPS `0.0614` и retrospective v2-shape ряд `≈0.052..0.061` не показывают прежней системной катастрофы, но только proper per-frame v2 rescore определит остаточный trend.
4. **BBox/zero boundary, exposure, source switching, global misalignment и LPIPS implementation — не являются причиной.** Controlled checks либо держат их неизменными, либо показывают эффект на порядки меньше ROI population error.

# Recommended canaries/next steps

## Решение для кампании

1. **Не использовать ROI-v1 `face_*` и связанные regression flags для model/config selection.** Сохранить их неизменными для аудита, но явно пометить как `face_roi_v1_invalid_semantics`; не переписывать опубликованные receipts.
2. **Создать versioned ROI-v2 backfill для всех retained predictions**, начиная с уже опубликованных `000899..000937` и control `000941`. Каждый polygon трассировать/проверять на соответствующем held-out GT вдоль видимой jawline; исключать neck, room, hair, clothing, hand и tube; prediction не открывать до фиксации polygon hash.
3. **Пересчитать PSNR/SSIM/LPIPS без реконструкции или GPU rerun.** Prediction уже retained; меняется только корректная post-hoc metric population.
4. **Заново установить regression baseline/thresholds по первым принятым ROI-v2 кадрам.** Старые medians и flags нельзя переносить между ROI versions, даже если scorer formula не меняется.
5. **Не выбирать текущие support-sweep параметры по старой face metric.** `000941` control `w2` с ROI v2 уже даёт PSNR `27.47`, SSIM `0.8687`, LPIPS `0.0614` и 99.9846% face support. TSDF sweep можно оценивать отдельно по actor/neck silhouette objective и floaters, но он больше не является первичным face fix.

GPU campaign может продолжать генерировать/retain'ить predictions, если это операционно нужно, но face acceptance следует считать provisional, пока будущие кадры не используют ROI v2 и backlog не пересчитан.

## ROI-v2 quality gates

- В receipt добавить `face_roi_schema_version=2`, явный anatomy contract и polygon hash до scoring.
- Для движущейся сцены exact duplicate polygon coordinates между последовательными кадрами должны автоматически поднимать audit flag. Повтор допускается только после отдельного GT-overlay подтверждения, что anatomical boundary действительно не сдвинулась.
- Перед scoring сохранять overlay только на GT и подтверждать, что нижняя граница следует jawline; отдельная reviewer стадия после scoring проверяет GT/pred, но не может менять polygon.
- Автоматически считать `face_invalid_support_fraction` на **полной corrected ROI**, не маскируя invalid pixels из official LPIPS. Canary threshold: warning >0.5%, hard review >1%; `000941` v2 имеет 0.0154%.
- Любой black/invalid pixel внутри corrected face overlay нельзя помечать как ignorable background без исправления ROI либо явного geometry verdict.

## Разделённые метрики

- **Official face:** PSNR, SSIM, LPIPS на полной per-frame GT-only ROI v2. Не использовать candidate-surface mask.
- **Face diagnostic:** `face_invalid_support_fraction`, dominant-source share и covered-surface LPIPS только как объясняющие canaries, не вместо official metric.
- **Actor geometry:** отдельные `actor_silhouette_support`, `neck/external-contour gap` и actor visual gate. Именно сюда относится удалённый wedge и возможный TSDF support sweep.
- **Reproducibility:** закрепить версии `torch`/`torchmetrics` и SHA-256 AlexNet/LPIPS weights; сохранять один reference-score canary с tight tolerance.

Критерий закрытия инцидента: все 20 опубликованных predictions пересчитаны с independently reviewed per-frame ROI v2; новый ряд не использует старые thresholds; `000941` same-prediction control воспроизводит `27.4698 / 0.868682 / 0.061396`; actor-silhouette defect остаётся видимым в отдельном QA signal, а не исчезает вместе с исправлением face metric.
