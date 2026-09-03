# Задача для новой Codex-сессии: COLMAP PatchMatch-TSDF для первых 50 кадров DEC5 5A-3

Работай автономно до получения полной, проверенной кампании. Сначала прочитай
`LookCloser/AGENTS.md`, затем `LookCloser/README.md` и разделы `PatchMatch ear-artifact ablation`
в `LookCloser/experiments/dec5_000899_offtheshelf_geometry.md`. Не скачивай статью LookCloser:
если она понадобится, используй локальный `Paper LookCloser.md`.

## Цель

Для первых 50 временных кадров сцены DEC5 5A-3 построить по одной реконструкции
`fixed-pose COLMAP PatchMatch -> TSDF mesh`, сохранить один held-out eval-рендер на кадр,
посчитать только face-ROI PSNR/SSIM/LPIPS и собрать общий CSV: одна строка на временной кадр.
После этого визуально проверить eval-рендеры относительно GT и убедиться, что на ухе и помаде
нет геометрических осколков, мозаики, двоения или размытия. Фон не является целевым объектом.

Это не обучение NeRF и не требует Splatfacto на каждом кадре. Итоговый временной 3D-артефакт —
извлечённый из TSDF mesh (`.ply`) плюс manifest. Open3D VoxelBlockGrid сейчас отдельно не
сериализуется; не называй mesh сохранённым сырым TSDF volume.

## Зафиксированный лидер, который надо воспроизвести

Код находится в коммите `dc913800` (`Add reusable COLMAP PatchMatch TSDF pipeline`):

- `LookCloser/scripts/run_colmap_patchmatch_tsdf.py` — основной runner;
- `LookCloser/scripts/export_nerfstudio_colmap_model.py` — экспорт фиксированных камер;
- `LookCloser/scripts/fuse_depth_tsdf_mesh.py` — CUDA TSDF;
- `LookCloser/scripts/render_tsdf_mesh_depth.py` и
  `LookCloser/scripts/render_mesh_image_blend.py` — raycast и hard texture mapping;
- `LookCloser/scripts/build_angular_camera_subset.py` — выбор texture-камер только по геометрии.

Принятый рецепт:

- геометрия: все 62 train-камеры;
- ровно одна eval-камера: physical camera `F004_B005_1210O9`, исходный файл
  `frame_eval_00001.exr`; две другие исходные eval-камеры не превращать в train;
- фиксированные GLOMAP poses и intrinsics, без feature matching, bundle adjustment или pose
  optimization на каждом временном кадре;
- COLMAP PatchMatch в два прохода: photometric, затем geometric;
- `image_size=1920`, `source_count=12`, три итерации на проход;
- depth range `4.5..20.0`, geometric consistency gates `6/2`, NCC `0.1`, minimum two
  consistent views, triangulation angle `1 degree`;
- tensor TSDF: voxel `0.0005`, truncation `0.004`, extraction weight `2`, depth truncation `4`,
  normalized crop `[-0.15, 0.15]^3`;
- disconnected-component threshold:
  `max(100 triangles, 0.002 * largest_component_triangles)`;
- texture: 16 angular cameras, selected from calibration only;
- RGB: hard `nearest-fill`, никакого усреднения источников, U-Net или LPIPS-refiner;
- маленькие согласованные screen-space depth holes можно заполнять текущим plane-fit правилом
  runner (`max_area=1000`); цвет всё равно берётся только перепроекцией train RGB;
- никаких person/face/background masks ни на одном этапе.

Проверенный кадр `000899` дал 62/62 full-resolution geometric depth maps, mean coverage
`0.3851568`, minimum per-camera coverage `0.2672078`, и один connected mesh component после
фильтра. На фиксированной диагностической поверхности left-ear метрика улучшилась с
`21.9858 / 0.7312 / 0.1448` до `24.8845 / 0.6998 / 0.1000` PSNR/SSIM/LPIPS, а видимый
отдельный осколок под ухом исчез. Эту конфигурацию заморозить для 50 кадров: не тюнить параметры
по отдельным кадрам.

Референсные артефакты кадра `000899`:

`/mnt/data/lookcloser_dec5_5a3_final/000899_colmap_patchmatch_tsdf_selected`

## Входы и кадры

Родительский immutable EXR dataset:

`/mnt/data/dec5_5a3_nerfstudio_exr_1920x1080`

Выбрать первые 50 директорий с шестизначным числовым именем в числовой сортировке. Ожидаемый
список начинается `000899, 000901, 000903, ...` и заканчивается `000997`. Перед запуском
записать точный список в campaign manifest и проверить, что он содержит ровно 50 уникальных
кадров. В каждом source dataset ожидается 65 EXR: 62 `frame_train_*` и три `frame_eval_*`.
Source EXR и их `transforms.json` не изменять.

### Фиксированная GLOMAP-калибровка

Риг физически фиксирован, поэтому один набор GLOMAP extrinsics/intrinsics надо переносить на все
временные кадры по уникальному полю `physical_camera`. Не запускать новый GLOMAP/SfM для каждого
кадра.

Основной calibration template:

`/home/brans/lookcloser_temp/dec5_000899_full65_glomap_pose_intrinsics_jpeg/transforms.json`

Ожидаемый SHA-256: `79a91edfd8b441df1ff229839e2cc5f0b861ebe3fd626f40b04280d76a5f3900`.

Достаточный 63-camera fallback (62 train + выбранная eval):

`/home/brans/lookcloser_temp/dec5_000899_eval1_angular62_glomap_pose_intrinsics_jpeg/transforms.json`

Ожидаемый SHA-256: `b0bb757d75d48f13ac36b7fd492c672b3b302aab947d82870d0dd887ba1aa638`.

Та же 63-camera калибровка имеется на `dev3` в
`/home/ubuntu/lookcloser_mvs/dec5_000899_glomap_fixed/data/transforms.json`.

В начале кампании скопировать выбранный template в постоянный config-каталог output root и
записать его hash. Для нового JPEG-кадра сохранять его `file_path`, RGB и frame identity, но
копировать из template по `physical_camera` поля `transform_matrix`, `fl_x`, `fl_y`, `cx`, `cy`,
`w`, `h`, `k1`, `k2`, `p1`, `p2`, `camera_model`. Требовать уникальное и полное соответствие
62 train-камер плюс `F004_B005_1210O9`; любое несовпадение должно останавливать кадр до запуска
PatchMatch. Записать explicit `train_filenames` (62) и `val_filenames`/`test_filenames` (одна и
та же выбранная eval). Не включать `J004_D005_1210TA` и `L004_B005_12106A` ни в train, ни в eval
этой первой кампании.

## Временный JPEG ingest

COLMAP и текущий renderer должны получать JPG, не EXR. Используй существующий
`LookCloser/scripts/convert_exr_nerfstudio_to_jpeg.py` как основу, но сделай preprocessing
resumable и пригодным для последовательной обработки parent dataset. Для воспроизведения
принятого `000899` используй display transform
`global exposure -> Reinhard -> sRGB`, `middle_gray=0.18`, JPEG quality 95, 4:4:4 и
`exposure_mode=per-image`. Сохраняй для аудита gain каждого изображения. Не добавляй новую
цветокоррекцию.

JPG — только временный рабочий формат. Обрабатывай один временной кадр за раз, чтобы не делать
постоянную 50x65 JPEG-копию. Сначала создай и проверь локальный staged dataset, затем передай его
на GPU host. Удаляй локальный/remote scratch только после возврата и проверки финальных hashes.

## Где выполнять PatchMatch

Проверенный binary находится на `dev3`:

`/usr/local/bin/colmap`

Он обязан сообщать `COLMAP 3.13.0.dev0`, commit `5509fffe`, `with CUDA`. Репозиторий и environment
на `dev3`: `/home/ubuntu/repos/nerfstudio` и
`/home/ubuntu/anaconda3/envs/nerfstudio`. Доступ: `ssh ubuntu@dev3`.

Не использовать локальный COLMAP 4.1.1: на тех же данных он дал около 7% coverage. Не использовать
другой packaged 3.13: ранее он дал duplicated photometric и пустые geometric maps. Если точный
binary недоступен, восстанови/собери именно проверенный commit с CUDA либо остановись с ясным
blocker; не обходи pin через `--allow-unverified-colmap-build` без отдельного depth-map canary.

Так как source EXR root находится на `clever-shadow`, а проверенный GPU binary — на `dev3`,
передавай на `dev3` только staged JPG dataset одного кадра. После завершения верни на
`clever-shadow` финальные mesh/render/manifests/compact logs и только затем очищай remote scratch.
Не запускай два PatchMatch job на одной GPU одновременно. Если найдутся несколько свободных GPU,
разрешена параллельность строго один frame worker на GPU, с отдельными workspace и `gpu_index`.

## Подзадача 1: реализация и первые quality gates

Добавь отдельный opt-in controller, например
`LookCloser/scripts/run_colmap_patchmatch_tsdf_campaign.py`. Он не должен менять defaults
LookCloser, Nerfacto, Splatfacto или существующего single-frame runner. Требования:

1. Числовая discovery первых 50 кадров и immutable `campaign_request.json` с hashes конфигурации,
   calibration template, source transforms и используемых scripts.
2. Atomic/resumable state на каждый кадр. Частично записанный output нельзя считать complete.
   Resume разрешён только при полном совпадении request/config hashes.
3. Последовательность на кадр: временный JPG -> применение фиксированной calibration по
   `physical_camera` -> explicit 62/1 split -> single-frame runner -> face metrics -> копирование
   постоянных артефактов -> checksum validation -> cleanup scratch.
4. Сохранять RGB prediction только для единственного eval view. Train RGB renders не нужны.
   Внутренние train-camera mesh-depth карты допустимы для visibility checks, но после успешной
   публикации их можно удалить вместе с dense PatchMatch scratch.
5. На каждый кадр постоянно сохранять:
   - `mesh/colmap_patchmatch_tsdf.ply` и JSON metadata;
   - `render/eval_pred_0000.png` и, желательно, linear/display EXR, который породил PNG;
   - eval GT или неизменяемую ссылку с hash;
   - source-selection image и reprojection audit;
   - compact stage logs, pipeline request/manifest и per-frame result JSON;
   - face metrics;
   - visual-review crops и verdict после проверки.
6. Перед публикацией проверить: 62 geometric maps, единственную форму depth map `1080x1920`,
   finite positive coverage, mean/min coverage, непустой mesh, component statistics, finite
   1920x1080 render и hashes всех retained outputs.

Сначала выполни кадры `000899`, `000901`, `000903` и остановись на обязательный visual gate.
Для `000899` разрешено переиспользовать уже опубликованные mesh/render только после проверки
manifest и SHA-256; face-ROI metrics всё равно пересчитать по новому единому протоколу. Если три
первых кадра не демонстрируют одну и ту же качественную поверхность уха/помады, не запускай ещё
47 кадров: локализуй системную причину. Не делай per-frame ручных исключений.

## Face-only метрики

Считать display-domain PSNR, SSIM и LPIPS; loss не включать. Метрики считать только в фиксированном
прямоугольнике лица из committed файла
`LookCloser/experiments/assets/dec5_000899_face_roi.json`, то есть `[650, 400, 1150, 900]`.
Не считать full-frame/room/actor aggregate. Не использовать candidate-defined surface mask:
пропавшая часть лица или чёрная дыра внутри face ROI должна ухудшать метрику, а не выпадать из неё.
Person/face segmentation mask также запрещена. GT — только held-out RGB камеры
`F004_B005_1210O9`; он не должен участвовать в PatchMatch, TSDF, source choice или prediction.

При необходимости добавь новый post-hoc ROI scorer или обратно совместимый opt-in metric mode в
runner. Existing defaults должны остаться без изменений. Зафиксируй, что новый ROI-only протокол
не численно идентичен старой `000899` таблице, где применялась независимая Splatfacto surface mask;
для temporal regression baseline используй пересчитанный кадр `000899` из этой кампании.

Главный CSV:

`/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50/metrics.csv`

Он должен содержать ровно одну строку на source frame и как минимум поля:

`frame_id, source_dataset, eval_physical_camera, train_camera_count, texture_camera_count,
face_psnr, face_ssim, face_lpips, depth_coverage_mean, depth_coverage_min,
mesh_vertices, mesh_triangles, mesh_components, render_path, mesh_path,
render_sha256, mesh_sha256, metric_status, visual_status, ear_artifact,
lipstick_artifact, visual_notes, status`.

Писать CSV атомарно и в строгом порядке выбранных 50 кадров. Не допускать duplicated rows.
До visual review допустим `visual_status=pending`; после финальной проверки pending не должно
остаться. Любой non-finite metric — fail. После первых трёх кадров зафиксировать baseline и
data-driven regression gates. Как начальный сигнал использовать падение PSNR больше 1 dB,
SSIM больше 0.03 или рост LPIPS больше 0.05 относительно медианы предыдущих принятых кадров,
но проверить фактический разброс и записать окончательные thresholds в campaign manifest.
Metric flag не заменяет визуальную проверку.

## Визуальная проверка LLM

Для каждого кадра сравнить prediction и held-out GT. Сохранять side-by-side crops минимум для:

- face/ear/hair: `[650, 400, 1150, 950]`, с отдельным 1:1 ear crop
  `[820, 740, 1060, 950]`;
- lipstick/lips/hand: `[687, 540, 987, 800]`;
- уменьшенный actor overview `[0, 150, 1400, 1030]` только для контекста, не для метрик.

Контакт-листы можно собирать по 10 кадров, но ear и lipstick crops надо смотреть в достаточном
разрешении, а не только как маленькие thumbnails. Для каждого кадра записать JSON verdict и затем
перенести его в CSV:

- `pass`: ухо, серьга, волосы, губы и помада имеют единственную согласованную форму;
- `fail`: detached/duplicated triangle, skin/hair island, мозаика, shear, ghost, source seam,
  чёрная дыра, проходящая через объект, или заметно более мутная помада/ухо относительно GT;
- `uncertain`: недостаточно уверенности — требует отдельного просмотра, не превращать в pass.

Игнорировать отсутствие комнаты и открытый background вне человека. Не игнорировать чёрные дыры,
которые режут силуэт, ухо, волосы, руку или помаду. Визуальный fail нельзя отменить хорошей
средней метрикой.

После initial gate проверять и фиксировать результаты батчами, не оставляя 50 visual verdicts на
самый конец. При единичном сбое можно один раз повторить тот же frozen recipe в чистом workspace,
чтобы исключить stale state. Нельзя менять TSDF/PatchMatch/source параметры только для одного
кадра. Если одинаковый дефект возникает минимум на двух кадрах, остановить новые запуски,
сформулировать наиболее вероятную системную гипотезу, проверить её и только затем делать общий,
обратно совместимый fix с тестом и rerun всех затронутых кадров.

## Надёжность и наблюдение

Кампания займёт много часов. Detached process — только механизм живучести, не завершение задачи.
Пока job работает, не реже раза в час проверяй controller/worker PID, текущий stage/frame,
количество depth maps, GPU process/memory, OOM/CUDA errors, свободное место и последний compact
status. Записывай проверки в `campaign_checks.jsonl`. После завершения кадра сразу продолжай
metrics/visual gate. Не оставляй job без активного supervising workflow.

Не удаляй source data, reference `000899` или чужие файлы. Чистить можно только точно известный
campaign scratch после checksum-верификации; failed workspace сохранять до диагностики либо
перемещать в отдельный quarantine.

## Итоговые артефакты и критерий готовности

Output root:

`/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50`

Ожидаемая структура:

```text
campaign_request.json
campaign_manifest.json
campaign_checks.jsonl
metrics.csv
contact_sheets/
frames/
  000899/
    mesh/
    render/
    metrics.json
    visual_review.json
    result.json
  ...
  000997/
```

Готово только когда:

1. опубликованы ровно 50 ordered frame results и 50 CSV rows;
2. у каждого кадра есть проверенные mesh, eval prediction, face PSNR/SSIM/LPIPS и hashes;
3. нет `pending`/`uncertain`, либо проблемные кадры явно помечены `fail` и диагностированы — не
   скрыты из CSV;
4. LLM просмотрел все ear/lipstick crops, initial gate и contact sheets сохранены;
5. написан `LookCloser/experiments/dec5_patchmatch_tsdf_50f.md` со структурой What was tested / Results / Insights,
   таблицей распределения метрик, списком worst frames и ссылками на contact sheets;
6. campaign audit повторно проверяет frame inventory, CSV, manifests, hashes и отсутствие
   full-frame metrics;
7. новый controller/scorer/tests/docs закоммичены отдельным целевым commit без захвата чужих
   dirty-worktree изменений и без изменения defaults существующих моделей.

В финальном ответе дай краткий статус, aggregate face PSNR/SSIM/LPIPS (min/median/max), worst
frames, число visual pass/fail, путь к output root и SCP-команды на скачивание `metrics.csv`,
contact sheets и всей папки. Если хотя бы один кадр failed, не называй кампанию полностью
успешной.
