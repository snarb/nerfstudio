# Handoff: новый диапазон Marshawn, рецепт native63 / hash25 / SAM15

## Задача следующей сессии Codex

Применить выбранный рецепт к другим кадрам **того же датасета** на другой машине. Порядок обязателен: проверить исходники/координаты → получить и проверить общий AABB нового диапазона → полностью проверить рецепт на одном кадре → только затем запускать очередь всех кадров. Этот документ не означает, что новый диапазон, новый AABB или финальная модель66 камер уже проверены. Старые обучения на olive-lark/dev3/silver/Clever остановлены; не возобновлять их.

Репозиторий: https://github.com/snarb/nerfstudio . Ветка сцены: `scene/marshawn1660-native63-20261109`. Базовый коммит сохранённого результата: `9d5b57ed7c60911e6fbb109b890a45736d0c7dc6`; данный handoff добавляется позднее поверх него. Общие исправления опубликованы в `main` (`279fabff`, затем индекс/документация `fbf16c20`). Не сливать целиком историческую ветку сцены в main.

```bash
git fetch origin
git switch --create marshawn-new-range --track origin/scene/marshawn1660-native63-20261109
```

Если локальная ветка уже существует, использовать её после проверки статуса; не перезаписывать чужие незакоммиченные изменения. Прочитать `LookCloser/AGENTS.md` и локальную `LookCloser/Paper LookCloser.md`.

## 1. Что именно выбрано лидером

**Обучение:** `P_B8beta99_SAM15_core001_hash25`, каждый кадр полностью с нуля,140000 реальных попыток optimizer update. **Рендеринг:** те же веса, только inference override `adaptive_max_frequency_level=10`. Во время обучения этот параметр **None**, не10 и не12. Не заменять обучение тестировавшимися сокращёнными curricula.

На кадре1660,63TRAIN/3validation: default PSNR27.228975/SSIM.590570/actor LPIPS.098840536; cap10:27.100272/.579104/.095144829. Cap9 дал меньший общий LPIPS.094086626, но ухудшил два HC лица на6–9% и ступню на7% против10. Поэтому cap9 не выбран для детализации. Hash26_100 тоже не подменяет выбранный hash25 рецепт. Полный Silver training-loop297.730558мин включает встроенную оценку, но исключает подготовку, setup и очередь; это не гарантированное время другого кадра/железа.

**Частотные карты — победитель `uniform16`: постоянный скаляр16, не индекс уровня15 и не render cap16.** Отдельную2D сеть на каждом изображении обучать не нужно. Генерировать карты в точном формате/размерах, ожидаемых native worker. Не применять learned-map fitter или gradient-rank карты по умолчанию. Runtime3D frequency grid всё равно начинается с нулей и обновляется по TRAIN глубинам; это не перенос сетки предыдущего кадра.

Сопоставимый hash25/SAM15/core контроль:32k LPIPS uniform16=.126402422, learned thresholds.90/.95/.98=.127776700/.132711335/.128082650. Сохранённый50k контроль: uniform16=.114365472, .90=.116187641, .98=.117045941. Поздних80/140 результатов новых learned-map вариантов нет: их остановил пользователь. У50k joint-кропов нет завершённой ручной visual-pass проверки. Значит свидетельство в пользу constant карты есть на32/50, но её универсальное превосходство на любых кадрах не доказано.

## 2. Источники истины и точные артефакты

Ссылки ниже открывают опубликованные файлы. Все пути внутри репозитория начинаются от `LookCloser/`:

- [Итоговые результаты, ограничения, остановленные этапы](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/marshawn_native63_final_results.md).
- [Точный выбранный рецепт](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/marshawn_native63_frozen_recipe.md).
- [Полная хронология и отрицательные результаты](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/marshawn_native63_24h.md).
- [Машиночитаемый выбор и SHA](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/recipes/marshawn1660_native63_hash25/selection.json).
- [Исходный native spec: provenance, а не готовый запуск](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/recipes/marshawn1660_native63_hash25/training_spec_native_PROVENANCE.json).
- [Terminal config: только provenance](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/recipes/marshawn1660_native63_hash25/terminal_config_native_PROVENANCE_ONLY.yml).
- [Геометрия исходного диапазона и оба4×4 преобразования](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/recipes/marshawn1660_native63_hash25/geometry.json).
- [Замороженные10 runtime-хелперов и README](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/scripts/marshawn_native63_frozen/README.md).
- [Общие выводы: frequency, sampling, occupancy](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/lookcloser_native_frequency_sampling_lessons.md).
- [AABB по маскам/идентичностям](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/marshawn_mask_derived_aabb.md).
- [Проверка occupancy](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/marshawn_native63_grid_audit.md).
- [Поправка к эффекту native crop](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/marshawn_native_crop_detail.md).
- [Сравнение со старым fight leader](https://github.com/snarb/nerfstudio/blob/scene/marshawn1660-native63-20261109/LookCloser/experiments/marshawn_vs_static_fight_recipe.md).

Точный выбранный checkpoint SHA256: `468bbf38fbcac82cb1dd681ded87688ee92ae5f4bc6035bd031cb32f48d847d3`; native training spec `4241952ed60d6c56464a81dab5aa81c19679fef6b986c419b9531aba78c001af`; terminal config `4f74fbe4d6eaaa0013b870f357e1db384c44f797fae484a915dc29d8085baf0b`. Worker `5324c6bd4e2ac68d95a6db6a88ce0e9d74662fb5882e3740e7c87b164f3e8d94`; base code seal `d873cd61ad5c00f9501f2f6c477cb42f962d4d15591f5785830e0aeae381ec14`.

**Репозиторий не является автономным runtime bundle.** Данные, веса и полный замороженный base checkout1531 файлов не включены в Git. В `scripts/marshawn_native63_frozen/` лежат10 точных overlay-хелперов, MANIFEST и BASE_SOURCE_SEAL. Они сохраняют исторические native пути и проверки источников. Нельзя просто запустить их на произвольном установленном nerfstudio или отключить guards для прохождения проверки.

На olive-lark имеются:

- База кода: `/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/code`, seal рядом `code_seal.json`.
- Кампания: `/fsx/tmp/marshawn_native63_24h_20261108`.
- Полный архив лидера: `delivery/model_archive/P_B8beta99_SAM15_core001_hash25_140k_native63/` внутри кампании, с `ARCHIVE_COMPLETE.json`.
- Выбранный raw HC: `ops/frozen_hash25_140_caps10to14_finite/actual_native/maxfreq10/HC.png`; GT: `data/uniform16/images/HC.png`. Это1464×2360 native crops, а не увеличенные4K.

Если новая машина не видит FSX, передать базу кода без обучения/весов, например выполнить **на новой машине**:

```bash
scp -r coder@olive-lark:/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/code /absolute/local/runtime_base
scp coder@olive-lark:/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/code_seal.json /absolute/local/runtime_base_seal.json
```

Создать отсутствующие родительские папки заранее. Проверить весь seal, происхождение импортов, Python/Torch/nerfacc/tiny-cuda-nn/LPIPS под реальным пользователем новой машины. Без доступа к обязательным артефактам зафиксировать конкретный блокер; не объявлять воспроизведение точным. Веса1660 можно брать только для отдельной контрольной оценки, **не** для инициализации новых кадров.

## 3. Приёмка данных нового диапазона

1. Найти новый путь, номера кадров, реальные timestamps/FPS, камеры, raster sizes, TIFF metadata/каналы/битность и solve calibration. Старый диапазон1660…2020 имел120fps и66 камер; новый нельзя считать идентичным без проверки.
2. Сопоставить camera IDs с калибровкой и кадрами; проверить пропуски, дубликаты, join, горизонтальный/вертикальный flip, rotation и pixel-center convention. На нескольких ракурсах вручную проверить цвет, лица, ступни, маски и калибровку. Не повторять Log/PQ преобразование на уже подготовленных PNG/EXR.
3. Сохранить принятую цепочку Log3G10/RWG→linear Rec709/gain4/cameraEV/radial→Reinhard+sRGB **только если формат входа тот же**. Слова raw/EXR не задают transfer function. При другом представлении требуется проверяемое преобразование к тому же training RGB; держать данные исходника неизменными.
4. Создать полноценные native actor crops без resize, обычно с исходным128px контекстом, корректно сдвинув cx/cy на integer crop origin. Не путать crop с масштабированием focal. Сохранить distortion/undistortion convention. Если crop меняется по времени, хранить его origin и преобразования для каждого кадра; фиксированный достаточный per-camera crop по диапазону предпочтителен для устойчивой геометрии.
5. Проверить original↔crop лучи: origins/directions/pixel area на corners/center и случайных пикселях совпадают в пределах обоснованной численной погрешности. Проверить реальные camera/image/cache IDs, включая shuffled/subset cache; код уже имел ошибки такого класса.
6. Новые masks/confidence/head/core/admission строить по текущим кадрам. Не переносить маски/статическую масочную геометрию1660 только потому, что calibration/box прежние. Выбирать двух актёров через явные проверенные IDs, не ranking SAM-instance. Работники/слитые/окклюдированные неопределённые detections не становятся доказанным background.

## 4. Сначала выбрать общий AABB на весь новый интервал

Старые bounds `[-.91,-.26,-.08]→[.15,.50,.63]` — только справочный старт. Они заданы **в floor-aligned box frame**, не в исходном мире. Новый диапазон может требовать другого бокса. Не менять масштаб сцены/нормализацию независимо по кадрам.

1. Зафиксировать единое world↔floor-box4×4 преобразование по принятой геометрии пола. В `geometry.json` старого рецепта есть оба преобразования. Если rig/world тот же, проверить их по новому source, а не заново наклонять box по экранной картинке.
2. Взять endpoints и кадры через примерно0.5с по всему диапазону; дополнительно экстремальные позы, шаги, выходы к краям и перекрытия с матом. Это выборка проверки, не гарантия, что движение между проверенными кадрами не выйдет наружу.
3. Получить высококачественные native masks двух актёров на всех доступных TRAIN ракурсах; head/hair coverage проверять отдельно. Использовать `scripts/mask_visual_hull_aabb.py` (также в main) для консервативной многокамерной mask-volume гипотезы. Helper требует явные два выбранных ID. UNKNOWN/неполный FOV/пересечение near plane не считать пустотой.
4. Учитывать весь projected cell footprint/8corners, не только центр. Отбрасывание работников не должно отрезать перекрытую ими ногу/волосы актёра. Если поиск клипует объём, расширить search domain, не принимать обрезанную оболочку за tight box.
5. Для каждого sampled frame получить enclosing retained-cell extents/проверенные anchors; сделать temporal union в одном floor frame, затем применить регулируемый минимум world-margin по XYZ и нижней границе, округляя наружу. Дополнительно сохранить padding в native pixels (исходный64px guard) и leaf-size uncertainty. `leaf_size`, support/contradiction и минимальный physical margin — **параметры нового bounds fit**, не универсальные константы лидера. Записать их и довести вручную до минимально достаточного бокса.
6. Проверять два человека целиком: обе ступни каждого, ладони, локти, дреды, верх головы. Для каждого контрольного ракурса сохранить исходник + корректный wireframe + вариант «снаружи бокса чёрное» + диагностику entry/exit лучей и depth при спорной ноге. См. `build_marshawn_aabb_visual_review.py`, `render_marshawn_foot_diagnostic.py`, связанные AABB отчёты. Это scene adapters: сначала проверить их calibration convention/зависимости, а не считать готовым универсальным CLI.
7. Проекция дальней грани на изображение не показывает, находится ли ступня в3D боксе. Использовать ray-box/depth и согласованность нескольких камер. Цвет wireframe не является доказательством containment.
8. По возможности исключить толстый мат/работников, но не ценой клиппинга актёров. Если человек реально лежит/наступает на мат, идеальное исключение всего мата AABB может быть невозможно. Сохранить этот факт, не рисовать ложное «чистое» подтверждение.
9. Заморозить bounds, оба4×4 преобразования, sampled frame list/timestamps, margins, source hashes и manual review с изображениями в отдельный `AABB_ACCEPTANCE.md/json`. Только после отсутствия известных actor clipping случаев допускать обучение одного кадра. Если на следующих кадрах найден выход за box — остановить очередь, исправить общий envelope и явно определить затронутые модели, а не незаметно менять координаты по кадрам.

## 5. Полный точный рецепт

Таблица ниже перенесена из замороженного рецепта. Старые hashes/пути масок и static admission — **идентичность старых артефактов**, не значения, которые надо навязать новым данным. На новом диапазоне они обязаны честно измениться при пересборке.

| Component | Frozen setting |
|---|---|
| Source frame / images |1660; original8192×4320 TIFF source; unresized native actor crops, per-camera integer crop; no training-image resize |
| Photometry |Accepted RAW Log3G10/RWG → linear Rec709, gain4/camera EV/radial correction → Reinhard+sRGB PNG; inherited preparation, not a new PQ conversion |
| Split |63 training cameras; DC/HC/KA held out; repeatedly used for validation, not an untouched test |
| Coordinate frame |Floor-aligned rigid box frame; both4×4 transforms in [geometry.json](experiments/recipes/marshawn1660_native63_hash25/geometry.json) are mandatory |
| AABB in that box frame |[-.91,-.26,-.08] → [.15,.50,.63]; these are not original-world coordinates |
| Independent initialization |Fresh field, Adam, learned occupancy/EMA and3D frequency; own-resume only; no cross-frame or cross-arm transfer |
| Training rays / seed |8192; seed42; fixed ray batch (`target_num_samples_per_batch=0`) |
| Sampling |`body80`, W64 confidence raster, precise native SAM head proposal15%; actual mixture normalization; no1/q correction |
| Full-ramp8192 quota |2116 core-uniform,2116 person-FAS,1092 protective-band,1639 whole-valid,1229 heads; head ramp4096→8192 |
| Input frequency map |Constant scalar `uniform16`; no learned2D fitting required;3D grid still starts empty and updates |
| Field |16 levels,2 features/level, resolution16→8192, log2 hash size25, hidden width64, SH4/corrected directions, trunc_exp density, sigmoid RGB |
| Reconstruction |Source-bound Charbonnier RGB objective with confidence normalization; no LPIPS training term |
| Optimizer |Adam LR.01, betas(.9,.99), epsilon1e-15, weight decay0; mixed precision |
| LR schedule |Exponential .01→.0001 over200000 steps, no warmup; do not shorten horizon to140k merely because training stops there |
| Original fixed-sampler warmup |4096 steps, then corrected adaptive ray marching |
| ARM training |Coarse.0015625; min step1e-4; max step.1; midpoint interval estimate; min frequency0/max unset; max1024 samples over the whole ray |
| Feature reweighting |Strength1 then.3 at75941, including the inherited seed-reset event; preserve the complete event |
|3D frequency updates |Every1024 steps, separate2048-ray no-gradient TRAIN depth probe; max-stamping/legacy projection |
| Occupancy |128³,1 level, EMA.95, threshold.01, update16, warmup4096; semantic static admission mapped hash b084919b… |
| Occupancy dilation |R0 in training and evaluation; no adopted R1, pruning or floor clipping |
| Opaque-material prior |Coefficient.01, fixed TRAIN-only32-native-pixel-eroded core assets |
| Distortion schedule |.01→.1 over4096→12288; preserve source schedule rather than the serialized initial scalar alone |
| Steps |140000 actual attempted optimizer updates maximum; replay consumes budget; terminal model reached the limit |
| Selected inference |Same140k weights, cap10; other ARM/FR/grid settings unchanged; no SSAA, image cleanup or masking in the scored raw renders |

Selected inference: PSNR27.100272, SSIM.579104, actor AlexLPIPS.095144829. Default inference on the same weights:27.228975/.590570/.098840536. Training loop297.730558 Silver minutes excludes preparation/setup/queue; this is not an end-to-end per-frame clock. Cap9 has lower aggregate LPIPS.094086626 but worsens HC face/boot detail, so it is retained as a numeric candidate rather than the selected detail renderer.



W64 — мягкая confidence raster[0,1] с64-native-pixel границей, **не64-кратный вес**. Защитная band и whole-valid floor нужны из-за волос/ошибок масок. Не заменять их жёстким binary cutout и не превращать confidence в physical alpha GT. Оpaque-core32px erosion — отдельно от head proposal и sampling core. Сохранять actual rounded mixture normalization; не добавлять1/q correction к этому objective.

`uniform16` — формат2D разрешения/скаляра16. Сохранять expected stride/patch/map raster metadata, перекамерные intrinsics и типы; генерировать по текущим crop dimensions.3D update uses2048TRAIN rays/no_grad каждые1024 итерации. Feature reweighting и число traversal samples — разные применения grid. Max1024 — общий corrected allocator на весь луч, а не1024 на каждый интервал. Не подменять legacy projection на normalized_jacobian: этот контроль ухудшил детализацию и не выбран.

## 6. Сначала один кадр: этап обязательной приёмки

Выбрать один репрезентативный кадр нового диапазона, с видимыми лицами/волосами/ступнями. Не выбирать специально самый лёгкий кадр только ради красивой оценки.

- Для честной проверки оставить DC/HC/KA вне обучения, остальные63 — TRAIN, если это те же camera IDs. При другой конфигурации заранее выбрать3 содержательных heldout ракурса. TRAIN-only admission, proposal normalization, head/core и maps не используют validation-контент. AABB/calibration preparation scope отметить отдельно.
- Создать **новый** output и frame-local spec/ready/head/core/semantic manifests. Исторические абсолютные пути в spec не являются шаблоном нового диапазона. Все path-prefix и frame-ID адаптации должны получить честные новые receipts/SHA; не редактировать data hash просто до совпадения старого.
- Сверить actual full config с эталоном: differences только data/calibration/AABB-dependent assets, host paths/output/stage plan, явно утверждённый split. Optimizer, schedules, neural shape, loss normalization, ARM сохраняются. `build_config(spec,resume=None)` должен иметь все load fields None, пустой Adam и нулевые learned occupancy/EMA/3Dfreq. **Не запускать terminal_config как cold config:** там есть old own-resume pointer.
- Точному worker нужны base imports строго из объявленного sealed checkout, overlay hashes и реальный native I/O. При необходимости портировать frame/path guards в отдельный проверяемый adapter без изменения математических вычислений; старые1660-specific guards не отключать молча. Если импорт/validation падает, это ошибка подготовки, не плохая модель.
- CPU-preflight с CUDA hidden, например после сборки корректного NEW_SPEC:

```bash
CUDA_VISIBLE_DEVICES='' python LookCloser/scripts/marshawn_native63_frozen/train_native_crop.py --spec /absolute/new/NEW_SPEC.json --dry-run > /absolute/new/dry_run.log 2>&1
```

- Это **не** turnkey запуск: production `--spec` запускать через проверенный supervisor/lease на новом хосте; source helper проверяет реальные родительские процессы/UUID/пути. Не подставлять вымышленные receipts. Достаточно новой эквивалентной безопасной регистрации job с реальными paths, без изменения модели. Готовые очереди/скрипты в репозитории имеют исторические scene paths и должны быть адаптированы/проверены.
- До длительного запуска проверить настоящие warmup и adaptive callbacks на отдельно помеченных коротких technical outputs, finite gradients, native RGB gathering и masks. Technical состояния в научную модель не переносить. Отличать основную adaptive forward с gradients от дополнительного no_grad frequency probe: на8192 есть оба вызова. Эти технические модели не считать обученными8194 шагами.
- После fresh start сохранить cold proof/manifest. Для пилота поставить явные manual gates32k→80k→100k→120k→140k, не сокращая LR horizon200k и не сдвигая FR75941. На каждом gate сохранить scored checkpoint/PSNR/SSIM/LPIPS и native renders. Gates служат проверкой/остановкой явных поломок, не отменяют140k выбранного полного рецепта.
- Продолжение **того же** кадра через собственный checkpoint допустимо с сохранением field/Adam/grid/RNG/counters. Нужны immutable stage receipts: старые stage/spec/source bytes не переписывать под живым процессом. Лимит140000 включает replay после восстановления; индекс139999 соответствует140000 обновлений без replay.
- Primary metric — equal-six masked AlexLPIPS двух актёров на3validation views; PSNR/SSIM рядом. Зафиксировать ROI лиц/волос/одежды/ступней до scoring и не менять маски ради метрики. Проверить masked metric footprint, поддержку LPIPS всех уровней и исключённый фон.
- Обязательно вручную сравнить native GT/default/cap10/alpha на лицах обоих, дредах, одежде,4ступнях, а также все fullbody views и несколько novel/interpolated views. Самый низкий scalar LPIPS не оправдывает локально размытую голову. Чёрные внешние cutouts хранить отдельно от raw renders, не выдавать их за исправленную geometry.
- На140k выполнить одинаковую оценку default и inference cap10; подтвердить отсутствие изменений checkpoint/model/grid/RNG при render-only overrides. Базовые метрики1660 не являются pass threshold для другой позы: критерий — корректность протокола, отсутствие известных clipping/цветовых/детальных регрессий и честно измеренное качество.

Сохранить `PILOT_ACCEPTANCE.md`: source/calibration/crop hashes, AABB receipt, cold proof, exact config/schedules, actual steps/replay, checkpoint SHA, все3метрики/actor breakdown, native ROI/fullbody renders, исходы manual review и нерешённые ограничения. Без успешной приёмки не запускать весь диапазон. Если данные явно ведут себя иначе, остановиться на пилоте и документировать причину; не объявлять перенос рецепта успешным по одному8k preview.

## 7. Occupancy: что проверять и чего не обещать

Стартовать с R0 как у выбранного лидера. По каждому кадру заново построить TRAIN-only semantic static admission и learned grid; старый mapped hash b084… нельзя копировать как истину нового движения. Alias/capture/core, actual R0, update-age≤15 при interval16 и absence active outside static allowance проверять по **реальному сохранённому** grid.

Сделать contour-ray replay для всех доступных камер с separate TRAIN/validation отчётом, source-bound projections/atlases и ручным осмотром спорных cell/foot/hair witnesses. Проверить voxel corners, lookup convention (occupancyR противfrequencyR−1), camera pixel centres, AABB transform и ray depth. Ray попал в какую-то occupied cell — не доказательство правильной поверхности/глубины.

У старого140k лидера29749active/108699allowed; heldout32804 nominal contour rays без пропусков. Исторические510TRAIN misses оказались статическим prop-contamination масок, не потерянными ступнями. **Этот конкретный erratum нельзя переносить на новые misses:** каждый новый набор ошибок проверяется заново. У старогоgrid8358cell имели≥2clear votes и234robust low candidates; это не safe-prune labels. UNKNOWN, окклюзия и выход изFOV не пустота. EMA — исторический proxy, не physical density/alpha.

При реальном clipping рассмотреть отдельный проверяемый R1 control и native boot/hair renders; **не включать R1 по умолчанию**, не накапливать его повторно на каждом chunk. Нужны exact restoration и сравнение false-positive shell/floor. Ни R1, ни aggressive floor clipping/pruning не являются принятым универсальным исправлением. Идеальный физический grid в этой сессии не получен: halo/floor sheets остались, несмотря на хорошие masked метрики.

## 8. Только после пилота: вся последовательность

1. Зафиксировать список source frames/timestamps, targetFPS и duration вместе с пользователем/доступными метаданными. По умолчанию прежняя задача хотела24fps, но новый диапазон ещё неизвестен. При120fps это шаг5 для24fps и шаг2 для60fps. Хранить полную source timeline, чтобы позже добавить пропущенные кадры без переименования уже обученных моделей. Проверять endpoint отдельно, не смешивать N output frames с inclusive boundary sample.
2. После63/3 приёмки финальный per-frame train может использовать все66 камер, как ранее разрешено пользователем. Это новый cold запуск: не продолжать63-view веса. Validation на включённых в train камерах становится in-sample, это явно маркировать. **Старой завершённой final66 модели нет**, только подготовка данных; её качество не известно заранее.
3. Каждому кадру отдельные output/spec/data/core/head/admission, uniform16 maps под его raster, seed42 и полностью пустые learned field/optimizer/grids. Никакого переноса learned occupancy/frequency/весов/Adam/прошлой density между кадрами. Общие calibration и фиксированный принятый box допустимы; frame-specific mask-derived geometry пересобирается.
4. Рецепт140k без самовольных291k, без изменения LR200k. Full lookcloser phases сохраняются. Не объявлять единую оптимальную раннюю остановку: она не доказана, старый120→140 default LPIPS улучшился ещё.735% за45.5мин, а cap10 old100/120 screen не был выполнен.
5. Использовать все доступные **свободные** GPU по одному worker на device; atomic UUID lease, realPID/start/boot identity, ограничение бюджета, OOM/telemetry checks. Не затрагивать чужие процессы. Не обходить несовместимые зависимости/архитектуры новым JIT во время большой очереди без квалификации.
6. Долговечная очередь должна исполняться без процесса Codex: persisted manifests, atomic receipts, own-checkpoint resume, idempotent outputs, write-permissions под фактическимUID, startup canary, retry policy с физическим debit и quarantine. Проверять минимальные nested output/eval/stop write операции заранее. На сигнал STOP закрывать и admissions, и workers; запрещать самоперезапуск после stop.
7. Писать stdout/stderr в файлы безTTY, compact progress с currentPID/step/checkpoint. Проверять минимум каждый час controllers/workers, свежесть реального progress, VRAM/OOM/checkpoints/pending eval и журналировать. Старый progress другогоPID не считать текущим обучением. Суммировать elapsed clocks только где они reset; persistent point counters читать один раз, не складывать resumed endpoints.
8. Для каждого кадра сохранять scored checkpoint и raw renders/metrics; latest unscored checkpoint не объявлять best. Technical/requested SIGTERM≠научный negative. Завершение worker≠успешное завершение оценки: требуется полный commit изображения/метрик/manifest.
9. До полного video export проверить несколько соседних обученных кадров и camera-motion на hair/feet/colour/floor. Даже хороший fixed-frame cap10 не гарантирует отсутствие flicker независимых моделей. Маски/композитинг и hybrid pixel transfer — отдельные post-training процедуры, не часть доказанного pure LookCloser training recipe.

## 9. Непереносимые предположения и запрещённые сокращения

- Не копировать AABB1660 или masks на новое движение без проверки всего интервала.
- Не считать scene bounds в box frame координатами original world.
- Не подменять cap10 inference на cap10 training; не брать cap9 только из-за среднего LPIPS.
- Не пересчитывать PQ на уже преобразованном sRGB/linear master; не обучать на уменьшенных картинках вместо native crop по привычке.
- Не сокращать neural capacity25 до23/24, SAM15 до bbox-top proxy или core/confidence до жёстких масок без нового отдельного контроля.
- Не выдавать all66 in-sample сравнение native/HD (19.5463% на старом контроле) за heldout эффект сильного текущего рецепта. Новый strict63 HD control был остановлен до32k.
- Не заявлять идеальную geometry/occupancy, насыщение validation или video stability: таких свидетельств нет.
- Не принимать любое число из исторического отчёта за актуальное состояние live queue. Для старой кампании окончательное состояние STOP; этот handoff — инструкция новой отдельной работы на другой машине.

## Минимальный deliverable следующей сессии

`DATA_AUDIT` + `AABB_ACCEPTANCE` с source-bound визуализацией + exact adapted recipe/spec/hashes + `PILOT_ACCEPTANCE` одного полностью проверенного кадра + только затем автономная video queue с per-frame cold proof,140k cap, актуальными метриками/рендерами и checkpoints. Сохранить evidence/inference/unknown отдельно, как в исходных итоговых отчётах.

## WS07/WS08 continuation — 2026-11-09, frames908–1268

Current work is a **new AABB review**, not the production LookCloser pilot.
User's default here is Full HD; four fight performers including fallen actors,
crew excluded semantically.120fps is inherited/assumed, not independently verified.
Read `experiments/marshawn_ws07_initial_aabb.md` and
`experiments/marshawn_908_1268_aabb/aabb_original_rig.json` before continuing.
V1/V2 boxes were withdrawn; V3 lower-Y correction uses NC to recover fourth-actor
boundary evidence missed by the other camera rasters. Practical V3 dimensions
3.5354×5.8677×1.7770m, floor-aligned rigid frame; not original-world axis bounds.

Seven short diagnostic bounded-NGP teachers and28 checkpoint partition renders
finished. DC/HC/KA held out; NC is an explicitly labelled training-view diagnostic.
Teacher backgrounds are masked, and learned depth has artifacts: these are not
accepted production LookCloser models or a dense geometry certificate. All361
frames were audited against72 coarse masks and original per-frame COLMAP seeds;
raw low-support outside witnesses and limitations are retained in the report.
No training remains active; original SAM workers were not killed.

Review folder: https://drive.google.com/drive/folders/1n70r-bKJ2fEnRUIa4ytPZbVXr6lT0jZG
Archive: https://drive.google.com/file/d/1vMLc-ecQ7JosAN9h1R3oTflJ5xTmIQJ4/view
Artifacts local WS07: `LookCloser/lookcloser_debug_outputs/marshawn_ws07_aabb/`.
Teachers on WS08: `/home/nvuser/lookcloser/logs/aabb_geometry_20261109/stage_campaign/`.
The user accepted the V3 box after visual review on 2026-11-09. The canonical
accepted geometry is `experiments/marshawn_908_1268_aabb/aabb_original_rig.json`:
apply its `world_to_box` before testing its `aabb` bounds. This closes the AABB
visual-review gate for frames908–1268; it does not certify a different interval.
Proceed through the exact adapted recipe/pilot before any
production video queue. Do not silently reuse the early silhouette box or declare
zero sparse outliers by filtering away occluded actor regions.

Live mask status checked 2026-11-09 08:22 PST: WS07 has14688/14688 nonempty
mask files matching source PNG names, WS08 has14616/14616. Missing and unexpected
counts are zero on both machines. Driver logs report completion at07:36:45 PST
(WS07) and07:12:03 PST (WS08). No SAM mask workers remain. Each GPU has only
an idle CUDA MPS server using68MiB in the compute-process listing. This was a
filename/nonempty-file completeness check, not a full mask-decode or quality audit.
