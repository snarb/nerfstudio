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

## Active native sequence campaign — 2026-11-09

User authorized six hours of first-frame tuning, then autonomous cold training of
all361 frames908–1268 on all72 cameras. V3 physical box remains accepted.
Three tuning heldouts are CB/GC/NB; final training includes them and labels its
checks in-sample. Primary selection is equal-person, equal-view masked AlexLPIPS.
Use native actor crops without resizing. Recovered historical model units are
0.2217203609534452 per rig unit (66-camera registration RMS4.69e-8); scale centers
and box uniformly, never rotation axes or intrinsics.

Tuning clock: 11:08:12–17:08:12 PST, persisted in `campaign_state.json` on Silver.
Coordinator: `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109`, systemd
`marshawn908-controller.service`. Eight Silver GPU agents plus WS07, WS08, Olive,
and Clever are active. Olive exposes one GPU in this login, despite an advertised
eight. Dev3 L40S has insufficient RAM for the current native training cache;
inference-only reload qualification is being prepared separately.

WS07 root `/home/pavlo/marshawn_campaign_20261109`; WS08 root
`/home/nvuser/lookcloser/campaigns/marshawn908_20261109`. Persistent seed copies
are under each root's `seed_points/`. Source ownership constrains final NVIDIA
allocation. All other machines fetch native prepared frames from the source
owners via calm-otter and keep independent per-frame outputs. Agents survive
SSH/controller loss and log GPU/process/progress checks every5min. Keep manual
supervision active at least hourly. Do not reset the tuning clock on restart.

The reference runtime's1531 source files match the historical seal byte-for-byte.
Worker/runtime/common and frozen source hashes bind scientific trajectories;
operational controller fixes do not invalidate weights. Failed metadata/env
preparation attempts are not scientific measurements. Warm technical weights
are never used for tuning/final starts. Follow
`experiments/marshawn_908_1268_tuned_recipe.md` for intermediate/final results;
it will be replaced by the controller's measured selection after the deadline.
Do not claim the final recipe or sequence is finished while tuning is active.

Early 8k review: https://drive.google.com/file/d/12FJ60o7WpBqlfhAVEV_Xwh3p7GDL_r0b/view

Operational update12:22 PST: all12 agents were restarted while their scientific
workers continued (identity checks passed). Production retries now allow12
attempts with backoff capped15min. Weighted final allocation balances both
source-constrained NVIDIA stations (~41frames each) rather than overloading the
later frame range. Dev3 exact-tensor inference parity passed (max LPIPS delta
1.03e-6); same-weight cap diagnostics and original-FOV Full HD raw renders are
separate helper jobs, with no training-image cache. Confirmed Olive/Clever can
fetch both NVIDIA producers through calm-otter; Clever can reach its Silver
archive target. Current best32k arm is half coarse step, actor LPIPS0.166759;
quarter step and W32-confidence ablations were added with unchanged17:08deadline.

Confidence-port correction12:53 PST: initial preparation incorrectly used a
linear outside falloff. Historical `marshawn_actor_confidence.py` uses exponent
log(.1)/log(.5)=3.321928…, with0.1weight at half width. Do not call earlier linear
arms exact historical-recipe baselines. New `reference_*` arms use immutable
variants computed by the exact sealed helper (SHA28b1c8aa…); they recompute the
falloff from qualified actor masks, also removing residual confidence halos near
rejected crew. RGB/actor/head labels and admission grid remain unchanged.
Qualified stored arrays match the actual helper on CB/GC/NB/CC. Historical
half/default/learned-half controls are now assigned; original deadline17:08:12
is unchanged. Keep every source run/profile labelled in the final comparison.

Operational update14:06 PST: use the durable private reverse proxy on WS07
localhost:22223 (`marshawn-ws07-proxy.service` on calm-otter). The old manual
22222 tunnel is no longer required for supervision; Silver's source-access path
already uses calm's Tailscale address directly.

Clever finished the32k resolution16384 control, then its qualified Python path
was removed externally. Another CA4D job owns its GPU; do not kill it. Clever is
`tuning_disabled`/`production_disabled` in fleet.json, preserving its measured
checkpoints and scores. CPU dependencies were staged in its campaign
`runtime_deps/` using the still-present SAM Python; GPU qualification is deferred
until the foreign job finishes. Main11GPU training continues. Per-host
preparation failures back off instead of crashing the entire scheduler.

Inferior linear controls arm_2048 and fr_initial03 were deliberately stopped
after their committed80k comparisons; `planned_pruning` records identities and
reasons. Their agent failed-status entries are planned preemption, not NaN/OOM.
Fresh `power_quarter`/`power_quarter_background` runs occupy Silver GPUs6/7.
All earlier measurements remain available. Whole-frame leases and late receipt
reconciliation harden executor crash recovery; regression checks passed.

The selected validation step and final training budget are separate: if the
winning trajectory has no confirmed late two-signal plateau, final frames may
use the150k-attempt ceiling and retain their best scored checkpoint. Never
relabel a shorter validation measurement as a150k observation. The coordinator
reuses the actual worker plateau function on committed metrics.

HPO operational update14:15PST: added `reference_background_strong` (historical power W64, half coarse step, conservative background prior0.1) onSilver2, replacing the measured inferior default-step power control. The live late80k GC raw actor overview still has opacity haze. Late raw full-FOV/novel Dev3 checks are being staged by `/tmp/late_visual_marshawn.py` onSilver; log `late_visual.log`. Canonical geometry and six-hour deadline remain unchanged.

14:24PST promotion: `reference_background`32k (actor LPIPS0.164264, auxiliary opacity0.098397) continues from its own checkpoint onSilverGPU3. The identical-runtime GPU migration preserves the shared immutable run/spec and imports only the verified completed32k screen receipt into that agent. Its prior GPU5 now continues the separate raw-rig-units control. The inferior learned090 linear trajectory was deliberately pruned after committed96k (LPIPS0.155552 versus half-step0.145254). Promotion is recorded in `campaign_state.json.planned_pruning`; no clock reset or cross-arm warm-start. Late80k full-FOV/novel review is on Drive: https://drive.google.com/file/d/1rOUx2L6r1gnMdQQiZMIUj9xLzql0Lbgx/view .

Selection policy clarified before quarter-power32k/strong-prior16k results: latest shared committed gate among continued finalists, with both inference caps scored at equal steps; only then pick the winning arm/cap's own best measured checkpoint. This avoids choosing an older configuration solely because it started earlier. `comparison_finalists` is captured at the fixed deadline; `matched_comparison` records the gate, membership and selected gate result. Unequal128k/64k regression test passed. The six-hour clock, LPIPS priority and0.5%/5% tie guards are unchanged.

15:18PST diagnostics: source mask production CSVs cover14,688WS07 +11,304WS08 =25,992target images, allok, no zero-area masks; audits saved in repo `experiments/marshawn_908_1268_diagnostics/`. Dev3 same-weight alpha-threshold sweep0/.001/.003 is ineffective (LPIPS delta<6e-8), because `FrequencyAwareVolumetricSampler.get_sigma_fn` returnsNone in eval. No inference policy change. Alpha diagnostic JSON is committed; executable pilot remains `/tmp/alpha_diagnostic_marshawn.py` onDev3. Stronger background prior32k0.167160 is worse than0.01prior0.164264; power-quarter32k0.157821 and power-quarter+0.01prior0.158535 continue. All11 workers verified alive around14:59, current job logs have no OOM evidence; next full manualcheck≤15:59 (prefer15:30). Helper with OOM scan: `/tmp/manual_health_with_oom.py` onSilver. Main freeze policy now uses equal-step finalist comparison; new deadline-stop audit classifies budget termination separately from numerical failure.

Production-runtime sealing update: WS07 settings now have `production_tools=/home/pavlo/marshawn_campaign_20261109/production_tools`, `production_frozen_helpers=.../production_frozen_helpers`; its system agent was moved to that snapshot with KillMode=process, preserving its live HPO child. HPO69 retains original repo paths and its existing spec reconstructs identically. Future production fixes must also be deployed to this snapshot (ops files only; do not edit sealed neural files). Every unique host has campaign-owned `ops/nerfacc_cuda.so` plus recorded SHA and `production_torch_extensions_dir`. Olive's copy came from the actual binary mapped by its running HPO worker, previously under an older FSX project. CPU bootstrap probes on all5unique roots pass and confirm `_C.__file__` is the sealed module, with JIT bypassed; Clever is still GPU-unqualified/disabled. Proof summary onSilver: `production_runtime_probes.json`; deployment helper `/tmp/probe_production_runtime_marshawn.py`. Silver SAM now lives at its own `sam3_overlay` and has exactly the same checkpoint SHA as NVIDIA (`9999e234...`). A real SAM forward watcher `/tmp/check_sam_owned_runtime_marshawn.py` waits for SilverGPU4's150k job to complete; output `sam_owned_runtime_forward.json/log`. This needs inspection before production.

Dense-inference pilot onDev3 is negative: quarter32k0.157821 -> half-step/2048samples0.159106 -> minimum-step/4096samples0.159740, with worse head metrics and slower evaluation. Half80k0.148268 -> denser0.150875. No inference policy changes. Pilot script `/tmp/dense_render_diagnostic_marshawn.py`, results committed under `experiments/marshawn_908_1268_diagnostics/`. Matched32k raw GC contact sheet uploaded and metadata-verified: https://drive.google.com/file/d/1WLejJmsmCPMCWFVtMNndvFBMJb9RYQqq/view .

Real Silver SAM forward from campaign-owned source/weights passed on idleGPU4 after the half-step arm completed150k: CB native7597×3367,8raw person proposals,~6.96GB peak allocation; no training data changed. Proof `sam_owned_runtime_forward.json`, also committed under diagnostics. Weights match NVIDIA exactly. Production crash recovery now rejoins any surviving neural grandchild via `train/worker.lock` before preprocessing, then skips retraining if the stage receipt already satisfies the frozen target/attempt ceiling. A real subprocess/lease regression passed. This closes the parent-crash retry-exhaustion gap; all5unique roots' production ops were synchronized by `/tmp/sync_production_ops_marshawn.py` (log `ops_orphan_recovery.log`). Disabled Clever is excluded from deadline signals, mandatory recipe delivery and production queues; its unrelated CUDA job remains busy. Finalist64k/80k raw/novel visual exports were queued onDev3 by `/tmp/finalists64_visual_marshawn.py`; log `finalists64_visual.log`. Labels: `reference_background_80000`, `power_quarter_64000`, `power_quarter_background_64000`. Last full manual process/GPU/OOM check16:05PST passed; next check≤17:05, preferably16:40, then supervise the17:08deadline/production transition.

FROZEN RECIPE / PRODUCTION: six-hour window11:08:12–17:08:12PST completed; controller phase `production_dispatched` by17:14. Winner `power_quarter_background`, selected cap10 checkpoint80k: actor LPIPS0.139517659, PSNR29.54254, SSIM0.690776. Continued finalists compared at common80k (11arms). Winning trajectory completed88,476iterations; plateau not confirmed. Production budget150,000attempts, best scored checkpoint retained. Params uniform16, coarse.000390625, historical_power W64, background prior.01, model scale.2217203609534452, RAM gathering. All72cameras TRAIN in final models. Frozen JSON is on each enabled root, copied to repo `experiments/marshawn_908_1268_selected_recipe.json`; MD mirrored to repo. Allocation44WS07 +44WS08 +8×31Silver +25Olive =361; Clever0. First Silver jobs909,910,911,912,914,915,916,918; Olive919; NVIDIA908/1112. At17:30 local72camera preparation complete on both owners; WS07 parts/qualification and WS08 admission/confidence building were progressing, remote source producers for9frames had72QC receipts. No production completion claimed yet.

Raw exact-winning80k GC FHD comparison uploaded and metadata-verified: https://drive.google.com/file/d/10Ormx7gGEV4bd397QNt7RpKuQzqo31DX/view . All3winner fullHD+nearby novel views are onDev3 `fullhd_power_quarter_background_80000`, `novel_power_quarter_background_80000`. Raw contour halos remain; source-mask tile seams and NB head false-positive were already documented. Additional late linear-quarter144k diagnostic was rendered directly onWS07 at root `mature_local_nv144/{fullhd,novel}` using its verified source export. A binary export transfer WS07→Silver was only~0.7MB/s and was cancelled by exact owned helper PID/children (recordSilver `mature_nv_transfer_cancelled.json`); partialSilver `helper_exports/nv_quarter_144000` is NOT verified and must not be loaded. Original verified export remains onWS07 `helper_exports/nv_quarter_144000/model`. Source PNG-derived frame transfers may be a throughput bottleneck; measure actual production traffic before changing it. Future safe performance work: pipeline one-frame CPU/native prefetch and/or parallel file transfer, preserving per-frame leases, source checksums and bounded staging. Do not edit frozen neural files. Final recipe JSON is immutable while frames run.

17:46PST production update: Clever's foreign GPU job exited independently. Recovered runtime passed real256-ray CUDA neural forward/backward and SAM native-image forward, with matching SAM weight SHA. Enrolled24never-started tail frames only; all running parents untouched. New allocation WS07/08 41each, Silver8×29, Olive23, Clever24;361unique. Silver `clever_enrollment_plan.json` retains before/after queues and qualification, `clever_enrollment_complete.json` commits allocation. Controller restarted to reload enabled Clever, phase remains production_dispatched. NVIDIA frame908/1112 neural SGD active; all8Silver native transfers complete, their SAM stages active. First908 production spec independently verifies72TRAIN,72head/admission domains, exact frozen parameters and all sealed source/CUDA SHAs; audit in repo diagnostics. Last full manual health/OOM check17:39PST; next due≤18:39. Exactwinner80k full/novel3camera contact reviewed/uploaded: https://drive.google.com/file/d/1Q3o86ptWya4MoLjYWmENt-XT-1GIcJPS/view .

17:49PST CPU prefetch deployed: `preparation_prefetch.py` selects one never-started next frame per active remote queue; watcher runs it every300s, source `producer.py --nonblocking` reuses existing per-frame lease. No model code/spec changes. Requests/logged faults inSilver `prefetch_status.json`; original verified transfer/release path unchanged. Deployedcampaign/producer/newmodule toall5roots includingWS07production snapshot; controller restarted, active parents preserved. Local NVIDIA path not prefetched. Real held-lease and queue selection tests pass. Next manual fleet healthcheck≤18:46PST, sooner for first Silver neural start.

17:51PST: CPU prefetch first pass succeeded forSilver920/921/923/924/925/926/928/929,Clever1106,Olive930. `prefetch_status.json` records10requests, no access faults; sourceproducers continue under existing leases. Inference loader now resolves exact archived sibling dataset relative to relocated train/spec before generic host variants; explicit override still wins. Temporary filesystem checks passed (archive/nativehost/fallback/override); deployedinference.py alone to5owned production tool roots. Sealed worker/runtime/common/wrapper unchanged. Controller/12productionparents checked17:46, no OOM; nextfullcheck≤18:46.

17:54PST verified supervision: controller systemd active; manual_health_with_oom appended live-identity checks for all12production parents, fresh agent states and no current-log OOM evidence. All8Silver neural worker_background processes now independently observed live (PIDs2910931,2910933,2910937,2911080,2911134,2911491,2912142,2912475), not merely preparation parents. WS07worker1385936 is evaluating8k; committed default-cap evaluation exists, cap10 pending (progress last7763 because evaluation publishes progress after both caps). No restart warranted. Sequence0completed/0quarantined/361. Next useful check18:05–18:15PST for committed8k/first Silver SGD, mandatory full check≤18:53PST. Previous turn made concrete progress (Clever requalification/enrollment, CPU prefetch, archive-loader fix); current turn verified live waits.

17:55PST first production8k evaluation committed both caps: frame908 cap10 in-sample actorLPIPS0.188911, PSNR30.893211, SSIM0.736407. GC review visually inspected; all4performers present, expected early halos persist. Review is actor-ROI canvas with unqueried exterior zero-filled, not a full-FOV render; rectangular ROI boundaries are not AABB clipping. Silver8actual neural PIDs confirmed live with3090MiB each while initialization/data caching proceeds; do not restart merely because progress.json is not published yet. Next useful poll≈18:00–18:05, full manual supervisor deadline≤18:53.

17:57PST verified wait: all8Silver neural workers advanced to381–1157steps, live PIDs unchanged, GPU utilization76–86%,~21GiB each. WS07worker1385936 advanced9948steps after committed8k evaluation. Source-side CPU overlap finished all10future frames920/921/923/924/925/926/928/929/1106/930; all producer_complete receipts inspected, no producer processes remain. Earlier scratch output labelledprefetch_active0 used a placeholder expression and is not evidence; actual /proc cmdline scan and10receipts provide the verified result. Current jobs continue; no restart/recipe change. Next useful check≈18:05–18:15, mandatory fullcheck≤18:53PST.

17:58PST full fleet live-identity/GPU/current-log OOM check appendedSilver manual_supervision.jsonl:12parents alive, no OOM evidence. Olive919 historical confidence reachedNC/ND/NE, no neural progress file yet; this is active preparation, not failure. Clever1102 actual SAMchild2183934 alive underproduction2175663, parts reachedMB; no restart needed. Previous turn was a verified wait with advancing8Silver SGD and10CPU receipts. Next useful poll18:05–18:15, mandatoryfullcheck≤18:57PST.

18:09PST verified wait and visual gate: frame90816k bothcaps committed; cap10 in-sample actorLPIPS0.165569 (8k0.188911), PSNR31.274744, SSIM0.751370. GC16k actor-ROI review inspected:4performers retained, halos persist; not full-FOV. WS07worker1385936 subsequently advanced16795steps. Silver8same neural PIDs live,7612–7943steps approaching8k gates. Cleverreal neural2195867 now1979steps/~9.86GBtorch allocation, no prep ambiguity. Manual checks inWS07 manual_local_checks.jsonl; lastfullfleetGPU/OOM17:57, nextmandatory≤18:57PST, prefer18:25 after Silver8k. Previous turns are verified waits for live advancing workers; no recipe changes or restarts.

18:16PST secondary-range visual check: WS08 frame1112 neuralPID1430376 confirmed live at27792steps. In-sample cap10 LPIPS8k0.244063 ->16k0.213102, PSNR16k31.496842, SSIM0.749528. GC16k review copiedWS07root/late_review/frame1112_GC_16000_review.jpg and visually inspected:4performers present, hair/hat retained, contour/floor halos persist; ROIoverview zero-filled exterior. Lastfullfleetcontroller/parent/GPU/OOM18:12PST clean, remoteSilver8+Olive+Clever neural PIDs separately live; nextmandatory≤19:12PST, prefer18:30. WS07latestverified22k checkpoint, Silver10k+.

18:30PST production update: all8Silver16k cap10 evaluations committed, in-sampleLPIPS0.170374–0.178264 vs8k0.194957–0.202487; all8same neural PIDs verifiedlive and advanced16914–17525. Silverroot/silver16_progress.json storesperframe rows. Frame909GC16k ROIreview copiedWS07late_review/frame909_GC_16000_review.jpg and visually inspected:4performers present, known halos remain. WS07realworker1385936 nowhascommitted32kcheckpoint and is evaluating; stale31,899 progress during eval is expected, no restart. WS08frame1112 passed32k cap10LPIPS0.184907 and continued>33k. Lastfullfleetmanual18:20PST all12parents/current-logOOMclean; nextmandatory≤19:20, prefer18:45.

18:34PST fullfleet supervision:18:33manual_health_with_oom appendedSilverlog,12parents live/fresh and no current-logOOM. WS07worker1385936 resumed after32k cap10LPIPS0.144240,PSNR31.758622,SSIM0.769149; now33858steps,32kcheckpoint preserved. Silver8at19536–20059steps, continuingafter16kgates. Firstframe908 in-sampleLPIPS8k0.188911→16k0.165569→32k0.144240; no plateau established. Lastfullcheck18:33, nextmandatory≤19:33, prefer19:00; goal still361models,0complete lastcontrollerreceipt. Recipe/weights unchanged.

18:58PST update: WS07frame90848k cap10 evaluation committed,in-sampleLPIPS0.132048,PSNR32.029465,SSIM0.780137; resumed48749steps. WS08frame1112 48kLPIPS0.171194 (32k0.184907), resumed48373 at18:48. Silver8near32kgates, exactPIDs unchanged; all12parent identities fresh/no current-logOOM in18:56fullmanualcheck storedSilvermanual_supervision.jsonl. Nextmandatoryfullcheck≤19:56PST, prefer19:20. No completion yet. Recipe remains150kmax attempts; current early scores do not establish plateau. All10prefetchedfuture source frames alreadyready. WS07manual_local_checks.jsonl has frequent actual/proc-verifiedprogress entries. Nextvisualgate: frame90848k andSilver32k if needed; priorframe90816k/frame90916k/frame111216k inspected andhalosdocumented.

19:18PST ongoing production: lastfullmanualfleetcheck19:14PST12parents/GPU/current-logOOMclean; controlleractive andSilver8/Olive/CleveractualneuralPIDs separatelyverifiedlive afterwards. Nextmandatoryfullcheck≤20:14PST, prefer19:40. WS07latest62kcheckpoint, live>62k; cap10in-sample48kLPIPS0.132048,PSNR32.029465,SSIM0.780137; GC48kROIreview inspected4people/knownhalos. Silver8all32kcommitted,LPIPS0.148418–0.154729 and now42090–42866steps. SnapshotSilverroot/silver16_progress.json currentlycontainslatestallgates despiteoldfilename; helperlocal/tmp/silver16_progress.py queriesactualPIDs andscores. WS08last48kLPIPS0.171194,resumed. Goal361independentmodels stillnotcomplete. Frequentactual/proc-checkedprogress inWS07manual_local_checks.jsonl; recipe/worker/runtime/core unchanged.

19:31PST latest supervised production: fullfleet19:27manualcheck12parents alive/current-logOOMnone, controller+SilveractualneuralPIDs live subsequently. Nextmandatoryfullcheck≤20:27PST, prefer19:50. WS07frame90864kcap10 in-sampleLPIPS0.124558,PSNR32.228670,SSIM0.787535; resumed>69k. Silver8all48kcommitted,LPIPS0.136071–0.142418, actualPIDs unchanged and resumed48274–49131steps. Latestperframe scores overwritteninSilverroot/silver16_progress.json (namehistorical,contentsuptodate48k). WS08lastknown48kLPIPS0.171194. No plateau established; no recipe/budgetchanges;361models stillrequired. Frequent/procverifiedwaits inWS07manual_local_checks.jsonl. Existinglargeuntrackeddebugfoldernotcommitted.

20:00PST ongoing supervision: lastfullfleetmanual19:56PST12parents/GPU/current-logOOMclean; nextmandatory≤20:56PST, prefer20:20. WS07frame90880kcap10 in-sampleLPIPS0.119887,PSNR32.357277,SSIM0.792395, resumed>85k; controlgate75941LPIPS0.121656,PSNR32.322631,SSIM0.791306. Checkpoint76kexists afterFRswitch, trainingcontinues. Silver8sameactualneuralPIDs live62909–63799steps approaching64k, latest48kLPIPS0.136071–0.142418. No parameter/attempt-budgetchanges; no confirmedplateau. Frequent actual/procverifiedwaits inWS07manual_local_checks.jsonl; first361framecampaign stillongoing.

20:12PST latestverifiedwait: WS07frame90894kcheckpointpresent/live>94k, latest80kcap10LPIPS0.119887,PSNR32.357277,SSIM0.792395. Silver8all64kcommittedLPIPS0.129632–0.134002; same8neuralPIDs live68272–69383steps. Olive3008268 andClever2195867 actualneuralprocesses separatelylive. Lastfullmanualfleet20:04PST12parents/GPU/current-logOOMclean; nextmandatory≤21:04PST, prefer20:35. First361framecampaigncontinues; no recipe/budget/codechanges. Frequent/procverifiedwaitlogsWS07manual_local_checks.jsonl. NextusefulqualitygateWS0796k andSilverFR75941/80k; beforefinalcomplete mustverify361frame receipts/exactexports and finalvisualreviews.

20:25PST latest production: fullmanualfleet20:23PST12parents/GPU/current-logOOMclean, nextmandatory≤21:23PST (prefer20:50). WS07frame908100kcheckpointpresent/live>100k;96kcap10in-sampleLPIPS0.113263,PSNR32.428990,SSIM0.794943. Silver8sameactualneuralPIDs live74812–75754steps atpre-FR75941gate, latest64kLPIPS0.129632–0.134002. Goalstill361 independentexports, noneclaimedcomplete. Recipe/150kattemptceilingunchanged. Frequent actual/proccheckedwaits inWS07manual_local_checks.jsonl; stagednext10CPUframesalreadyready. NeedwatchSilverFR/80k andWS07112k next, thenfirst150kexports/queueadvancement.

20:46PST ongoingcampaign: fullmanualfleet20:44PST12parents/GPU/current-logOOMclean; nextmandatory≤21:44PST, prefer21:05. WS07frame908112kcap10 in-sampleLPIPS0.110172,PSNR32.467600,SSIM0.796876; resumed>112k. Silver8all80kcommittedLPIPS0.123914–0.129688; actualPIDs unchanged/live82598–84268steps. TheirFR75941switchcomplete andstepsgrowingafter80k. Latestallgates snapshotSilverroot/silver16_progress.json; scriptlocal/tmp/silver16_progress.py. No budget/recipechanges; keepgoalactiveuntil361verifiedexports, notmerelyfirstbatch. Frequentactual/proc-verifiedwaitsWS07manual_local_checks.jsonl.

21:30PST supervisedwait: WS07frame908140kcheckpointpresent/live>140k, approaching150kfinalgate/export; latest128kcap10in-sampleLPIPS0.108500,PSNR32.503637,SSIM0.798186. Silver8all96kcommittedLPIPS0.116795–0.125327 andsameactualneuralPIDs live103726–105330steps. Lastfullmanualfleet21:24PST12parents/GPU/current-logOOMclean; nextmandatory≤22:24, prefer21:50forfirstfinalization. SnapshotSilverroot/silver16_progress.json containslatest96kgates. No completionclaimed; mustverifyfinalexportexacttensorreceipt/modelSHA/72TRAIN andqueueadvancement, thenfinalvisuals. All361modelsrequired. Currentgoalturnsareverifiedwaits, progressrecordedWS07manual_local_checks.jsonl.

FIRST VERIFIED COMPLETION21:38PST: WS08frame1112 completed150000iterations; selectedcap10checkpoint144000, in-sampleLPIPS0.137924938,PSNR32.774396,SSIM0.798674. Exacttensor-roundtrip exportmanifestpassed and independent streamedSHA256 verified6587e9e294dd434efa1fec24b454f3268b4c63ac92e0818a8df55200df96a4b7. WS08root/first_export_manual_audit.json, compactrepo diagnostics/first_completed_model_audit.json. Queueadvancedframe1116 withoutmanualintervention. GCselected144kROIreview copiedWS07late_review/frame1112_GC_final144000_review.jpg and visually inspected21:38:4fighters retained, hair/hatpresent, knowncontour/floorhalosremain; rawfullFOVfinalreviewstillneeded. NeedfinalCB/NBreviews too. Do notclaim361complete; firstverifiedmodelonly. Currentgoalworkkeepactive; allotherqueuescontinuing.

Firstcompletedframe1112 finalFullHDreview finished: WS08frames/001112/final_review/{fullhd,novel}/render_receipt.json bothpresent;sourcecanvasreadfrom72originalPNGheaders,FullHD1920x1080letterbox,rawunmaskedfield+straightRGBA. Sixviewcontact visuallyinspected21:41PST:4fighters retained, nativeFOVcutoffwhereapplicable, knowncontour/floorunsupportedopacitypersists. UploadedDrivehttps://drive.google.com/file/d/1Cxf144SEHeQHFyF5PVK2ymMbeDN61xVX/view metadatareadbackverified. LocalWS07late_review/frame1112_final_contact.jpg. SourceexportSHAalreadyindependentlyverified; queueadvanced1116. Goalcontinuesforremaining360frames.
