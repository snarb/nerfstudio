# Marshawn 908–1268 — передача работы на silver-sable

Обновлено 2026-11-12 04:49 UTC. Этот файл подготовлен для запуска новой сессии Codex непосредственно на silver-sable.

## Текущее состояние — читать первым

На проверке 2026-11-12 04:33 UTC завершена 101/361 исходная модель. Все 72 sparse24 модели обучены, визуально проверены, а их выбранные веса физически подтверждены потоковым SHA на silver-sable. Готовое трёхсекундное Full HD GC видео: https://drive.google.com/file/d/11MyVEOnvsiXGFrcAR0hnQdUFlSLoD9h0/view . Исправленные 923/988 включены.

Новая партия прошла окончательный аудит и визуальный просмотр:1111,931,944,1240,935,1234,949,1215,1220. У последних восьми LPIPS людей 0.119–0.136; явного обрезания боксом нет, ореолы/фоновые фрагменты остаются. Рендеры и receipts сохранены и закоммичены.

Активные задания Silver:954,956,1227,947,937,961,939,940; Clever:1244; Olive:975. Рабочие службы живы, текущие OOM отсутствуют; задания могут находиться на подготовке до появления фактического NN PID. WS07/WS08 executor services остановлены внешним TERM, планировщик пометил их retired; исходные файлы ещё доступны для переноса. Не перезапускать их обучение по старым progress files.

`marshawn908-priority.service` штатно завершилась после sparse24; это ожидаемо. Полный диапазон из 361 кадра теперь обслуживает `marshawn908-dense-scheduler.service`. Все 361 seed cloud подтверждены на Silver. Полный native перенос и архив исходных SAM3 masks/calibration ещё выполняются; готовность проверять по complete receipts, не по одному active статусу служб. Ниже сохранены подробности и исторические снимки, более старые счётчики не являются текущими.

## Репозиторий

- GitHub — https://github.com/snarb/nerfstudio
- Новая ветка — `scene/marshawn908-silver-handoff-20261111`.
- Основа — `scene/marshawn1660-native63-20261109`, исходный handoff был в коммите `57d35d7e`.
- Отчёт по коду — `LookCloser/experiments/marshawn908_code_report_20261111.md`.
- Этот файл — `LookCloser/handof.md`. Старый `handoff.md` относится к предыдущей передаче.
- Не коммитить существующий локальный untracked каталог `LookCloser/lookcloser_debug_outputs/marshawn_ws07_aabb/` целиком.

## Цель и приоритет

Исходная кампания содержит 361 независимую модель, кадры 908–1268 включительно. Сначала закончить общее покрытие трёх секунд при 24 fps — 72 кадра `range(908, 1268, 5)`, последний 1263. Затем идут остальные кадры исходной кампании. Пользователь просил всем машинам помогать общей очереди 24 fps.

Последняя проверка планировщика показала 72 завершённые задачи из 72 приоритетных. Визуально проверены все 72 приоритетные модели. Это разные счётчики. Уточнять их по актуальным receipts и журналам. Последние завершения ещё требуют аудита и визуальной проверки.

Не объявлять всю кампанию завершённой, пока не проверен исходный объём. Для переноса с удаляемых workstation пользователь запросил обучающие данные именно 72 приоритетных кадров. Данные остальных кадров исходной кампании могут потребовать доступ к оригиналам на workstation/S3. Автономность после удаления workstation обеспечивается для покрытия 24 fps с уже сохранёнными данными. Плотная часть всей кампании требует отдельной проверки доступности входных данных.

## Где запускать на silver-sable

Подготовлен checkout ветки для новой сессии:

```bash
cd /cache/ubuntu/lookcloser/nerfstudio_handoff_20261111/LookCloser
# Запустить Codex и попросить прочитать handof.md.
```

Дополнительная независимая копия handoff находится в `/cache/ubuntu/lookcloser/handoff_20261111/handof.md`.

Хост `ubuntu@silver-sable`, Tailscale `100.67.28.20`. Основной каталог:

```
/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109
```

Python обучения:

```
/home/ubuntu/lookcloser/envs/marshawn_leader/bin/python
```

8 × RTX PRO 6000 Blackwell Server, по 96 GiB. Полный цикл 150k шагов на одной GPU занимает около 5 ч 17 мин, выборка 21 завершённого запуска текущего рецепта без совместного использования GPU. На GB300 Workstation07 среднее около 4 ч 04 мин по 4 кадрам. Время включает периодическую оценку и исключает подготовку/перенос.

Рабочие файлы на silver-sable:

- `tools/` — развёрнутые скрипты orchestration и обучения.
- `runtime_base/`, `frozen_helpers/`, `ops/torch_extensions/` — квалифицированный runtime.
- `fleet.json` — актуальные адреса, очереди и каталоги хостов.
- `agents/gpu0` … `agents/gpu7` — очереди и состояния восьми GPU.
- `frames/<frame>/` — native data, spec, train, model, receipts.
- `external_models/clever/<frame>/` — архив моделей clever-shadow.
- `final_reviews/<frame>/` — финальные Full HD изображения и receipts.
- `priority_scheduler_status.json` — счётчик задач 24 fps.
- `manual_supervision.jsonl`, `manual_neural_checks.jsonl` — проверки.

Начальная проверка новой сессии:

```bash
cd /cache/ubuntu/lookcloser/campaigns/marshawn908_20261109
systemctl is-active marshawn908-controller.service marshawn908-priority.service
systemctl list-units 'marshawn908-agent@*.service'
cat priority_scheduler_status.json
cat /cache/ubuntu/lookcloser/emergency_workstations_20261111/data24_complete.json
```

## Автономные службы

- `marshawn908-controller.service` обслуживает кампанию.
- `marshawn908-agent@0.service` … `@7.service` независимо запускают задания, сохраняют состояние и повторяют recoverable failures.
- `marshawn908-priority.service` перераспределяет незапущенные приоритетные задания, ведёт журнал транзакций и открывает общую dense gate после 72 завершений.
- На Olive и Clever продолжают работать их собственные durable agents. Адреса брать из `fleet.json`.

Добавлен `retiring_host.py` и обновлён `priority_scheduler.py`. Планировщик сохраняет наблюдения workstation в `evacuated_hosts/ws07` и `ws08`. После повторяющихся ошибок соединения в течение как минимум 120 секунд он использует сохранённое состояние и перераспределяет незавершённые задания. Завершёнными сохраняются только задачи, чьи веса присутствуют в проверенном backup inventory. При длительном сетевом разрыве это может вызвать повторное холодное обучение кадра на другом хосте. Кэшированный хост после retirement больше не получает задания. Текущие нейронные процессы при развёртывании этого изменения не перезапускались.

Входные данные всех 72 приоритетных кадров теперь локальны на silver-sable. `native/transfer_complete.json` и `source_cache_released.json` позволяют обычному входу production использовать их без обращения к workstation. Службы самостоятельно продолжат обучение после окончания этой сессии. Человеческий финальный визуальный просмотр остаётся задачей следующей сессии.

Рутинные проверки пользователь просил проводить не чаще раза в 15 минут. Локальные AGENTS требуют проверять живые процессы, прогресс и OOM хотя бы раз в час при активном наблюдении. Использовать quiet логи, без TTY с rich progress.

## Эвакуация workstation — наиболее срочная часть

Workstation07 и Workstation08 скоро будут удалены. На silver-sable backup находится здесь:

```
/cache/ubuntu/lookcloser/emergency_workstations_20261111
```

Под `ws07/filesystem/` и `ws08/filesystem/` сохранена исходная структура абсолютных путей.

Первый snapshot — 14 выбранных inference exports и два полных чекпоинта активного обучения. Все 16 файлов проверены SHA-256. По просьбе пользователя старые промежуточные чекпоинты исключены из дальнейшего переноса. У завершённых production кадров optimizer checkpoints ранее штатно удалялись после durable export. Их отсутствующие пути отражены в inventory_errors.

После этого выполнен delta scan. WS07 дал восемь готовых моделей, WS08 восемь моделей и один полный чекпоинт. Эти два набора прошли проверку 8/8 и 9/9. Дополнительный полный чекпоинт WS07 1108, step 79999, сохранён и проверен. Итоговый обновлённый snapshot содержит 9 файлов WS07 и 9 файлов WS08, все 18 прошли SHA-256. Среди них 16 выбранных моделей и два полных чекпоинта активных кадров. Предыдущие snapshot сохранены.

Службы и доказательства:

- `marshawn-emergency-backup.service` — restartable rsync с параллельными потоками, сборкой крупных чекпоинтов из частей и полным SHA-256.
- `marshawn-emergency-capture-latest.service` — дополнительная фиксация чекпоинта 1108.
- `marshawn-emergency-metadata.service` — конфигурации, manifests, код и состояния. Проверено 3385 файлов WS07 и 3417 WS08 на наличие в backup.
- `marshawn-emergency-data24.service` завершилась успешно.
- `data24_complete.json` содержит все 72 кадра и 5184 camera views. RGB и semantic NPZ сверены по SHA-256 с native manifests.
- `ws07/source_inventory.json`, `ws08/source_inventory.json` — текущие списки сохранения.
- `ws07/checkpoint_verification.json`, `ws08/checkpoint_verification.json` — результаты проверки текущего snapshot.
- `history/` под каждым хостом хранит предыдущие inventories/receipts.
- `backup.log`, `capture_latest.log`, `metadata.log`, `data24.log` — журналы.

Дополнительный чекпоинт 1108 проверен. Перед разрешением удаления workstation проверить только файлы, появившиеся после последнего snapshot, поскольку обучение продолжается.

S3 upload был отклонён `403 AccessDenied`: workstation credentials имеют только чтение. Ничего не считать загруженным в S3. Резервная копия находится на silver-sable. На `/cache` оставалось около 2.2 TB.

## Данные 72 кадров

Канонический путь на silver-sable:

```
/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/frames/<six-digit-frame>/native
```

Для каждого кадра сохранены RGB-кропы 72 камер без ресайза, semantic soft alpha, sampling masks, transforms, native manifests и seed points. Это данные для выбранного PNG-рецепта и принятого AABB. Полные исходные 16-bit PQ16 PNG за пределами кропов и удалённые linear masters этим переносом не сохранялись.

`native/masks/*.png` задаёт допустимый sampling canvas. Маска людей находится в `native/semantics/<CAM>.npz`, ключ `soft_alpha`. Не путать эти два вида масок.

## Рецепт и AABB

Рецепт зафиксирован после шести часов HPO. Подробности в `experiments/marshawn_908_1268_tuned_recipe.md` и JSON рецептов. Выбран PNG, RAM gather, native RGB crops, uniform16 frequency maps, 150000 optimizer attempts максимум. Все 72 камеры включаются в финальный train. Метрики этих финальных моделей in-sample.

AABB в исходном floor-aligned rig:

```
[[-0.41, -1.20, -0.15], [3.41, 5.14, 1.77]]
```

Применять `world_to_box` из box.json, затем model_units_scale 0.2217203609534452. В боксе четыре дерущихся исполнителя, crew исключены. Нижняя плоскость параллельна полу.

Исходный рецепт SHA256:

```
0fb8447277238890dce1c710bd5e390c4a7246cca0175442efb27a9d7fde5f11
```

Исправленный рецепт для новых холодных кадров `recipes/admission_votes3.json`:

```
bcaec09aa472f8baa43f998b0af54de555cd90b48619ef67cc7f8d48fb665ae7
```

Второй рецепт требует три фоновых голоса для hard semantic carving вместо двух. Остальные выбранные параметры сохраняются. Старые запущенные кадры привязаны к исходному рецепту через immutable `recipe_binding.json`. Не менять запущенные numerical specs или четыре sealed файла common.py, runtime.py, worker.py, worker_background.py.

## Бита 923/988

Причина дырок подтверждена диагностикой. Ошибочные фоновые маски двух камер вырезали voxel cells биты из hard admission grid, обнуляя density. Изменение ARM не исправляло дефект. Глобальное раскрытие admission на старых весах давало цветные артефакты.

Порог три голоса проверен холодными 8k/16k ablations. 923 и 988 независимо переобучены до 150k, прошли export audit и просмотр шести Full HD видов. На GC биты цельные. Ореолы и фрагменты фона сохраняются.

Использовать исправленные версии:

```
/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/repairs_votes3_20261110/frames/000923
/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/repairs_votes3_20261110/frames/000988
```

SHA256 исправленных inference.ckpt:

- 923 — `048320d107643858fef5b134d738cf43c58c932649bd3c7b0edaace718b94919`.
- 988 — `2a19cef44182e27a721f4e85a099ebf71a195c43d4f07bf429a62dbf5da21a52`.

У обеих есть `quality_verified.json`. `priority_quality_requirements.json` теперь удовлетворён. При сборке итоговой последовательности явно выбрать исправленные версии. Не считать две версии одного кадра двумя кадрами. Общий combine audit пока требует отдельного выбора этих overrides.

## Проверка и рендер

Использовать `tools/audit_exports.py` для numerical/export audit. Пример:

```bash
python3 tools/audit_exports.py --root "$PWD" --queue agents/gpu0/queue.json --output agents/gpu0/export_audit.json
```

Для перенесённых данных Clever требуется актуальная копия его queue и `--frames-dir external_models/clever`. Копия очереди может устареть после work stealing.

`tools/render_fullhd.py` строит Full HD 1920×1080 с исходным FOV, RGB и straight RGBA. Реальные камеры CB/GC/NB и три interpolated вида с `--novel-fraction .25`. Проверять SHA в render_receipt, cap 10 и output_canvas.

Основная метрика — equal-person/equal-view actor LPIPS, без фона. В отчётах использовать PSNR, SSIM, LPIPS. Ореолы пользователь считает переносимыми. Дыры в реквизите требуют исправления. У 1143/1153/1173 и некоторых соседних кадров остаются заметные плавающие фрагменты фона над людьми.

## Что делать следующей сессии

1. Прочитать этот файл и локальные AGENTS.md, Paper LookCloser.md, architecture.md.
2. Проверить, появились ли новые модели после snapshot. Последний зафиксированный snapshot 9+9 файлов и дополнительный чекпоинт 1108 уже проверены. Все 72 native datasets сохранены.
3. Проверить живые controllers/agents, прогресс и OOM. Пользователь просит паузы минимум 15 минут между рутинными проверками.
4. Следить за завершением 72 кадров, аудитом экспортов и финальными рендерами. Отдельно публиковать счётчик обученных и визуально проверенных моделей.
5. Проверить учёт workstation после их отключения в `evacuated_hosts/`. Незавершённые или неподтверждённые задания будут назначены другому GPU.
6. Собрать трёхсекундную последовательность из 72 уникальных моделей, используя исправленные 923/988.
7. Для остальных кадров исходной кампании проверить источники данных перед продолжением. Сохранённый перенос гарантирует входные данные для 72 кадров сетки 24 fps.

Пароли, токены GitHub и облачные ключи в этот документ не включены.

## Проверка восстановления экспортов на silver-sable

Дополнительно работает `marshawn-emergency-frame-support.service`. Она сохраняет точные admission grids, frequency maps, semantic/head/core arrays, recipe и runtime manifests в `external_models/ws07/<frame>` и `external_models/ws08/<frame>`. RGB и sampling masks используют уже проверенные канонические native datasets. Проверять `support_preserved.json` для каждого архива и `emergency_workstations_20261111/frame_support_complete.json` для завершения прохода. Проход завершён для 18 кадров, включая 17 завершённых экспортов и input/support активного кадра 1108. Проверка всех найденных admission grids прошла без ошибок. Включены также ранее обученные кадры 1112/1116 вне сетки 24 fps. Планировщик учитывает SHA-проверенные архивы, появившиеся после исходного snapshot.

Модель 998 восстановлена из перенесённой копии на silver-sable. Три исходных Full HD вида WS07 сравнивались с новыми CB/GC/NB. Максимальное отклонение один код 8-bit RGB, различаются 17 цветовых компонентов из 18,662,400. Результаты находятся в `external_models/ws07/000998/restoration_check/` и в git-отчёте `experiments/marshawn_workstation_backup_20261111/restoration998/`.

## Последняя передача, 21:55 UTC

Остаются активными 1128, 1213 и 1218, все три на silver-sable; worker identity проверена, новых OOM не найдено. 1108 на WS07 и 1183 на Olive завершили обучение и ожидают финального визуального просмотра. 1158 проверен в шести Full HD видах: обрезания исполнителей боксом не видно, ореолы и фоновые фрагменты сохраняются. Финальные веса 1108 дополнительно эвакуируются после завершения обучения; старый полный checkpoint step79999 уже сохранён.

В `preserve_frame_support.py` исправлен ранний выход по receipt без весов: повторный запуск теперь подхватывает финальный export кадра, который был ещё активен при первой эвакуации.

На silver-sable также установлен `marshawn-final-workstation-export.timer`: повторная проверка и захват позднего финального export через 15 минут после передачи; связанная служба запускается от ubuntu и имеет Restart=on-failure. Проверять `external_models/ws07/001108/support_preserved.json`: после успешного переноса ожидается `weights_present:true`. Пока передача идёт, не считать финальные веса 1108 эвакуированными только по наличию manifest.

## Обновление 22:31 UTC

Все 72 sparse24 задачи обучены; глобальная dense gate автоматически открылась, промежуточные кадры уже обучаются. 1108 и 1183 визуально проверены, 1128 тоже; для 1213 и 1218 запущены финальные шесть видов. Финальный export 1108 теперь успешно сохранён, receipt `weights_present:true`; все 18 исходных кадров workstation имеют веса в точных архивах.

Для сохранения исходной плотной цели запущена `marshawn-emergency-data361.service`: тот же проверенный native transfer, два CPU workers, диапазон 908–1268 включительно, receipts `emergency_workstations_20261111/data361/`, итог `data361_complete.json`. Автоматический restart при ошибках; резерв свободного места минимум 500 GiB. На запуске /cache свободно 2.1 TiB; пример native кадра занимает 2.3 GiB. Полный перенос 361 пока НЕ подтверждён: до его завершения исходные workstation ещё нужны для недостающих кадров. Это дополнение не меняет уже проверенные 72 набора.

## Финальный просмотр sparse24

1213 и 1218 проверены в шести Full HD видах, явного обрезания людей боксом нет, ореолы и фоновые фрагменты остаются. Для всех 72 кадров собран индекс предпочтительных модельных SHA в `experiments/marshawn_908_1268_sparse24_selected_models.json`, копия на silver-sable в корне кампании. Он явно выбирает исправленные 923/988. Индекс основан на сохранённых export/visual-review доказательствах; актуальное наличие всех весов на одном хосте требует отдельного storage audit. Исходная цель 361 модели ещё не завершена.

## Сбор весов на одном хосте

Storage audit `sparse24_storage_audit.json` на silver-sable подтвердил SHA 67/72 выбранных весов. Пять отсутствующих на этом хосте моделей находятся на Olive: 953, 1053, 1098, 1158, 1183. Для них запущена `marshawn-sparse24-olive-archive.service`, два worker, Restart=on-failure. Цель `external_models/olive/<frame>`; итог `emergency_workstations_20261111/frame_support_olive.json`. После завершения повторить потоковый SHA audit всех 72. Наличие исходных весов на Olive подтверждалось финальными рендерами, но это не заменяет подтверждение их копии на silver-sable. Перенос data361 дошёл до 18 проверенных кадров на 23:02 UTC.

## Готовое превью 24 fps

Собраны все 72 выбранных GC рендера в `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/operations/sparse24_review_20261111/Marshawn908_1263_GC_24fps_fullhd.mp4`. 1920×1080, 24 fps, 3 секунды; совпадение render receipt/model SHA проверено для каждого кадра, исправленные 923/988 включены. Receipt с SHA видео и изображений: `experiments/marshawn_908_1268_sparse24_preview_receipt.json`. Google Drive: https://drive.google.com/file/d/11MyVEOnvsiXGFrcAR0hnQdUFlSLoD9h0/view . Контакт из 12 временных срезов просмотрен: ориентация и расположение исполнителей согласованы, контурные и фоновые артефакты присутствуют. Это review preview, не clean matte.

## Обновление после остановки workstation

Службы `marshawn-sequence-agent.service` на WS07/WS08 остановлены TERM примерно 23:26 UTC, фактические нейронные процессы 917/1119 отсутствуют. Источники данных ещё доступны, службы не перезапускались. Старые agent_state/progress сами по себе не доказывают живое обучение.

На silver-sable теперь повторно подтверждены все 72 выбранных веса по потоковому SHA, включая пять экспортов Olive; `sparse24_storage_audit.json` имеет verified=72, missing=[] (проверять results).

Добавлен `tools/dense_scheduler.py`, служба `marshawn908-dense-scheduler.service`: работает после sparse24 gate, сохраняет 361 уникальный frame ID, использует существующие durable queue transactions. На остановленных workstation сначала обновляет queue/state/completion cache, затем retirement сохраняет только подтверждённые архивами завершения. Незавершённые кадры снова доступны work stealing. Чужие активные GPU jobs не переносит и не перезапускает. Нейронные файлы и рецепты не менялись. 11 unit tests пройдены: три dense scheduling, пять priority и три retirement.

Dense scheduler подтвердил 91/361 завершение и retirement WS07/WS08 на 23:57 UTC. Все 10 оставшихся фактических нейронных PID живы: восемь Silver около 43k, Clever1111 около48k, Olive942 около35k. Полный native перенос остаётся активен, подтверждены 32/361; его завершение ещё не заявляется. Для checks исключать retired hosts по `evacuated_hosts/<host>/offline.json`, но не считать старые workstation progress/PID живыми.

## Дополнительная защита плотных данных

Для параллельного переноса второй половины запущена `marshawn-emergency-data361-ws08.service`: frames1112–1268, четыре CPU workers; исходная `marshawn-emergency-data361.service` продолжает весь диапазон с двумя. Они используют те же per-frame leases; итог половины имеет отдельное имя `data361_1112_1268_complete.json`, не подменяет итог361. Проверка готового native вынесена под lease, чтобы не валидировать ещё записываемую соседним процессом папку.

`marshawn-emergency-source-aux.service` сохраняет полноразмерные 8K masks и per-frame COLMAP texts для всего908–1268 в `source_archive/raid/lookcloser`. Проверяется ожидаемое количество 361×72 масок и SHA каждого файла; resumable batches по864 файла, два хоста параллельно. Ожидаемый итог `emergency_workstations_20261111/source_aux_complete.json`. Это защита результатов SAM3 на случай раннего удаления workstation. Полные исходные PQ16 PNG остаются на S3; для их повторной подготовки после удаления workstation ещё потребуется настроить S3 source fallback. Пока ни итог data361, ни итог source_aux не подтверждены.

## Общие seed clouds для плотных кадров

При переносе дополнительных архивных кадров обнаружен отсутствующий на Silver общий `seed_points/`: native-наборы72 уже содержали собственные clouds, но1112 требовал общего источника. Запущена `marshawn-emergency-seed361.service`, копирует оригинальные361 clouds из WS07 и проверяет каждый SHA. Ожидаемое доказательство `emergency_workstations_20261111/seed361_complete.json`. Проверка архивных native теперь использует только этот подтверждённый seed manifest и проверяет SHA выбранного seed перед копированием; частично скопированный общий каталог не считается готовым. Ошибка вызвала автоматические повторы WS08 data service, обучение моделей продолжалось. После seed receipt повтор должен пройти дальше1112.

Seed evacuation завершена 2026-11-12 02:04 UTC: `seed361_complete.json` подтверждает361 исходный cloud по SHA, каталог `/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/seed_points`. Условие для копирования недостающих seeds архивных native теперь выполнено. Текущие десять NN PID проверены 02:24 UTC, все живы и продвигаются, OOM нет; Silver около112k, Clever1111 около123k, Olive942 около98k.

Кадр942 завершён на Olive, экспорт прошёл аудит и визуальный просмотр шести Full HD ракурсов; явного обрезания боксом нет, ореолы/фоновые фрагменты сохраняются. Всего101 обученная модель, последние завершения проверены; исходная цель361 остаётся активной.

Архив WS08 SAM3 masks/calibration завершён 2026-11-12 06:31 UTC:11304 полноразмерные маски для157 кадров1112–1268 и628 калибровочных text files, все SHA совпали. Receipt `source_aux_ws08_complete.json`. Общий `source_aux_complete.json` ещё ожидает WS07. На проверке около06:40 UTC все10 NN живы, OOM нет; native data361 receipts74 (это прогресс отдельного полного прохода, дополнительно все72 sparse24 datasets уже были проверены раньше). Свободно около1.8 TiB /cache.
