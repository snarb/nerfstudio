# Forensic-отчёт: полная траектория лидера LookCloser 007740

## Краткий вывод

Архивный LPIPS-лидер для сцены 007740 — checkpoint `A_fw03@106316`.

Его точная весовая ancestry:

`A-from-scratch@75940` → continuation с Feature Reweighting (FR) `0.3` и FAS → `A_fw03@106316`.

Независимая повторная оценка этого checkpoint дала:

| PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|
| 29.618 | 0.668 | 0.231 |

Это не старая ветка без FAS, для которой в документации встречается другой набор метрик:

| PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|
| 29.858 | 0.695 | 0.272 |

Следовательно, формулировка «лучший LookCloser» неоднозначна без указания метрики:

- `A_fw03@106316` — архивный лидер по LPIPS;
- старая no-FAS lineage имеет более высокие PSNR и SSIM, но хуже LPIPS;
- эти строки нельзя объединять в один checkpoint или одну training lineage.

## 1. Область и метод проверки

Отчёт разделяет четыре разных вида истории:

1. точную ancestry весов;
2. исследовательскую и tuning-историю;
3. старую no-FAS lineage;
4. сырую траекторию метрик фактического source run.

Такое разделение обязательно: рецепт, повлиявший на выбор гиперпараметров,
не является доказательством того, что конкретный checkpoint был получен из весов этого рецепта.

Проверяемый артефакт лидера находится в:

`/home/brans/lookcloser_temporal_artifacts/static_lookcloser_leader_007740/`.

Локальная копия source run находится в:

`/home/brans/lookcloser_temporal_artifacts/source_run_007740_fromscratch_s42_A_fw03/`.

Документальные источники находятся в static-репозитории:

`/home/brans/repos/nerfstudio_static_lookcloser/LookCloser/experiments/`.

## 2. Exact weight ancestry

### 2.1. Установленная цепочка

Весовая цепочка лидера имеет два этапа:

1. родительский run `A-from-scratch`, checkpoint на шаге `75940`;
2. continuation `A_fw03`, дошедший до checkpoint `106316`.

Continuation использовал:

- Feature Reweighting с коэффициентом `0.3`;
- FAS;
- унаследованные веса Phase A как стартовую точку.

Итоговый объект сравнения — именно веса шага `106316`, а не конфигурационный рецепт
и не метрики родительского шага `75940`.

### 2.2. Что ancestry не означает

Ancestry не доказывается сходством названий экспериментов.

Она также не может быть восстановлена простым объединением лучших чисел из разных таблиц.

Исследовательские пробы до Phase A объясняют выбор модели, но не входят автоматически
в прямую цепочку инициализации весов.

### 2.3. Проверка parent run

`/fsx` не смонтирован локально, однако основной forensic-проход прочитал parent run
read-only через `ssh ubuntu@dev3`. Первично проверены `config.yml`,
`metrics_compact.csv`, `train_stdout.log`, список checkpoint и их размеры/mtime.
Поэтому таблицы parent trajectory ниже основаны на raw CSV, а не на пересказе Markdown.

## 3. Research/tuning history

Исследовательская история шире точной ancestry весов.

Она включает:

- исходную Phase-A репродукцию from scratch;
- анализ бюджетного ARM-рецепта;
- выбор continuation с Feature Reweighting;
- включение FAS;
- сравнение perceptual и distortion-метрик;
- архивирование кандидатов и повторную статическую оценку.

Документы `budget_arm_recipe.md` и `from_scratch_reproduction_recipe.md`
следует читать как контекст проектирования и воспроизводимости.

Они не заменяют `config.yml`, checkpoint metadata и сырые eval JSON конкретного лидера.

`fromscratch_repro_phase_a.md` является важным вторичным источником для родительской Phase A,
но его численные утверждения ниже перепроверены по первичному parent CSV через dev3.

## 4. Старая no-FAS lineage

Старая линия без FAS должна быть сохранена как отдельная ветка сравнения.

Ей соответствует документированный результат:

`PSNR 29.858 / SSIM 0.695 / LPIPS 0.272`.

По PSNR и SSIM эта строка лучше повторной оценки LPIPS-лидера.

По LPIPS она хуже: `0.272` против `0.231`.

Следовательно:

- no-FAS ветка может быть distortion-лидером;
- она не является архивным LPIPS-лидером;
- её метрики нельзя подписывать именем `A_fw03@106316`;
- её lineage нельзя описывать как continuation с FAS.

## 5. Raw metrics trajectory

Первичный источник траектории continuation:

`source_run_007740_fromscratch_s42_A_fw03/metrics_compact.csv`.

Лог выполнения:

`source_run_007740_fromscratch_s42_A_fw03/train_stdout.log`.

Эти файлы имеют приоритет над округлёнными числами в обзорных Markdown-документах
при ответе на вопросы о шагах, eval boundaries и динамике метрик.

Сырая траектория не должна подменяться одной строкой «best».

Для каждого eval boundary необходимо различать:

- training step;
- PSNR;
- SSIM;
- LPIPS, если он записан;
- источник значения: online eval или отдельный `ns-eval`.

Повторная статическая оценка checkpoint `106316` дала `29.618 / 0.668 / 0.231`.

Это наиболее надёжная строка для сопоставления архивированных candidate weights
в едином evaluator-контексте.

Возможные несовпадения с online-метриками не следует автоматически считать ошибкой весов:
они могут возникнуть из-за различий evaluator path, набора изображений, масок,
округления или версии кода.

## 6. Exact config лидера

Первичный источник exact config:

`static_lookcloser_leader_007740/config.yml`.

Source-run копия для перекрёстной проверки:

`source_run_007740_fromscratch_s42_A_fw03/config.yml`.

Ключевой forensic-факт exact config:

- `feature_reweighting_strength = 0.3` при `enable_feature_reweighting = true`.

Также установлен факт применения FAS в continuation.

Любые прочие поля необходимо цитировать из YAML буквально и с учётом вложенности.

Нельзя считать значение, описанное в `LEADER_INFO.md`, более точным,
если оно прямо противоречит архивированному `config.yml` фактического run.

## 7. Противоречия источников

### 7.1. Feature Reweighting: 1.0 против 0.3

`LEADER_INFO.md` сообщает `FR=1.0`.

Архивный `config.yml` сообщает `FR=0.3`.

Для exact run config приоритет имеет YAML фактического run.

Вывод: `FR=1.0` в `LEADER_INFO.md` — ошибочная или устаревшая аннотация,
а не конфигурация весов `A_fw03@106316`.

### 7.2. Смешение лидеров в документации

Обзорные документы смешивают как минимум два понятия:

- лучший результат по PSNR/SSIM;
- лучший результат по LPIPS.

Строка `29.858 / 0.695 / 0.272` относится к старой no-FAS lineage.

Строка re-eval `29.618 / 0.668 / 0.231` относится к `A_fw03@106316`.

Составлять «идеальную» строку `29.858 / 0.695 / 0.231` нельзя:
таких совместно измеренных метрик для одного checkpoint данным forensic не установлено.

### 7.3. Имя кандидата против фактического содержания

Имя `A_fw03` согласуется с `FR=0.3` и поддерживает чтение YAML.

Однако имя — лишь corroborating evidence, а не самостоятельное доказательство.

Решающими остаются config, checkpoint identity и candidate eval JSON.

## 8. Инвентарь источников

| Источник | Роль | Класс |
|---|---|---|
| `static_lookcloser_leader_007740/config.yml` | exact config архивного лидера | первичный |
| `static_lookcloser_leader_007740/eval_results.json` | итоговая статическая оценка | первичный |
| `static_lookcloser_leader_007740/all_candidate_evals/*.json` | сравнение кандидатов | первичный |
| `static_lookcloser_leader_007740/LEADER_INFO.md` | человекочитаемая аннотация | вторичный, противоречивый |
| `source_run_007740_fromscratch_s42_A_fw03/config.yml` | config source continuation | первичный |
| `source_run_007740_fromscratch_s42_A_fw03/metrics_compact.csv` | raw online trajectory | первичный |
| `source_run_007740_fromscratch_s42_A_fw03/train_stdout.log` | chronology и runtime evidence | первичный |
| `fromscratch_repro_phase_a.md` | parent Phase-A evidence | вторичный |
| `budget_arm_recipe.md` | tuning/research context | вторичный |
| `from_scratch_reproduction_recipe.md` | recipe context | вторичный |
| `experiments_overview.md` | сводная навигация | третичный/обзорный |

## 9. Градация доказательств

### Grade A — прямые первичные артефакты

- фактический YAML run;
- eval JSON конкретного checkpoint;
- candidate eval JSON;
- raw metrics CSV;
- runtime log с шагами и путями.

### Grade B — документация конкретного эксперимента

- Phase-A отчёт;
- recipe-документы;
- human-authored leader note без полного первичного подкрепления.

### Grade C — обзор и косвенные признаки

- experiments overview;
- имена run и сокращения вроде `fw03`;
- агрегированные таблицы, где lineage не указана явно.

При конфликте Grade A должен иметь приоритет над Grade B/C.

## 10. Пробелы и ограничения

1. `/fsx` не смонтирован локально; parent проверен read-only через dev3, но не скопирован в этот архив.
2. Run-артефакты не записали точный git commit; версия executable-кода выводится из времени и git-history,
   но не имеет Grade-A provenance.
3. Не следует считать online eval и static re-eval взаимозаменяемыми без проверки evaluator context.
4. Для окончательной chain-of-custody parent checkpoint желательно также архивировать с SHA-256.

## 11. Каноническая формулировка для будущих отчётов

«Архивный LPIPS-лидер сцены 007740 — `A_fw03@106316`, полученный continuation
от `A-from-scratch@75940` с Feature Reweighting `0.3` и FAS.
Его единая повторная оценка: PSNR `29.618`, SSIM `0.668`, LPIPS `0.231`.
Результат `29.858 / 0.695 / 0.272` принадлежит отдельной старой no-FAS lineage
и не должен смешиваться с ancestry или метриками LPIPS-лидера».

## 12. Практический вывод

Для perceptual baseline следует использовать `A_fw03@106316` и строку static re-eval.

Для distortion-сравнения допустимо отдельно показывать no-FAS результат,
но с явной подписью lineage и без переноса его PSNR/SSIM на LPIPS-лидера.

При воспроизведении continuation стартовать следует от Phase-A checkpoint `75940`,
использовать exact archived config с `FR=0.3` и включённым FAS,
а не значение `FR=1.0` из противоречивой аннотации `LEADER_INFO.md`.

## 13. Дополнение после первичной проверки parent run на dev3

Эта секция заменяет более осторожную реконструкцию выше там, где основной forensic-проход
получил первичные данные read-only через `ssh ubuntu@dev3`.

Parent run:

`/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/lookcloser_runs/007740_fromscratch_s42_A/lookcloser/20260623_142529/`

Continuation source run:

`/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/fromscratch_repro/007740_fromscratch_s42_A_fw03/lookcloser/20260624_002610/`

Архивированная локальная копия continuation:

`/home/brans/lookcloser_temporal_artifacts/source_run_007740_fromscratch_s42_A_fw03/`

### 13.1. Полная raw trajectory точной ancestry лидера

`Cumulative points` для старых CSV оценены как сумма логируемого раз в десять шагов
`train_num_samples_per_batch × 10`. Это достаточно точная оценка exposure, но не побитовый счётчик.
В continuation-строках показана сумма parent exposure и exposure продолжения.

| Этап | Step | Approx cumulative points | LR | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| A from scratch | 15,188 | 17.565 B | 0.007050 | 28.5960 | 0.651726 | 0.371653 |
| A from scratch | 30,376 | 50.671 B | 0.004971 | 29.2098 | 0.676160 | 0.305969 |
| A from scratch | 45,564 | 87.566 B | 0.003504 | 29.3952 | 0.673553 | 0.279821 |
| A from scratch | 60,752 | 126.588 B | 0.002470 | 29.5279 | 0.677030 | 0.262007 |
| A from scratch, выбранный parent | 75,940 | 166.919 B | 0.001742 | 29.6217 | 0.675272 | 0.252857 |
| A_fw03 continuation | 91,128 | ~208.140 B | 0.001228 | 29.6920 | 0.672744 | 0.240396 |
| A_fw03 continuation, archive | 106,316 | ~250.035 B | 0.000866 | 29.6180 | 0.668451 | 0.231120 |

Parent начал запись config `2026-06-23 14:25:36`, а checkpoint `75940` записан
в `20:15:40`: около `5 ч 50 мин`. Continuation записал config `2026-06-24 00:26:17`,
а checkpoint `106316` — в `02:24:21`: около `1 ч 58 мин`. Итого около `7 ч 48 мин`
активного controller/training времени, не считая паузы между run. Эти времена относятся к dev3,
а не к локальной Blackwell, поэтому используются только для масштаба, не как benchmark.

Parent попытался сохранить checkpoint `91128`, но `torch.save` упал с
`PytorchStreamWriter failed writing file` из-за storage failure. Файл имеет меньший размер
и не является надёжным parent. Поэтому continuation корректно стартовал с последнего целого
checkpoint `75940`.

### 13.2. Exact config: этап A от нуля

Первичный `config.yml` parent подтверждает:

| Группа | Значение |
|---|---|
| Initialization | `load_checkpoint: null`, seed `42` |
| Optimizer | Adam, base LR `0.01`, exponential decay до `0.0001`, `max_steps=200000`, cosine ramp |
| Rays | fixed `train_num_rays_per_batch=4096` |
| FAS | on с шага 0, strength `1.0`, warmup `0`, ramp `0`, distribution `1→3` |
| FR | on с шага 0, strength `1.0` |
| Frequency grid | on, resolution `128`, update batch `2048`, interval `1024` |
| ARM | adaptive, coarse `0.00625`, min/max step `0.0001/0.1`, warmup `4096`, cap `1024` |
| Frequency bounds | min `0.0`, max `null` |
| Hash field | hash23, 16 levels × 2 features, `max_res=8192`, `max_res_base=2048` |
| Occupancy | threshold `0.01`, EMA `0.95`, update `16`, warmup/binary warmup `4096/4096` |
| Losses | Charbonnier RGB, distortion `0.01`, depth `0.001` первые `5000` steps |
| Precision | mixed precision on |
| Data | 66 train + 3 eval, filename split, scene scale `1.5`, focus/up, scale factor `1.0` |

Таким образом, текущий архивный лидер **не** формировал геометрию без FAS/FR. FAS и FR=1
влияли на оптимизацию с нулевого шага; первые `4096` steps были warmup ARM, а не отдельная
долгая geometry-only фаза.

### 13.3. Exact config: A_fw03 continuation

Continuation загрузил:

- `step-000075940.ckpt` из Run A;
- model weights;
- Adam state (`load_optimizers: true`);
- exponential scheduler state (`load_scheduler: true`).

Он сохранил FAS `1.0`, ARM, frequency grid, batch `4096`, losses и модель без смены capacity.
Существенное намеренное отличие — FR strength `1.0 → 0.3`. LR не был вручную сброшен:
raw CSV показывает продолжение той же экспоненциальной кривой (`0.001742 → 0.001228 → 0.000866`).

Контроль `A_fw00` загрузил тот же parent, также сохранил FAS `1.0`, но отключил FR.
Fresh re-eval на step `106316`:

| Candidate | FR | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| A_fw03 | 0.3 | 29.617966 | 0.668451 | 0.231120 |
| A_fw00 | off | 29.589718 | 0.667971 | 0.231929 |

Последний FR-switch объясняет только `−0.00081` LPIPS в этом сравнении. Оба кандидата
унаследовали parent, обученный с FR=1 и FAS=1, поэтому это не абляция роли раннего FR/FAS.

## 14. Две разные исторические lineage

### 14.1. Старая no-FAS PSNR/SSIM lineage

Эта ветка возникла до архивного A_fw03 и использовалась для открытия работающего рецепта:

1. ARM-native модель без FR дошла до `maxfreq13_cont3@38912`:
   `29.535 / 0.689 / 0.401`.
2. Budget-aware ARM + FR=1 + Charbonnier, FAS off:
   - step `45564`: `29.338 / 0.682 / 0.367` — адаптационный dip;
   - step `60752`: `29.715 / 0.694 / 0.343`;
   - step `75940`: `29.911 / 0.703 / 0.326`.
3. Из step `75940` сделали два fork:
   - FAS on: `0.277` на `91128`, `0.258` на `106316`, `0.249` на `121504`,
     при постепенном падении PSNR до `29.750`;
   - FAS off: `29.917 / 0.700 / 0.280` на `91128`, затем
     `29.858 / 0.695 / 0.272` на `106316`.

Именно здесь находится упомянутый эксперимент, где LPIPS явно упал после включения FAS:
`0.326 → 0.249`, но ценой PSNR/SSIM. На одинаковом step `106316` разница FAS против no-FAS
составляла примерно `0.258` против `0.272`, то есть чистый observed gap около `0.014`;
остальная часть падения от `0.326` включает дополнительные training steps.

### 14.2. Точная lineage архивного LPIPS-лидера

Позже Run A проверил современный default **с нуля**: FAS=1 и FR=1 с самого начала.
Он не является продолжением `maxfreq13_cont3`; его `load_checkpoint` равен `null`.
После step `75940` от него были сделаны FR continuation, и `A_fw03@106316` выиграл
архивный fresh re-eval по LPIPS. Это отдельная весовая цепочка.

### 14.3. Почему старые Markdown вводили в заблуждение

- `from_scratch_reproduction_recipe.md` называет reproducible leader строкой
  `29.858 / 0.695 / 0.272`, хотя archive selection позднее выбрал другой checkpoint.
- `experiments_overview.md` помещает `A_fw03` под заголовок budget-ARM + FR и рядом обсуждает
  no-FAS lineage, что визуально создаёт одну историю.
- `LEADER_INFO.md` правильно указывает source run `A_fw03`, но ошибочно пишет FR strength `1.0`.
- Архивный YAML однозначно фиксирует FR `0.3`; свежие candidate JSON однозначно фиксируют
  `29.618 / 0.668 / 0.231`.

## 15. Наша локальная Blackwell trajectory

Первичный отчёт и raw runs:

- `/home/brans/repos/nerfstudio_static_lookcloser/LookCloser/experiments/static_blackwell_reproducibility.md`;
- `/home/brans/lookcloser_static_runs/007740_static_acceptance_60m/`;
- `/home/brans/lookcloser_static_runs/007740_static_plateau_fork/`;
- `/home/brans/lookcloser_static_runs/007740_static_occupancy_quality/`.

### 15.1. Рецепт

Локальный controller был оптимизирован под milestone ≤60 минут и clean gate, а не под точное
повторение старого dev3 run:

1. `geometry`: occupancy traversal, frequency grid/FR/FAS off, LR `0.01`;
2. `frequency`: corrected capped ARM + grid, LR `0.005`;
3. `feature`: FR ramp до `0.3` за `4096` updates, LR `0.0025`;
4. `perceptual`: FAS ramp `0→1` за `4096` updates, LR `0.0025`;
5. point target `2^19 = 524288` с dynamic ray batch вместо fixed 4096 rays.

На каждом переходе Adam загружался, но scheduler не загружался; LR задавался controller вручную.
Использовался новый deterministic RNG split и исправленный ARM allocator
minimum-one + largest remainder с детерминированным merge при `intervals > cap`.

### 15.2. Все scheduled full-eval boundaries автоматического run

| Phase | Step | Cum. points | LR | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| geometry | 2,048 | 1.074 B | .0100 | 24.8011 | .542042 | .627769 |
| geometry | 4,096 | 2.147 B | .0100 | 23.1882 | .617630 | .590839 |
| geometry | 6,144 | 3.213 B | .0100 | 27.3448 | .607338 | .512669 |
| geometry | 8,192 | 4.287 B | .0100 | 27.7616 | .621483 | .485506 |
| geometry | 10,240 | 5.361 B | .0100 | 28.2451 | .631753 | .472926 |
| geometry | 12,288 | 6.434 B | .0100 | 28.3813 | .636181 | .464909 |
| geometry | 14,336 | 7.508 B | .0100 | 28.5838 | .644045 | .455393 |
| geometry | 16,384 | 8.582 B | .0100 | 28.6847 | .646064 | .452636 |
| geometry | 18,432 | 9.655 B | .0100 | 28.8997 | .649530 | .444195 |
| geometry | 20,480 | 10.729 B | .0100 | 28.9999 | .651740 | .437671 |
| geometry | 22,528 | 11.803 B | .0100 | 29.0021 | .657701 | .438750 |
| frequency | 24,576 | 12.879 B | .0050 | 29.1291 | .656846 | .439343 |
| frequency | 26,624 | 13.956 B | .0050 | 29.1309 | .659046 | .432228 |
| frequency | 28,672 | 15.030 B | .0050 | 29.2296 | .663018 | .426300 |
| frequency | 30,720 | 16.105 B | .0050 | 29.2759 | .664696 | .425586 |
| frequency | 32,768 | 17.179 B | .0050 | 29.3349 | .672592 | .424710 |
| frequency | 34,816 | 18.253 B | .0050 | 29.4349 | .668470 | .421242 |
| feature | 36,864 | 19.327 B | .0025 | 29.5347 | .671782 | .418659 |
| feature | 38,912 | 20.401 B | .0025 | 29.6149 | .672755 | .418121 |
| feature | 40,960 | 21.475 B | .0025 | 29.7287 | .673393 | .418338 |
| feature | 43,008 | 22.549 B | .0025 | 29.6144 | .674960 | .414065 |
| feature | 45,056 | 23.622 B | .0025 | 29.6923 | .676910 | .413228 |
| feature | 47,104 | 24.696 B | .0025 | 29.8094 | .678099 | .413834 |
| feature | 49,152 | 25.770 B | .0025 | 29.7520 | .676477 | .412450 |
| feature | 51,200 | 26.844 B | .0025 | 29.6655 | .679317 | .411954 |
| feature | 53,248 | 27.918 B | .0025 | 29.8695 | .678668 | .408684 |
| feature | 55,296 | 28.992 B | .0025 | 29.8376 | .678306 | .407483 |
| feature | 57,344 | 30.065 B | .0025 | 29.7775 | .680639 | .405418 |
| perceptual | 59,392 | 31.139 B | .0025 | 29.9093 | .681085 | .403497 |
| perceptual | 61,440 | 32.212 B | .0025 | 29.9164 | .680060 | .404825 |
| perceptual | 63,488 | 33.286 B | .0025 | 29.7253 | .679227 | .404028 |

Controller остановился на plateau с full-resume step `63554`. Полная финализация заняла
`2401.8 s` (`40.0 min`). Автоматически выбран step `59392`; он имел один significant artifact
и провалил detail gate.

Продолжение до step `91870` накопило `48.166 B` points. Лучший numeric step `88064`:
`29.8985 / 0.695282 / 0.375183`, но с двумя significant artifacts и крупными чёрными/missing
областями. Это visual fail.

### 15.3. Прямое сравнение checkpoints

| Checkpoint | Points | PSNR | SSIM | LPIPS | Significant artifacts |
|---|---:|---:|---:|---:|---:|
| Archived A_fw03@106316, local full precision | ~250.0 B | 29.617964 | .668450 | .231135 | 2 views |
| Blackwell automatic@59392 | 31.139 B | 29.9093 | .681085 | .403476 | 1 view |
| Blackwell long diagnostic@88064 | ~46 B | 29.8985 | .695282 | .375183 | 2 views |

Архив получил примерно `8.0×` больше point exposure, чем автоматический selected checkpoint,
и примерно `5.2×` больше, чем весь long diagnostic boundary (`48.166 B`). Поэтому одинаковый
номер optimizer step не является одинаковым training budget.

## 16. Почему мы не достигли LPIPS лидера

### 16.1. Причины высокой уверенности

1. **Недостаточный cumulative exposure.** Лидер видел ~`250 B` points, автоматический run —
   `31 B`. Цель ≤60 минут была радикально жёстче фактического ~7.8-часового пути лидера.
2. **FAS включён слишком поздно для повторения лидера.** У лидера FAS=1 работал с нуля.
   У нас FAS warmup завершался только после global step `58446`; до selected step он успел
   влиять менее пяти тысяч updates. Geometry-first сознательно оптимизировался под устойчивость,
   но не повторяет perceptual trajectory лидера.
3. **FR history другая.** Parent лидера обучался с FR=1 до `75940`, затем FR снизили до `0.3`.
   У нас FR был off до `34816`, после чего ramp только до `0.3`.
4. **Это другой sampler algorithm.** Новый minimum-one/largest-remainder ARM и interval merge —
   correctness-изменения, не semantics-preserving speed patch. Старый checkpoint под новой ARM/grid
   семантикой не обязан давать ту же траекторию.
5. **Разный batch/optimizer режим.** Лидер использовал fixed 4096 rays и примерно
   `0.9–2.8 M` samples/step; у нас target `0.524 M` и около тысячи dynamic rays. На одинаковом
   cumulative point count у нас больше Adam updates, меньше rays на update и другая gradient noise.
6. **Разный LR/state path.** Лидер плавно снижал LR `0.01→0.000866` и переносил scheduler/Adam.
   Мы ступенчато использовали `.01/.005/.0025`, переносили Adam, но задавали новый constant LR
   в каждой фазе. Это не эквивалентная оптимизация.

### 16.2. Причины средней уверенности

7. **FAS LUT/metadata semantics изменены.** Новая mixed/legacy LUT correction могла менять
   фактическое распределение пикселей относительно старого run. Это правильно с точки зрения
   metadata, но требует paired old-vs-new training, чтобы измерить вклад в LPIPS.
8. **Зрелость frequency grid.** У лидера grid обновлялся с начала. У нас он был выключен всю
   geometry-фазу и начал формироваться одновременно с переходом на ARM, что согласуется с более
   слабой ранней ARM веткой.
9. **Artifact-clean selector задаёт более строгую задачу.** Архивный LPIPS-лидер сам имеет
   significant artifacts в двух eval views. Наш selector обязан был бы его отвергнуть. Часть
   низкого LPIPS режима исторически сосуществует с неприемлемыми holes/stand/cable артефактами.

### 16.3. Что почти наверняка не объясняет разрыв

- Dataset: локальный и dev3 manifest совпали (`69` images, `66+66` frequency artifacts),
  `transforms.json` SHA-256 совпадает.
- Evaluator: локальный full-precision re-eval повторил архивный JSON с дельтой около
  `2e-6 dB` PSNR, `2e-6` SSIM и `1.5e-5` LPIPS.
- Последний FR switch: A_fw03 против A_fw00 даёт всего `0.00081` LPIPS.
- Одна только FAS: исторический paired fork на step `106316` показывает около `0.014` LPIPS,
  а не весь observed gap `0.144–0.172`.

## 17. Точный рецепт весовой ancestry лидера

Это forensic-рецепт воспроизведения архивных весов, а не принятый быстрый production-рецепт.

1. Использовать dataset с filename split, `66` train + `3` eval и точными frequency maps.
2. Использовать executable-код эпохи run. Read-only reflog dev3 ставит HEAD на `fe9dd951` с
   `13:56:19 UTC`, до старта parent config в `14:25:36`; отличие `85818149→fe9dd951` состоит только
   в добавлении Markdown-рецепта. Поэтому `85818149` точно представляет committed executable tree
   на момент запуска. Run artifacts не позволяют ретроспективно исключить незакоммиченные правки,
   поэтому cleanliness worktree остаётся узким provenance gap.
3. Запустить seed42 с `load_checkpoint=null`, fixed 4096 rays, adaptive ARM warmup4096,
   frequency grid on, FAS=1 без ramp/warmup, FR=1, exact optimizer/model/data config из §13.2.
4. Продолжать ту же exponential schedule до целого checkpoint step `75940`.
5. Создать новый run, загрузить step75940 вместе с Adam и scheduler.
6. Изменить FR strength на `0.3`; FAS оставить `1.0`; не reset LR/Adam/scheduler.
7. Продолжить до `106316`, сохранить checkpoint.
8. Оценить ровно три filename-eval views full precision и отдельно прогнать artifact/detail gates.

Рецепт считается точным по config и ancestry, но не полностью hermetic до архивирования exact
source commit/container. Он также не проходит современную acceptance цель: SSIM ниже `0.695`,
significant artifacts есть в двух views.

## 18. Следующие проверки, которые различат гипотезы

1. На текущем коде paired seed42: leader-style FAS/FR-from-zero против geometry-first,
   одинаковые LR, batch и cumulative points.
2. Fixed 4096 rays (~2.5–2.8 M points/step) против p19 dynamic batch, сравнить одновременно
   при одинаковых wall-clock, optimizer updates и cumulative points.
3. Smooth exponential scheduler с полным state restore против текущих piecewise LR.
4. Старый per-ray-dt ARM против corrected allocator на одном codebase и одном seed; это
   алгоритмическая абляция и обязана пройти полный visual gate.
5. Не запускать seeds 43–46, пока seed42 не даст один clean checkpoint, одновременно прошедший
   numeric, artifact и archived-detail gates: variance sweep не исправит систематический LPIPS gap.

Главный вывод: мы не «не смогли повторить последний тюнинг». Мы обучили существенно другой,
жёстко ограниченный по времени и exposure phased recipe. Исторический LPIPS-лидер получил низкий
LPIPS в течение всей ~250B-sample FAS/FR trajectory; его финальная FR=0.3 continuation была лишь
последними ~83B samples, а не источником всего улучшения.
