# Marshawn1660: зафиксированный рецепт и итоги

Срез результатов: **7 ноября2026,11:57UTC**. По последнему указанию пользователя **остановлены все эксперименты и все обучения, без ожидания плато**. Автопродолжения отключены. Три последние холодные траектории прерваны пользователем с сохранением atomic checkpoints; это не завершённые quality-сравнения и не научная неудача. Прежний минимум20часов отменён этим последующим указанием.

## Что проверяли

Цвет и форматы исходников, калибровку, AABB двух актёров на2секунды, веса RGB и семплинг, интегрирование вдоль лучей, регуляризацию плотности, frequency maps, HD/4K, обход occupancy R0/R1 и отделение актёров посредством замороженного3D-семантического поля. Проверка включает сохранённые рендеры, участки волос/рук/ступней, реальные веса лучей и фактические occupancy buffers.

**Результат:** лучший завершённый вариант по LPIPS двух актёров — HD LookCloser, quarter-step + distortion.1, checkpoint126316. Идеально чистая геометрия и полностью корректный физический occupancy grid **не получены**: остаются ореолы, плотность между ногами и фрагменты пола около обуви.

## Результаты: победитель и точный рецепт

| Выбор | Шаг | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|---:|
| **Победитель по целевой actor-LPIPS** | **126316** | **31.639727** | **.831264** | **.018773121** |
| Исторический автоматический селектор |60752|31.810748|.837931|.019472740|
| Максимальный PSNR / LPIPS tie-break в.07dB |15188|32.091200|.854067|.022190927|

Это три разных критерия. Исторический worker сначала оставлял варианты в пределах.001 от минимального LPIPS, затем выбирал больший PSNR; **60752 не является глобальным максимумом PSNR**. Отдельный CPU-аудит всех11 записанных оценок из самого checkpoint подтвердил15188 для правила PSNR/.07dB. Выбор126316 соответствует явно заданной пользователем цели LPIPS по двум актёрам. Исторические оценки/маски/селекция не переписывались.

Метрики: неизменный V1, камеры DC/HC/KA, равный вес двух актёров и трёх видов; фиксированные crop, Alex LPIPS без ресайза. Вне бинарной маски и GT, и prediction заменяются наRGB.5. Поэтому хороший actor-LPIPS сам по себе не оценивает чистоту фона или корректность3D-пустоты.

| Компонент | Зафиксированное значение |
|---|---|
| Данные | Frame1660;63TRAIN +3EVAL(DC/HC/KA);RGB1920×1012 |
| AABB в подготовленных координатах пола | min `[-.84,-.21,-.07]`, max `[.15,.42,.63]` |
| Parser | `center=none`, `orientation=none`, `auto_scale_poses=False`, scale1, downscale1, eval по filename |
| Model | LookCloser, bounded, corrected SH directions, density `trunc_exp` |
| Hash encoding |16уровней ×2features, min16, фактический max8192 (`max_res_base=2048`), log2hash23 |
| RGB | Confidence-weighted Charbonnier; W64; аналитический нормализатор ожидания proposal |
| Семплинг | `body80`;4096 лучей/итерацию; seed42 |
| Установившиеся квоты body80 |1/3 core-uniform +1/3 person-FAS +2/15 protective-band +1/5 whole-valid-image |
| Ранние квоты | До4096:2/3 core +2/15 band +1/5 valid;4096→8192 половина core плавно заменяется person-FAS |
| Интегрирование | Warmup4096 с256 fixed samples; затем ARM, coarse step **.0015625**, legacy allocator, cap1024 |
| Distortion | Фактически .01 до4096; линейный переход к **.1** на4096→12288; затем .1 |
| Feature reweighting |1.0 до75940 включительно; **.3 с75941**; исходный uniform global frequency update |
| Frequency grid | Исходные карты; legacy projection, update interval1024,2048 depth samples |
| Frequency maps | Только63TRAIN, SSIM.95,1000updates/level, LR.01 constant,16уровней16…8192,patch8/stride8,SSIMwindow7 |
| Occupancy обучения | Resolution128,1level,R0,EMA.95,threshold.01,update interval16,warmup4096; исходный mapped semantic admission/pad1 |
| Optimizer | Adam, LR.01→.0001, decay horizon200000, epsilon1e-15, mixed precision |
| Длительность | Холодный seed42 от0; checkpoint126316, около4860секунд собственно обучения в исходном запуске |
| Остановка | Без22k-cap; исходные контрольные шаги до106316, далее шаг5000 и patience4 прежнего validation plateau |

**Сохранённый YAML содержит `distortion_loss_mult=.01`, но worker меняет его во время исполнения.** Запуск одного YAML через произвольный `ns-train` не воспроизводит рецепт. Также необходимо восстановить FR-switch, исходный sampler/confidence normalizer, admission и frequency cache. Значение W64 относится к исходному native-confidence контракту; его нельзя интерпретировать как новый произвольный64HD-пиксельный hard crop.

### Фотометрия и геометрия

Исходники8192×4320 — Log3G10/RWG TIFF, не готовые PQ-коды. Победитель обучен на зафиксированных SDR PNG. В них уже применены Rec709/D65, base gain4, общая радиальная **фотометрическая** гипотеза k=4 и TRAIN-only scalar EV, затем Reinhard и sRGB. Это не коэффициент дисторсии камеры и не измеренный flat-field всех камер.

Точная сохранённая поправка: `gain4_linear * 2**(2*log2(1+r2) - .4574742245269814 + camera_EV)`, где `r2=((u-cx)/fx)**2+((v-cy)/fy)**2`; далее фиксированное отображение в PNG. Не применять gain или старую per-camera scalar повторно. Frequency maps должны совпадать по SHA с этими фактическими RGB.

Подготовленные poses/XYZ уже находятся в координатах пола: world-to-box применяется один раз, parser сохраняет identity. Общий box проверялся на1660/1720/1780/1840/1900, то есть через.5секунды при исходных120fps. Выход24fps за2секунды соответствует48кадрам1660…1895 с шагом5; обучен здесь только1660. Проверенные пять поз — консервативное основание общего box, не доказательство оптимального минимального объёма или покрытия всех промежуточных движений.

### Неизменяемые источники и артефакты

- Dataset: `/fsx/tmp/marshawn_001660_hd/tuning_20261106/data`; transforms SHA `64176bce20fcf0f08427a89b492003adcce6a0b513d9fffd6bbc583db172f01a`.
- [AABB и преобразование](../configs/aabb_review/marshawn_1660_1900_floor.json); SHA `292609eebe9939e7ce5f840a1b317d963d39ec738cafaad8500abf3a2cd0620f`.
- [Фотометрический контракт](/fsx/tmp/marshawn_001660_hd/color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png/color_contract.json).
- [Фактический spec](/fsx/tmp/marshawn_001660_hd/tuning_20261106/semantic_quarter_distortion_candidate_20261107/spec.json); исходный1495-file manifest SHA `ad32a034c492314162fac8d966be2215c834393a1bb984beb05f1eed8f9c43b3`.
- [Локальный неизменяемый checkpoint126316](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_capture_plan_v2/local_source/checkpoint.ckpt); SHA `340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641`.
- Полный source pipeline digest: `c83bc7dbb1504063c35f0a2b14ae3fb329ff27ed2eb6b3661eb89b4918d46636`.
- [Исходный HC победителя](/fsx/tmp/marshawn_001660_hd/tuning_20261106/independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin/renders/HC.png); DC/KA находятся рядом.
- [CPU-рейтинг всех checkpoint оценок](/fsx/tmp/marshawn_001660_hd/tuning_20261106/root_source_checkpoint_selection_CPU.json); [сохранённые3рендера PSNR/.07-победителя15188](/fsx/tmp/marshawn_001660_hd/tuning_20261106/best_recipe_frozen_20261107/PSNR07_15188/renders/HC.png).
- [Отчёт исходной завершённой траектории](marshawn_quarter_distortion_interaction.md), [машиночитаемые ссылки на raw/clean preview](/fsx/tmp/marshawn_001660_hd/tuning_20261106/root_best_artifacts_20261107_1130.json).

## Результаты сравнения завершённых вариантов

Ниже сохранены собственные выбранные шаги разных траекторий; строки с разным возрастом не являются чистым однофакторным причинным сравнением. В подробных тематических отчётах имеются matched-step таблицы и исходные selector rankings.

| Вариант | Шаг | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Перенос закреплённого старого leader recipe, full_baseline_r0 |106316|23.134166|.657569|.11039852|
| Исходный semantic HD R0 |111316|31.518449|.838073|.02233986|
| Half-step .003125/.01 |126316|31.49708|.833326|.01961766|
| Quarter-step .0015625/.01 |111316|31.427557|.827178|.01923172|
| **Quarter-step/.1, текущий лидер** |**126316**|**31.639727**|**.831264**|**.01877312**|
| Frequency maps: per-level LR decay |126316|31.59188|.830857|.01903432|
| Frequency maps: SSIM.93, raw minimum |121316|31.64880|.832046|.01879139|
| Отдельная4K RGB+4000maps траектория |126316|31.581180|.842665|.02151916|
| Legacy warm continuation лидера |150316|31.575576|.830046|.01889728|
| Corrected-allocator warm continuation |155316|31.577419|.829857|.01881533|
| Below-floor control, без штрафа |155316|31.584046|.829860|.01880035|
| Below-floor λ1e-4 |150316|31.581187|.830152|.01891676|
| Below-floor λ1e-3 |155316|31.579324|.829835|.01886600|

Обе frequency-map альтернативы прошли полное NeRF обучение до собственного плато126316/131316. Исходные карты сохранены. На4K есть отдельная завершённая RGB+4000maps траектория; полного4K-сравнения двух новых вариантов карт не проводилось. Увеличенный max hash resolution8192 сам по себе не означает обучение на4K RGB.

Подробности: [HD frequency trials](marshawn_frequency_HD63_trials.md), [half-step](marshawn_semantic_halfstep_final.md), [quarter-step](marshawn_semantic_quarterstep_mature.md), [4K](marshawn_semantic_4k_feasibility.md), [warm allocator](marshawn_corrected_arm_warm.md), [below-floor](marshawn_below_floor_density_prior.md).

### Исходный контроль по закреплённому рецепту старой сцены

`full_baseline_r0` — перенос [static leader recipe](static_leader_reproduction_recipe.md) на новую сцену: full RGB objective, исходный FAS sampler, softplus, исторические SH directions, legacy ARM/coarse.00625, distortion.01, B4096/hash23/max8192, fixed256 warmup4096, FR1→.3 с75941 и исторический seed reset на границе с сохранением Adam/scheduler/scaler. Адаптация данных необходима: новые63/3камеры, подготовленная фотометрия и AABB/identity parser вместо старого датасета007740. Это контроль нового Marshawn по старому weight recipe, а не утверждение тождества разных датасетов.

Запуск завершился на126316; исторически выбран106316. Его [HC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/jobs/full_baseline_r0/run/renders/step-000106316/HC.png), [DC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/jobs/full_baseline_r0/run/renders/step-000106316/DC.png) и [KA](/fsx/tmp/marshawn_001660_hd/tuning_20261106/jobs/full_baseline_r0/run/renders/step-000106316/KA.png) сохранены. Root просмотрел HC: выраженный фон/туман и слабая реконструкция актёров; не финальный clean asset. [Первичный отчёт](marshawn_1660_leader_tuning.md) сохраняет исходный протокол и его ограничения, включая историческую FAS cache-index особенность.

### Очистка посредством3D-семантики и occupancy

Это отдельные inference-варианты исходного замороженного поля. Они выглядят чище, но не заменяют raw-LPIPS победителя. Transmittance пересчитывается из исходной sigma, умноженной на выбранный3D-selector; это не подстановка GT или отсечение2D-рендера по eval-маске.

| Фиксированный вариант | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Старый p01 head, hard.1, исходный обход |30.887904|.828332|.02216182|
| Тот же head, R0 |30.887912|.828332|.02216197|
| Тот же head, **R1** |30.895068|.828455|.02207324|
| Matched coarse128 refinement, hard.1 |30.898952|.828318|.02201018|
| Matched frozen128+residual256, hard.1 |30.893036|.828252|.02214944|
| Независимый новый256 head, hard.1 |30.603789|.826405|.02410645|

R1 уменьшил пропуски выбранной плотности на проверенных dense-ray выборках: одежда1.244%→.0112%, обувь.783%→.00407%. Это покрытие **того же обученного поля**, не физическая истина о сцене. R1 также открывает ошибочную плотность. Полный native FHD render имеет cap-hit пиксели, хотя старая выборка115лучей не имела ни одного; sparse-аудит нельзя превращать в сертификат всего грида.

Root просмотрел все384среза и три actor-вида/девять detail-областей соответствующих завершённых проверок. Независимый256 head ухудшил рендеры и потерял один sparse shoe witness при threshold.5. Matched refinement сохранил все109наблюдений, но увеличил диагностическую внешнюю opacity и не очистил контакт с полом.109наблюдений —41уникальная3D-точка, не полная геометрия ступней/волос.

Наиболее удачный из новых очищенных preview: [coarse128 hard.1 HC](/fsx/tmp/marshawn_001660_hd/tuning_20261106/frozen_LC_semantic_matched_refinement_physical_plan_v1/coarse128/renders/HC/hard_theta01.png). Его head SHA `3859f5882cb032cb5d0ae8fd164b3469ad71f35851834e00e7b34fd8f24cbc84` относится только к исходному c83 полю. На новые обученные поля этот head/occupancy certificate автоматически не переносится.

Для below-floor опытов фактическое число active cells, пересекающих z[-.07,-.05), изменилось115→120/79/10 (control/λ1e-4/λ1e-3). В сильной ветке семь клеток отдельного нижнего компонента стали inactive. Это результат обхода, **не доказательство исчезновения sigma**: отдельный actual-density probe подготовлен, но по команде пользователя не запускался. Root также просмотрел терминальные3+9рендеры: видимого устранения облака пола нет.

Подробности: [R0/R1](marshawn_semantic_grid_replay.md), [matched refinement](marshawn_matched_semantic_refinement.md), [независимый256](marshawn_LC_semantic_resolution.md), [ниже пола](marshawn_below_floor_density_prior.md).

### Что отклонено или осталось только подготовкой

| Направление | Итог |
|---|---|
| Калибровка по печатным мишеням/общая радиальная геометрическая поправка | Held-out ошибки ухудшились; новые поправки не применены. [Отчёт](marshawn_printed_target_calibration.md). |
| RGB confidence W32 и дополнительные политики весов | Новый лидер не получен; не заменяют W64. [W32](marshawn_sourceW32_continuation.md), [другие веса](marshawn_rgb_confidence_continuation.md). |
| Более сильная distortion-регуляризация | Часть геометрических диагностик улучшается, детали/LPIPS могут ухудшаться; новый лидер не получен. [Warm.3](marshawn_distortion03_warm.md), [ранние вмешательства](marshawn_semantic_late_regularization.md). |
| Положительная semantic responsibility только первых95%native веса | CPU-гипотеза не убирает давление на ложный нижний компонент; не обучалась. [Отчёт](marshawn_positive_tail95_responsibility.md). |
| Исправление28пикселей сумки в метрике | Только отдельная диагностика; изменение слишком мало, чтобы объяснить regression очистки. Исторический V1 не меняли. [Отчёт](marshawn_source_bag_annotation_diagnostic.md). |
| Дополнительные15TRAIN class labels пола/обуви | CPU-предложение и native-проверки готовы; fitter/новый head не обучались, все GPU flags false. [Отчёт](marshawn_extra15_floor_class_evidence.md). |
| Actual-density probe четырёх terminal полей | STAGED_NOT_RUN; новых sigma-результатов нет. |

### Отдельный4K-артефакт по запросу пользователя

**Реально4K-обученная модель** — checkpoint SHA `cb6aa1bc3e54011df9995f744f9b4d4a1beb29ebbb261147baa00ecd28402136`, а не HD-лидер340e. Для неё отдельно рассчитан HC3840×2024 по новым лучам камеры, без апскейла PNG: [RGB4K](/fsx/tmp/marshawn_001660_hd/tuning_20261106/user_4K_trained_native_render_v1/render/HC_4K.png), [alpha16](/fsx/tmp/marshawn_001660_hd/tuning_20261106/user_4K_trained_native_render_v1/render/HC_4K_alpha16.png), [отчёт и источник](marshawn_4k_trained_native_export.md). Это сохранённый экспорт, не новый training experiment и не устранение оставшихся артефактов.

## Выводы и статус фиксации

1. Зафиксировать quarter-step/.1 +W64/body80 +исходные frequency maps как лучший **завершённый** рецепт actor-LPIPS. Результат зависит от комбинации и длительного обучения: раннее ухудшение не предсказывало окончательного исхода.
2. Сохранять вместе RGB, confidence, карты, координаты/box, frozen source и checkpoint hashes. Одного YAML недостаточно; фотометрия, runtime distortion и FR являются частью рецепта.
3. Не подменять целевую детализацию чистотой силуэта или PSNR. Для финального clean asset нужны обе проверки, и текущий набор их полностью не проходит.
4. Не делать carving из видимого background-класса за непрозрачным объектом. Маска может пропустить настоящую обувь/волосы; физическая sigma, semantic probability, optical alpha и binary traversal — разные величины.
5. **Все запуски остановлены пользователем.** Cold hash23 legacy, hash23 corrected и hash24 legacy получили SIGTERM; SIGKILL не потребовался. Последние сохранённые шаги:52000/52180/17652. Траектории имеют статус **USER_CANCELLED**, не plateau, без повышения до победителя. Retainers и gallery watchers отключены до остановки workers. [Точные checkpoint hashes, источники,20k-gate и отмена](marshawn_cold_hash_capacity_corrected_allocator.md).

Детальный текст статьи и первоначальный рецепт остаются в [Paper LookCloser](../Paper%20LookCloser.md) и [static leader recipe](static_leader_reproduction_recipe.md); выводы выше основаны на измеренных результатах именно этого кадра.
