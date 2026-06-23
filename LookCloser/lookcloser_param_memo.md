# LookCloser: краткая памятка по frequency-пайплайну

## 1. Овервью компонентов

LookCloser добавляет над Instant-NGP hash-grid несколько frequency-aware механизмов. Важно: они независимы и должны тестироваться по одному.

### Frequency Grid

3D voxel grid, где каждая ячейка хранит frequency level сцены в этой области.

Это не обучаемый параметр optimizer-а. Он хранится как buffer и обновляется процедурно:

```text
2D frequency map + rendered depth -> 3D point -> voxel max update
```

Сам по себе Frequency Grid не меняет rendering. Он влияет только если включены потребители:

- Feature Re-weighting;
- Adaptive Ray Marching / Interval Adjustment;
- runtime grid update может менять будущие значения grid, но без потребителей это диагностическое состояние.

### Feature Re-weighting

Использует Frequency Grid при field query.

Обычный hash encoding возвращает признаки всех уровней. Re-weighting смотрит frequency level в 3D-точке и приглушает high-frequency hash levels там, где сцена считается low-frequency.

Это не "разная capacity в разных местах" буквально. Hash-grid параметры глобально те же, но градиент и feature contribution по уровням локально масштабируются.

Если `enable_feature_reweighting=false`, то `freq_grid.query(...)` для features не вызывается, и hash-grid работает как обычный encoding.

### FAS

Frequency-Averaged Sampling влияет на выбор train pixels/rays.

FAS обычно использует 2D frequency maps, а не обязательно 3D Frequency Grid. Поэтому FAS может влиять на обучение даже если:

```text
enable_adaptive_ray_marching=false
enable_feature_reweighting=false
```

Если FAS включен, разные frequency buckets получают разную вероятность попасть в batch.

### Adaptive Ray Marching / Interval Adjustment

Adaptive RM меняет sampling interval вдоль луча по Frequency Grid.

Interval Adjustment является частью Adaptive RM. Если:

```text
enable_adaptive_ray_marching=false
```

то Interval Adjustment тоже выключен.

Adaptive path:

```text
nerfacc coarse occupancy traversal -> query Frequency Grid -> subdivide intervals by frequency level
```

Low frequency -> крупнее шаг. High frequency -> мельче шаг.

### Rendering / fixed uniform path

Если Adaptive RM выключен, LookCloser использует fixed renderer:

```text
fixed_num_samples_per_ray
```

Это главный параметр плотности семплинга для LookCloser fixed/uniform rendering.

Если речь про nerfstudio/nerfacc-style default rendering, а не LookCloser fixed path, плотность обычно задается:

```text
render_step_size
```

или через wrapper:

```text
render_step_size_mult
```

В таком случае меньше `render_step_size` -> чаще samples вдоль луча -> дороже, но меньше шанс пропустить тонкую геометрию.

## 2. Гиперпараметры

### Frequency Grid

`enable_frequency_grid`

Включает создание/query/update 3D frequency grid. Без downstream-потребителей сам по себе на render/loss не влияет.

`grid_resolution`

Разрешение 3D voxel grid. Больше resolution -> более локальная частотная карта, но больше шум/память/разреженность.

`grid_update_interval`

Как часто pipeline обновляет Frequency Grid во время training.

`grid_update_batch_size`

Сколько rays/pixels использовать для одного runtime update.

`fallback_frequency_level`

Level, который возвращается, если Frequency Grid выключен.

`min_res`, `max_res`, `max_res_base`, `num_frequency_levels`

Определяют шкалу frequency levels:

```text
level_to_freq(l) = min_res * per_level_scale^l
per_level_scale = (max_res / min_res) ^ (1 / (num_levels - 1))
```

Для текущих HD frequency maps важно держать:

```text
min_res = 16
max_res = 8192
num_frequency_levels = 16
```

Если поменять `max_res`, `max_res_base` или `num_frequency_levels`, старые frequency maps становятся несовместимыми и их надо регенерировать.

### Feature Re-weighting

`enable_feature_reweighting`

Включает локальное масштабирование hash-grid levels по Frequency Grid.

Потенциальный риск: thin geometry может потерять нужные high-frequency gradients, если локальный level недооценен или patch signal разорван.

### FAS

`enable_fas`

Включает frequency-aware pixel sampling.

`fas_strength`

Сила смешивания FAS с uniform sampling. `1.0` - полный FAS, `0.0` - uniform.

`fas_warmup_steps`

Сколько training steps ждать перед включением FAS.

`fas_ramp_steps`

За сколько steps плавно довести FAS до заданной силы.

`sampling_ramp_start`, `sampling_ramp_end`

Диапазон относительных sampling weights по frequency levels. Например, `[1, 3]` означает, что high-frequency bucket может семплироваться примерно в 3 раза чаще low-frequency.

`fas_level_count_alpha`

Count-aware correction. `1.0` ближе к flat/count-proportional behavior, полезно для диагностики, чтобы большие noisy high-frequency области не забирали слишком много batch.

`fas_max_sampling_level`

Ограничивает максимальный frequency level, который FAS использует для sampling.

`fas_patch_group_size`

Диагностический параметр группировки patch sampling.

`fas_decay_start_steps`, `fas_decay_steps`

Диагностика: можно включить FAS рано, а потом плавно вернуть sampling к uniform.

### Adaptive Ray Marching / Interval Adjustment

`enable_adaptive_ray_marching`

Включает adaptive renderer. Если выключен, используется fixed renderer.

`adaptive_warmup_steps`

Сначала fixed sampling, потом adaptive. Важен, потому что early density и Frequency Grid могут быть шумными.

`adaptive_min_step_size`

Минимальный allowed step. Защищает от слишком большого числа samples.

`adaptive_max_step_size`

Максимальный allowed step. Защищает от слишком редкого sampling в low/unknown frequency зонах.

`adaptive_coarse_step_size`

Шаг первого nerfacc occupancy traversal. Если слишком крупный, можно пропустить тонкую стойку еще до frequency subdivision. Меньше значение -> плотнее и дороже.

`adaptive_min_frequency_level`, `adaptive_max_frequency_level`

Clamp frequency level только для interval sizing. Не меняет значения Frequency Grid.

Полезно для диагностики:

```text
adaptive_min_frequency_level = adaptive_max_frequency_level = K
```

Так adaptive renderer становится почти constant-frequency sampler.

`max_steps_per_ray`

Ограничение числа samples на ray. Если saturation растет, adaptive settings слишком агрессивные.

### Fixed / uniform rendering

`fixed_num_samples_per_ray`

Главный параметр LookCloser fixed renderer. Больше samples -> плотнее uniform sampling.

`render_step_size`

Для nerfacc/default-style traversal и occupancy grid update. Меньше -> чаще samples.

`render_step_size_mult`

Wrapper-level multiplier для расчета `render_step_size`.

## 3. Что может вызывать артефакт "часть трубы/стойки пропала"

Это отличается от общей размытости. Если исчезает конкретный фрагмент thin geometry, вероятные причины такие.

### Adaptive RM / Interval Adjustment

Самый вероятный источник, если artifact похож на пропущенную геометрию.

Возможные механизмы:

- `adaptive_coarse_step_size` слишком крупный, nerfacc traversal пропускает тонкую область;
- `adaptive_max_step_size` слишком крупный в low/unknown frequency voxel;
- Frequency Grid недооценил level вокруг стойки;
- adaptive включился слишком рано, когда geometry/density еще нестабильна;
- `max_steps_per_ray` режет samples, если есть saturation.

Диагностика:

```text
ARM off -> fixed_num_samples_per_ray 512/768
ARM on -> constant frequency clamp K
потом постепенно разжимать adaptive_max_frequency_level
```

### FAS

Может ломать не renderer напрямую, а optimization trajectory.

Возможные механизмы:

- batch переобучается на noisy high-frequency texture, а thin pole получает недостаточно стабильных rays;
- high-frequency labels слишком широкие: кирпич/фон забирает samples;
- стойка представлена patchy/discontinuous frequency labels;
- eval-loss checkpoint может быть глобально лучше, но визуально ломать thin object.

Диагностика:

```text
FAS off from scratch
FAS on only from stable checkpoint
fas_strength 0.2-0.35
fas_level_count_alpha=1.0
strict crop gate for stand
```

### Feature Re-weighting

Может приглушить нужные high-frequency hash levels на тонкой детали.

Особенно опасно, если Frequency Grid local level на стойке ниже, чем должен быть.

Диагностика:

```text
Reweight off
Reweight on with same renderer/sampler/checkpoint
```

### Frequency maps / preprocessing

Paper overfit frequency map может быть нестабильной: SSIM threshold `0.96` и `0.97` дают резко разные heatmaps.

Для thin metal geometry overfit map может быть хуже OpenCV edge/Laplacian map, потому что она меряет patch reconstruction threshold, а не object continuity.

Диагностика:

```text
A: constant frequency map
B: paper overfit map
C: OpenCV edge/Laplacian map
D: max(paper, edge)
```

Остальные механизмы держать одинаковыми.

### Runtime Frequency Grid updates

Runtime update не исправляет плохую 2D frequency map. Он в основном переносит 2D signal в 3D через текущий depth.

Если early depth неправильный, update может положить high/low frequency в неправильные voxels.

Диагностика:

```text
freeze grid / no runtime updates
runtime updates on
same fixed renderer, FAS off, Reweight off
```

## 4. Минимальная схема изоляции виновника

Идти только по одному включенному механизму за раз.

```text
0. fixed uniform, FG off, Reweight off, FAS off, ARM off
1. fixed uniform, constant freq maps, FG on, Reweight off, FAS off, ARM off
2. same + runtime grid updates
3. constant freq maps + ARM on with fixed level clamp
4. real maps + ARM on, Reweight off, FAS off
5. real maps + Reweight on, ARM same, FAS off
6. real maps + FAS on only from stable checkpoint
```

Для каждого шага фиксировать:

```text
map type | FG update | ARM | adaptive clamp | Reweight | FAS | render samples | stand crop ok?
```

Главный visual gate для проблемы стойки:

```text
left_stand_connector_eval0 --stride 1
```

Если стойка пропала на переходе N -> N+1, виноват почти наверняка единственный включенный механизм или связанный с ним параметр.
