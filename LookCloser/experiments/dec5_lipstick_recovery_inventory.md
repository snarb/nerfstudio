# DEC5 5A-3: поиск данных, сессий и checkpoint

## What was tested

2026-10-27, read-only поиск на `/fsx`, `clever-shadow` и во всех доступных
Git refs; `git fetch origin --prune` выполнен успешно. Обучение не запускалось.
Цель: сцена с девушкой, наносящей помаду, и прежние эксперименты LookCloser.
Идентификация **DEC5 5A-3**, кадры `000899` / `000973`, подтверждена визуальным
просмотром сохранённого teacher/prediction кропа из candidate-ветки.

## Results

### Датасет на FSX

| Путь | Что подтверждено |
|---|---|
| `/fsx/tmp/dec5_5a3_nerfstudio_exr_1920x1080` | 106 числовых каталогов, `000899..001257`; это диапазон, не непрерывная последовательность |
| `…/000973/transforms.json` | 65 ракурсов, относительные пути к EXR, train/eval filenames |
| `…/000973/images/frame_eval_00002.exr` | Файл 8,481,643 bytes; заголовок успешно прочитан, OpenEXR magic `762f3101` |
| `/fsx/oregon/projects/Dec5Shoots/workspace/DEC5_5A_3/pq16_extract/fullres` | 360 временных каталогов `000899..001258`; в `000973` — 65 TIFF |
| `/fsx/oregon/projects/Dec5Shoots/workspace/DEC5_5A_3` | Также есть `calib`, `tri`, extraction log и contact sheets |

Это проверка структуры и выборочного чтения, не полный checksum/decode аудит.
Попытка получить TIFF-thumbnail системным ImageMagick завершилась exit 134;
визуальная идентификация опирается на кроп из Git, а не на этот TIFF.

### Clever-shadow и сетевой бекап

SSH работает через `ubuntu@clever-shadow`. Вход `brans@clever-shadow` отклонён
Tailscale policy: `tailnet policy does not permit you to SSH as user "brans"`.
Пароль не потребовался и не записывался в файлы.

На текущем сервере `/home/brans` содержит только пустой `.bash_history`.
`/home/brans/.codex`, `/home/brans/repos`, `/home/brans/lookcloser_artifacts`
и `/mnt/data` отсутствуют. Поиск checkpoint по домашним каталогам и локальным
`/mnt/footage`, `/mnt/footage2` не нашёл `best.pt` / `*.ckpt` LookCloser.

Обнаружен **незавершённый**, а не полный бекап:

```text
clever-shadow:/mnt/tank/clever-shadow_backup/home_brans_2026-10-05
```

Это SMB share `//heavy-panther/tank`, отдельный от `/fsx`. Сохранилась структура
`.codex/sessions/2026/{07,08,09,10}`, worktrees `545e`, `952d`,
`look_closer_on_lipstick_girl_candidate`, а также `lookcloser_artifacts`.
Рекурсивная проверка фактических файлов дала:

| Каталог внутри бекапа | Файлов |
|---|---:|
| `.codex/sessions` | 0 |
| `.codex/worktrees/look_closer_on_lipstick_girl_candidate/nerfstudio/LookCloser/experiments` | 0 |
| `lookcloser_artifacts/dec5_core_retraining` | 0 |

В частности, `native_pose_uncertain_hull` содержит только каталоги eval/source,
без `best.pt`. Лог рядом, `home_brans_2026-10-05.rsync.log`, заканчивается:

```text
xfr#12086, to-chk=514873/528759
rsync error: unexplained error (code 255)
write error: Broken pipe (32)
rsync error: received SIGUSR1 (code 19)
```

### Сессии/бекапы на FSX и ограничения доступа

Найден `/fsx/oregon/luster_solve_inputs_2026_10_23/codex_backups`, но перечислить
содержимое невозможно: `Permission denied`. В ACL есть `user:pavlo:rwx`, однако
`mask::---` обнуляет effective permissions. Поэтому наличие там старых сессий
clever-shadow или весов **не установлено**. Права не менялись.

`/fsx/oregon/session_records` содержит записи более поздних GTA/PrisonVan работ;
в доступных текстах не найдено экспериментов по lipstick girl. Некоторые
подкаталоги также закрыты правами. `/fsx/chai/code_backup_2026-07-22` содержит
bundle/knowledge archives; отношение к этой сцене не подтверждено.

В доступных целевых FSX-каталогах LookCloser/DEC5 `best.pt` / `*.ckpt` не найдены.
Это не доказывает отсутствие весов на всём сетевом хранилище: закрытый
`codex_backups` не проверен, полного рекурсивного аудита всего FSX не было.
Часть других файлов в luster offload возвращает `No data available (os error 61)`.

### Что сохранилось в Git

В `main` и `origin/main` есть по **150** Markdown-отчётов `experiments/dec5_*`.
Например:

- [Раннее исследование blur](dec5_000899_lookcloser_blur_causal.md).
- [Сравнение LookCloser и baseline по камерам](dec5_000899_heldout_camera_ladder.md).
- [Подготовка mesh-distillation](dec5_mesh_distillation_preparation.md).
- [План обучения distillation](dec5_mesh_distillation_training_task.md).
- [Диагностика геометрии помады](dec5_lipstick_fin_measured_free_space.md).

Специальная ветка:

```text
origin/look_closer_on_lipstick_girl_candidate
8389770b21c54a30a36712f50da1bc8250168458 — 2026-09-29
Preserve lipstick LookCloser candidates and ablation evidence
```

В ней сохранены:

- `LookCloser/experiments/dec5_lookcloser_candidates.md` — сводка экспериментов,
  качества, ограничений и кандидатов на улучшение;
- `LookCloser/experiments/assets/lipstick_lookcloser_candidates/` — request JSON,
  PSNR/SSIM/LPIPS, provenance, сравнительные кропы и таблицы ablation;
- `LookCloser/recipes/dec5_lookcloser_candidates.json` — рецепты и исторические пути.

Чтение без переключения рабочей ветки:

```bash
git show origin/look_closer_on_lipstick_girl_candidate:LookCloser/experiments/dec5_lookcloser_candidates.md
```

Сводка не включена в `main`; код/отчёт/компактные свидетельства сохранены,
большие веса не коммитились. Исторический manifest ссылается на commit `d36e1068`,
но этот object отсутствует в текущем клоне даже после fetch. Поэтому упомянутые
там полные `dec5_core_retraining.md` / `dec5_lookcloser_recovery.md` не следует
считать восстановленными; доступна их сводка и компактные свидетельства.

Исторические checkpoint-пути из отчёта, **не подтверждённые сейчас**:

```text
/mnt/data/dec5_lookcloser_recovery_v3/actor_tube_rgb/best.pt
/mnt/data/dec5_lookcloser_recovery_v2/real_empty_tight_refine/best.pt
/home/brans/lookcloser_artifacts/dec5_core_retraining/native_pose_uncertain_hull/best.pt
/home/brans/lookcloser_artifacts/dec5_core_retraining/observed_lc_extended/best.pt
```

В recipes availability была `true` на **2026-09-29**; это историческая запись,
не подтверждение наличия файлов 2026-10-27.

## Insights

Датасет найден; Git-отчёты, код и компактные результаты сохранены. Старые сессии
и checkpoint в проверенном октябрьском бекапе не сохранились. Окончательный
вывод о потере весов делать рано: нужно проверить закрытый FSX `codex_backups`
с доступом владельца и возможные отдельные бекапы старого `/mnt/data`.
