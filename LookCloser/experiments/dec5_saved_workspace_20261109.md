# DEC5 research workspace saved on2026-11-09

## What was preserved

This branch archives previously uncommitted DEC5 reports, measured JSON/CSV evidence, image/video reviews, experiment scripts and tests. Existing field/model extensions and the optional renderer hair-override path are preserved with their actual dependencies. The full architecture snapshot is retained as working-state documentation and includes historical references to other scenes. No training or evaluation was launched to make this archive.

## Results and validation

The topic reports in this branch remain the scientific source: they distinguish accepted outputs, negative controls and unfinished pilots. Saving a script is not a new acceptance result. Shared reviewed numerical/traversal fixes live on main; ROI density, view-direction and DEC5 sampling hooks remain experimental on this scene branch. Python syntax and selected CPU regression checks are recorded in the commit workflow. GPU smoke tests are disabled.

## Inventory and limits

`workspace_save_inventory_20261109.json` enumerates the original907 pending tracked/untracked file states and which existing branch already preserved them. The remaining515 states are now saved in this DEC5 branch (514) and the Luster diagnostics branch (1). Some CSVs retain original CRLF and snapshots preserve original bytes. Scene datasets, neural weights and large external runtime artifacts remain referenced by their reports rather than bundled into Git. Historical budget/status snapshots are archival, not live controllers.

## Save-time checks

181 newly saved Python files parsed successfully. Selected CPU regression suite:25passed/1skipped. The first run exposed a stale synthetic fixture without camera dimensions; a follow-up supplies its16×16 camera raster without changing experiment/runtime behavior. The original fixture remains in archive parent7cc24fec. An exact path/blob audit covers all907 pending states across shared/scene refs and their reachable archive history, including all four core-model source files; see `workspace_save_verification_20261109.json`. These checks do not constitute fresh GPU training or scientific re-evaluation.
