## Agent Instructions

Your goal is to help implement **"LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene"**.

We use **nerfstudio** (parent dir ./../) as the base codebase and extend it with the LookCloser model.

A rough, unverified baseline implementation is already committed. Treat it critically. Do not assume the current code is correct.

Use the local paper document as the primary reference:

`Paper LookCloser.md`

It contains the full paper text, figures, and links. Always use this local file instead of downloading the paper from the internet.

The main source of truth is measured experimental behavior: metrics, rendered outputs, and visual inspection. The paper is the starting point, but some implementation details or hyperparameters may be incomplete, unclear, or suboptimal for our dataset.

Start from the paper’s design and hyperparameters, then iteratively:
- verify the implementation against the paper;
- fix bugs and inconsistencies;
- improve the code and adapt hyperparameters based on measured results.

At the initial stage, final validation stage, and when debugging unclear problems, render and visually inspect outputs. Save and compare problematic patches visually. Use small/low-resolution crops or thumbnails when possible to avoid wasting tokens.


## Key files:

- `nerfstudio/scripts/lookcloser_preprocess.py` — 2D patch frequency preprocessing.
- `nerfstudio/model_components/lookcloser_grid.py` — 3D frequency grid.
- `nerfstudio/fields/lookcloser_field.py` — frequency-aware field.
- `nerfstudio/lookcloser_pixel_sampler.py` — frequency-aware sampler.
- `nerfstudio/models/lookcloser.py` — LookCloser model and adaptive ray marching.
- `nerfstudio/pipelines/lookcloser_pipeline.py` — training pipeline and grid updates.
- `nerfstudio/configs/method_configs.py` — `lookcloser` method config.

## Environment

Use the following Conda environment:
```bash
conda activate /home/ubuntu/anaconda3/envs/nerfstudio
```

## Data

Use following images for preprocessing and overfitting tuning and debugging: 

 /home/ubuntu/repos/look-closer/E004_D014_graded.png  - 6K image for overfitting 

/home/ubuntu/repos/look-closer/E004_D014_HD.jpg - the same image in HD for a fast visual inspection if needed.