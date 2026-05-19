## General Instructions

Your goal is to help implement **"LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene"**.

We use **nerfstudio** (parent dir ./../) as the base codebase and extend it with the LookCloser model.

A rough, unverified baseline implementation is already committed. Treat it critically. 

Use the local paper document as the primary reference:

`Paper LookCloser.md`

Always use this local file instead of downloading the paper from the internet.

The main source of truth is measured experimental behavior: metrics, rendered outputs, and visual inspection. The paper is the starting point, but some implementation details or hyperparameters may be incomplete, unclear, or suboptimal for our dataset.

Start from the paper’s design and hyperparameters, then iteratively:
- verify the implementation against the paper;
- fix bugs and inconsistencies;
- improve the code and adapt hyperparameters based on measured results.

At the initial stage, final validation stage, and when debugging unclear problems, render and visually inspect outputs. Save and compare problematic patches visually. Use small/low-resolution crops or thumbnails when possible to avoid wasting tokens.


## Architecture
The architecture documentation is stored in ./architecture.md. Update it after each major change, but keep cosine.

## Experiments
Save experiment results in the ./experiments directory. Use one file per topic. 

Experiment Markdown File Structure:

- What was tested: The hypothesis or configuration.

- Results: Tables and rendered crops/images with links.

- Insights: What was learned and next steps.

Keep cosine.