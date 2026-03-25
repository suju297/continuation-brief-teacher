# Continuation Brief Teacher

This repo contains the Kaggle single-T4 notebook path for teacher generation.

Files:
- `teacher_notebook.ipynb`: the notebook to run on Kaggle
- `teacher_runtime.py`: the helper runtime imported by the notebook

Kaggle setup:
1. Use a single `NvidiaTeslaT4` runtime.
2. Attach the dataset `sujendragharat/qwen4b-teacher-2gpu-inputs`.
3. Open `teacher_notebook.ipynb`.
4. Run the smoke cells first.
5. Enable the full run only after smoke passes.

Notes:
- The notebook keeps the existing merged JSON and summary output schema.
- The notebook is sequential and avoids the older 2-GPU shard launcher path.
