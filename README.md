# Continuation Brief Teacher

This repo contains the Kaggle notebooks and helper runtimes for continuation-brief generation and judging.

Files:
- `teacher_notebook.ipynb`: the notebook to run on Kaggle
- `teacher_runtime.py`: the helper runtime imported by the notebook
- `teacher_judge_notebook.ipynb`: the dual-T4 notebook with a stronger judge on GPU 1
- `teacher_judge_kaggle.py`: the teacher-plus-judge pipeline imported by the stronger-judge notebook

Kaggle setup:
1. Use a single `NvidiaTeslaT4` runtime for `teacher_notebook.ipynb`, or a dual `NvidiaTeslaT4` runtime for `teacher_judge_notebook.ipynb`.
2. Attach the dataset `sujendragharat/qwen4b-teacher-2gpu-inputs`.
3. Open the notebook you want to run.
4. For the judge notebook, install `bitsandbytes` and `accelerate` in the setup cell.
5. Review the config cell before starting the full run.

Current default input override:
- `/kaggle/input/datasets/sujendragharat/qwen-4b-teacher-2gpu-inputs-new`

Notes:
- The notebook keeps the existing merged JSON and summary output schema.
- The notebooks are sequential and avoid the older 2-GPU shard launcher path.
- The stronger-judge path defaults to `Qwen/Qwen3-4B-Instruct-2507` on `cuda:0` and `Qwen/Qwen2.5-14B-Instruct` in `4bit` on `cuda:1`.
