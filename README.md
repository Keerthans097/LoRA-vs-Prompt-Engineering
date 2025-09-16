# LoRA vs Prompt Engineering (Flexible Datasets)

This repo compares **Prompt Engineering (0-/few-shot)** vs **LoRA fine-tuning (via QLoRA)**
on multiple datasets (default: **PubMedQA**). It is **config-driven** and **cluster-friendly**.

## Features
- Datasets: PubMedQA (PQA-L), AG News, SciCite, SST-2 (GLUE) — easy to add more.
- Prompt baselines (zero-/few-shot) with multiple seeds.
- QLoRA fine-tuning with rank ablations (r=8,16).
- Metrics: Accuracy, Macro-F1, **BERTScore**.
- Cost metrics: GPU hours, peak VRAM, trainable params.
- Plots: F1 vs N, F1 per GPU-hour, prompt sensitivity.
- SLURM job scripts + Jupyter demo notebook.

## Quickstart

### 1) Environment
```bash
conda create -y -n lvp python=3.10
conda activate lvp
pip install -r requirements.txt
# (optional) huggingface-cli login   # if using gated models like LLaMA-2
```

### 2) Config (defaults to PubMedQA)
`configs/config.yaml` controls model/dataset/params.
Change `dataset:` to `ag_news`, `scicite`, or `sst2` to switch.

### 3) Prompting Baselines
```bash
python src/prompt_eval.py --config configs/config.yaml --k_shot 0 --seed 41 --out results/prompt_0shot_seed41.json
python src/prompt_eval.py --config configs/config.yaml --k_shot 4 --seed 41 --out results/prompt_4shot_seed41.json
```

### 4) QLoRA Fine-tuning + Eval
```bash
python src/lora_train_eval.py --config configs/config.yaml --n_train 128 --r 8 --seed 41 --outdir results/
python src/lora_train_eval.py --config configs/config.yaml --n_train 512 --r 8 --seed 41 --outdir results/
python src/lora_train_eval.py --config configs/config.yaml --n_train 1000 --r 16 --seed 41 --outdir results/
```

### 5) Aggregate and Plot
```bash
python src/analysis.py --results_dir results/ --out_csv results/summary.csv
```

### 6) Notebook Demo
Open `notebooks/01_experiments.ipynb` for an interactive run.

## Datasets Supported (initial)
- `pubmed_qa` (config: `pqa_labeled`) — labels: yes/no/maybe
- `ag_news` — labels: World/Sports/Business/Sci&Tech
- `scicite` — labels: Background/Method/Result
- `sst2` (GLUE) — labels: positive/negative

## Notes
- If VRAM is limited (e.g., T4 16GB), keep `--max_length` moderate (e.g., 384–512) and batch size small.
- If you lack LLaMA-2 access, set `base_model: "mistralai/Mistral-7B-Instruct-v0.2"` in config.
