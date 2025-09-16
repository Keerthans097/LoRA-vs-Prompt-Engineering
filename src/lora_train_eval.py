import os, json, time, argparse, random, numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from bert_score import score as bertscore
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.utils import load_and_prepare_dataset, build_supervised_text, instruction_from_spec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_train", type=int, default=128)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config)) if args.config.endswith(".json") else None
    if cfg is None:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]; dataset_config = cfg.get("dataset_config", None)
    base_model = cfg["base_model"]
    max_len = cfg["lora"]["max_length"]
    lr = float(cfg["lora"]["lr"]); epochs = int(cfg["lora"]["epochs"])
    per_bs = int(cfg["lora"]["per_device_bs"]); grad_accum = int(cfg["lora"]["grad_accum"])
    alpha = int(cfg["lora"]["alpha"]); dropout = float(cfg["lora"]["dropout"])

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    ds, spec = load_and_prepare_dataset(dataset_name, dataset_config, seed=args.seed)
    train = ds["train"].shuffle(seed=args.seed).select(range(min(args.n_train, len(ds["train"]))))
    test  = ds["test"]

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True, device_map="auto")
    mdl = prepare_model_for_kbit_training(mdl)

    lcfg = LoraConfig(
        r=args.r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM"
    )
    mdl = get_peft_model(mdl, lcfg)

    def tok_fn(batch):
        texts = []
        for i in range(len(batch[spec["fields"].get("label","label")])):
            ex = {k: batch[k][i] for k in batch.keys()}
            sup_text, _ = build_supervised_text(ex, spec)
            texts.append(sup_text)
        enc = tok(texts, truncation=True, padding="max_length", max_length=max_len)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tok_train = train.map(tok_fn, batched=True, remove_columns=train.column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    outdir = os.path.join(args.outdir, f"{dataset_name}_lora_r{args.r}_n{args.n_train}_seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    trainer = Trainer(
        model=mdl,
        args=TrainingArguments(
            output_dir=outdir, num_train_epochs=epochs,
            per_device_train_batch_size=per_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr, fp16=True,
            logging_steps=50, evaluation_strategy="no", save_strategy="no",
            report_to="none"
        ),
        train_dataset=tok_train, data_collator=collator
    )
    trainer.train()
    t1 = time.time()

    peak_gb = (torch.cuda.max_memory_allocated()/(1024**3)) if torch.cuda.is_available() else 0.0
    gpu_hours = ((t1 - t0)/3600.0) * (1 if torch.cuda.is_available() else 0)

    # Evaluation via generation
    gen = pipeline("text-generation", model=mdl, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    labels = spec["label_list"]
    preds, golds = [], []
    for ex in test:
        instr = instruction_from_spec(spec)
        if spec["task_type"]=="biomed_qa_ynm":
            inp = instr.format(input_question=ex[spec["fields"]["question"]], input_context=ex[spec["fields"]["context"]])
        else:
            inp = instr.format(input_text=ex[spec["fields"]["text"]])
        out = gen(inp, max_new_tokens=int(cfg.get("max_new_tokens",12)), do_sample=False)[0]["generated_text"]
        # map to canonical label
        lower = out.strip().lower()
        pred = None
        for lab in labels:
            if lab in lower:
                pred = lab; break
        pred = pred or labels[-1]
        preds.append(pred); golds.append(ex[spec["fields"]["label"]])

    acc = accuracy_score(golds, preds)
    f1  = f1_score(golds, preds, average="macro")
    P,R,F = bertscore(preds, golds, lang="en", rescale_with_baseline=True)
    bert_f1 = float(F.mean().item())
    cm = confusion_matrix(golds, preds, labels=labels).tolist()

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({
            "dataset": dataset_name,
            "config": dataset_config,
            "model": base_model,
            "n_train": args.n_train,
            "r": args.r,
            "acc": acc, "macro_f1": f1, "bertscore_f1": bert_f1,
            "gpu_hours": gpu_hours, "peak_vram_gb": peak_gb,
            "labels": labels, "confusion_matrix": cm
        }, f, indent=2)

    print(f"Saved metrics to {outdir}/metrics.json")

if __name__ == "__main__":
    main()
