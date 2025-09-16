import os, json, time, argparse, random, numpy as np
from sklearn.metrics import accuracy_score, f1_score
from bert_score import score as bertscore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
from src.utils import load_and_prepare_dataset, build_few_shot_block, make_zero_shot_prompt, make_few_shot_prompt, closest_label_from_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k_shot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_eval", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config)) if args.config.endswith(".json") else None
    if cfg is None:
        # YAML fallback
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]
    dataset_config = cfg.get("dataset_config", None)
    base_model = cfg["base_model"]
    few_shot_k = args.k_shot

    random.seed(args.seed); np.random.seed(args.seed)

    ds, spec = load_and_prepare_dataset(dataset_name, dataset_config, seed=args.seed)
    labels = spec["label_list"]
    test = ds["test"]
    train = ds["train"]

    # few-shot exemplars
    kblock = build_few_shot_block(train, spec, few_shot_k, seed=args.seed) if few_shot_k>0 else ""

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", load_in_8bit=True)
    gen = pipeline("text-generation", model=mdl, tokenizer=tok, device=0 if hasattr(mdl,"device") else -1)

    preds, golds, outs = [], [], []
    n = min(args.max_eval, len(test))
    t0 = time.time()
    for i in range(n):
        ex = test[i]
        gold = ex[spec["fields"]["label"]]
        prompt = make_zero_shot_prompt(ex, spec) if few_shot_k==0 else make_few_shot_prompt(kblock, ex, spec)
        out = gen(prompt, max_new_tokens=int(cfg.get("max_new_tokens",12)), do_sample=False)[0]["generated_text"]
        pred = closest_label_from_text(out, labels, default=labels[-1])
        preds.append(pred); golds.append(gold); outs.append(out)
    t1 = time.time()

    acc = accuracy_score(golds, preds)
    f1  = f1_score(golds, preds, average="macro")
    P,R,F = bertscore(preds, golds, lang="en", rescale_with_baseline=True)
    bert_f1 = float(F.mean().item())

    res = {
        "dataset": dataset_name,
        "config": dataset_config,
        "model": base_model,
        "k_shot": few_shot_k,
        "seed": args.seed,
        "n_eval": n,
        "acc": acc,
        "macro_f1": f1,
        "bertscore_f1": bert_f1,
        "wall_time_s": t1 - t0
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(res)

if __name__ == "__main__":
    main()
