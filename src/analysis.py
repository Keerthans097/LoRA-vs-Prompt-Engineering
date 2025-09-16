import os, argparse, json, glob, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/")
    ap.add_argument("--out_csv", default="results/summary.csv")
    args = ap.parse_args()

    rows = []
    for p in glob.glob(os.path.join(args.results_dir, "prompt_*.json")):
        d = json.load(open(p))
        d["track"] = "prompt"
        d["tag"] = os.path.basename(p)
        rows.append(d)
    for p in glob.glob(os.path.join(args.results_dir, "*_lora_r*/metrics.json")):
        d = json.load(open(p))
        d["track"] = "lora"
        d["tag"] = os.path.dirname(p)
        rows.append(d)
    if not rows:
        print("No results found in", args.results_dir); return
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)

    # Simple plots
    try:
        dfl = df[df["track"]=="lora"].sort_values("n_train")
        if len(dfl):
            plt.figure(figsize=(6,4))
            plt.plot(dfl["n_train"], dfl["macro_f1"], marker="o")
            plt.xlabel("Training examples (LoRA)")
            plt.ylabel("Macro-F1")
            plt.title("LoRA Macro-F1 vs N")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir,"plot_f1_vs_n.png"))
            print("Saved plot_f1_vs_n.png")
    except Exception as e:
        print("Plot error:", e)

if __name__ == "__main__":
    main()
