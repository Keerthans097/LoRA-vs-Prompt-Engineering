from datasets import load_dataset, DatasetDict

def load_and_prepare_dataset(name: str="pubmed_qa", config: str="pqa_labeled", seed: int=42):
    if name != "pubmed_qa":
        raise ValueError("Only pubmed_qa is supported.")

    ds = load_dataset("pubmed_qa", config or "pqa_labeled")

    def map_lab(b):
        b["label_text"] = b["final_decision"].lower()
        return b
    ds = ds.map(map_lab)

    if "test" not in ds and "validation" in ds:
        ds = DatasetDict({
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["validation"]
        })

    spec = {
        "task_type": "biomed_qa_ynm",
        "label_list": ["yes", "no", "maybe"],
        "fields": {"question": "question", "context": "context", "label": "label_text"},
        "templates": {
            "zero_shot": (
                "You are a biomedical QA assistant.\n"
                "Answer the following question strictly with one token from [yes, no, maybe].\n\n"
                "Question: {question}\n"
                "Abstract: {context}\n\n"
                "Answer:"
            ),
            "few_shot_example": (
                "Question: {question}\n"
                "Abstract: {context}\n"
                "Answer: {label}\n"
            )
        }
    }
    return ds, spec

def build_few_shot_block(ds, spec, k: int, seed: int=42) -> str:
    import numpy as np
    if k <= 0:
        return ""
    tmpl = spec["templates"]["few_shot_example"]
    fields = spec["fields"]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(k, len(ds)), replace=False)
    parts = []
    for i in idxs:
        ex = ds[int(i)]
        parts.append(tmpl.format(question=ex[fields["question"]], context=ex[fields["context"]], label=ex[fields["label"]]))
    return "\n".join(parts)

def make_zero_shot_prompt(ex, spec) -> str:
    tmpl = spec["templates"]["zero_shot"]
    fields = spec["fields"]
    return tmpl.format(question=ex[fields["question"]], context=ex[fields["context"]])

def make_few_shot_prompt(kblock: str, ex, spec) -> str:
    return kblock + "\n" + make_zero_shot_prompt(ex, spec)

def instruction_from_spec(spec):
    return ("You are a biomedical QA assistant.\n"
            "Answer the following question strictly with one token from [yes, no, maybe].\n\n"
            "Question: {input_question}\n"
            "Abstract: {input_context}\n\n"
            "Answer:")

def build_supervised_text(ex, spec):
    instr = instruction_from_spec(spec)
    input_text = instr.format(input_question=ex[spec["fields"]["question"]], input_context=ex[spec["fields"]["context"]])
    gold = ex[spec["fields"]["label"]]
    return input_text + " " + gold, gold

def closest_label_from_text(text: str, labels, default: str=None) -> str:
    t = text.strip().lower()
    for lab in labels:
        if lab in t.split():
            return lab
    for lab in labels:
        if lab in t:
            return lab
    return default or labels[-1]
