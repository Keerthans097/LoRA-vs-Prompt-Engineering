import os, random, numpy as np
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset, DatasetDict

# label normalization utilities
def canonicalize_label(label: str) -> str:
    return label.strip().lower()

def closest_label_from_text(text: str, labels: List[str], default: str=None) -> str:
    t = text.strip().lower()
    # try exact token match
    for lab in labels:
        if lab in t.split():
            return lab
    # try substring
    for lab in labels:
        if lab in t:
            return lab
    return default or labels[-1]

def load_and_prepare_dataset(name: str, config: str=None, seed: int=42) -> Tuple[DatasetDict, Dict]:
    """
    Returns (dataset_splits, spec) where:
      - dataset_splits is DatasetDict with train/validation/test where available
      - spec contains:
          task_type: 'classification' or 'biomed_qa_ynm'
          label_list: list[str]
          fields: dict of field names used by this task
          templates: dict with 'zero_shot' and 'few_shot_example' format strings
    """
    name = name.lower()
    if name == "pubmed_qa":
        ds = load_dataset("pubmed_qa", config or "pqa_labeled")
        # Fields: question, context, final_decision (Yes/No/Maybe)
        # Normalize labels
        def map_lab(b):
            b["label_text"] = b["final_decision"].lower()
            return b
        ds = ds.map(map_lab)
        spec = {
            "task_type": "biomed_qa_ynm",
            "label_list": ["yes","no","maybe"],
            "fields": {"question":"question","context":"context","label":"label_text"},
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

    # elif name == "ag_news":
    #     ds = load_dataset("ag_news")
    #     ## Fields: text, label (int)
    #     labels = ["World","Sports","Business","Sci/Tech"]
    #     def map_ag(b):
    #         b["text"] = b["text"]
    #         b["label_text"] = labels[b["label"]].lower()
    #         return b
    #     ds = ds.map(map_ag)
    #     ## Train/Test only; create a small validation from train
    #     if "validation" not in ds:
    #         ds = ds.rename_column("train","train") if "train" in ds else ds
    #         tr = ds["train"].train_test_split(test_size=0.05, seed=seed)
    #         ds = DatasetDict({"train": tr["train"], "validation": tr["test"], "test": ds["test"]})
    #     spec = {
    #         "task_type":"classification",
    #         "label_list":[l.lower() for l in labels],
    #         "fields":{"text":"text","label":"label_text"},
    #         "templates":{
    #             "zero_shot": (
    #                 "Classify the following news into one of "
    #                 "[world, sports, business, sci/tech].\n\n"
    #                 "Text: {text}\n"
    #                 "Label:"
    #             ),
    #             "few_shot_example": (
    #                 "Text: {text}\n"
    #                 "Label: {label}\n"
    #             )
    #         }
    #     }
    #     return ds, spec

    # elif name == "scicite":
    #     ds = load_dataset("scicite")
    #     labels = ["Background","Method","Result"]
    #     def map_sc(b):
    #         b["text"] = b["string"]
    #         b["label_text"] = labels[b["label"]].lower()
    #         return b
    #     ds = ds.map(map_sc)
    #     spec = {
    #         "task_type":"classification",
    #         "label_list":[l.lower() for l in labels],
    #         "fields":{"text":"text","label":"label_text"},
    #         "templates":{
    #             "zero_shot": (
    #                 "Classify the citation intent into one of "
    #                 "[background, method, result].\n\n"
    #                 "Citation: {text}\n"
    #                 "Label:"
    #             ),
    #             "few_shot_example": (
    #                 "Citation: {text}\n"
    #                 "Label: {label}\n"
    #             )
    #         }
    #     }
    #     return ds, spec

    # elif name == "sst2":
    #     ds = load_dataset("glue","sst2")
    #     def map_sst(b):
    #         b["text"] = b["sentence"]
    #         b["label_text"] = "positive" if b["label"]==1 else "negative"
    #         return b
    #     ds = ds.map(map_sst)
    #     ## glue has train/validation/test (no labels on test); we use validation as test
    #     spec = {
    #         "task_type":"classification",
    #         "label_list":["negative","positive"],
    #         "fields":{"text":"text","label":"label_text"},
    #         "templates":{
    #             "zero_shot": (
    #                 "Is the sentiment of the following sentence negative or positive?\n\n"
    #                 "Sentence: {text}\n"
    #                 "Label:"
    #             ),
    #             "few_shot_example": (
    #                 "Sentence: {text}\n"
    #                 "Label: {label}\n"
    #             )
    #         }
    #     }
    #     ## remap to have test=validation for simplicity
    #     ds = DatasetDict({"train": ds["train"], "validation": ds["validation"], "test": ds["validation"]})
    #     return ds, spec

    else:
        raise ValueError(f"Unsupported dataset: {name}")

def build_few_shot_block(ds: Dataset, spec: Dict, k: int, seed: int=42) -> str:
    if k <= 0: return ""
    tmpl = spec["templates"]["few_shot_example"]
    fields = spec["fields"]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(k, len(ds)), replace=False)
    parts = []
    for i in idxs:
        ex = ds[int(i)]
        if spec["task_type"]=="biomed_qa_ynm":
            parts.append(tmpl.format(question=ex[fields["question"]], context=ex[fields["context"]], label=ex[fields["label"]]))
        else:
            parts.append(tmpl.format(text=ex[fields["text"]], label=ex[fields["label"]]))
    return "\n".join(parts)

def make_zero_shot_prompt(ex: Dict, spec: Dict) -> str:
    tmpl = spec["templates"]["zero_shot"]
    fields = spec["fields"]
    if spec["task_type"]=="biomed_qa_ynm":
        return tmpl.format(question=ex[fields["question"]], context=ex[fields["context"]])
    else:
        return tmpl.format(text=ex[fields["text"]])

def make_few_shot_prompt(kblock: str, ex: Dict, spec: Dict) -> str:
    return kblock + "\n" + make_zero_shot_prompt(ex, spec)

def instruction_from_spec(spec: Dict) -> str:
    # Used for supervised LM-style training with LoRA
    if spec["task_type"]=="biomed_qa_ynm":
        return ("You are a biomedical QA assistant.\n"
                "Answer the following question strictly with one token from [yes, no, maybe].\n\n"
                "Question: {input_question}\n"
                "Abstract: {input_context}\n\n"
                "Answer:")
    else:
        labels = ", ".join(spec["label_list"])
        return (f"Classify the following text into one of [{labels}].\n\n"
                "Text: {input_text}\n"
                "Label:")

def build_supervised_text(ex: Dict, spec: Dict) -> Tuple[str, str]:
    """Returns (input_text_for_lm, gold_label)"""
    instr = instruction_from_spec(spec)
    if spec["task_type"]=="biomed_qa_ynm":
        input_text = instr.format(input_question=ex[spec["fields"]["question"]],
                                  input_context=ex[spec["fields"]["context"]])
        gold = ex[spec["fields"]["label"]]
        return input_text + " " + gold, gold
    else:
        input_text = instr.format(input_text=ex[spec["fields"]["text"]])
        gold = ex[spec["fields"]["label"]]
        return input_text + " " + gold, gold
