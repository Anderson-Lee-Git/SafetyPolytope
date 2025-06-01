# Modified from Github: Baichuan-7B/blob/main/evaluation/evaluate_mmlu.py

import logging
import os
from functools import partial
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
import torch
from llm_jailbreaking_defense import (
    BacktranslationConfig,
    DefendedTargetLM,
    ICLDefenseConfig,
    ParaphraseDefenseConfig,
    ResponseCheckConfig,
    SelfReminderConfig,
    SemanticSmoothConfig,
    SmoothLLMConfig,
    TargetLM,
    load_defense,
)
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety_polytope.evaluation.mmlu_categories import (
    categories,
    subcategories,
)
from safety_polytope.polytope.safe_rep_model import SafeRepModel

choices = ["A", "B", "C", "D"]
log = logging.getLogger("polytope")


def format_subject(subject):
    s = ""
    for entry in subject.split("_"):
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    # Make the line <79 chars
    prompt = "Multiple choice questions (with answers) about "
    prompt += f"{format_subject(subject)}:\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(cfg, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []

    for i in range(test_df.shape[0]):
        k = cfg.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while len(tokenizer(prompt).input_ids) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # Handle different model types
        if isinstance(model, DefendedTargetLM):
            # For defended models, evaluate likelihood for each choice
            log_probs = []
            for choice in choices:
                log_prob = model.evaluate_log_likelihood(prompt, choice)
                log_probs.append(log_prob)

            # Convert log probabilities to probabilities
            log_probs = torch.tensor(log_probs).cuda()
            probs = torch.exp(log_probs)
            # Normalize probabilities
            probs = probs / probs.sum()
        else:
            # For regular HuggingFace models, use direct logits
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1].flatten()

            choice_logits = torch.tensor(
                [
                    logits[
                        tokenizer(choice, add_special_tokens=False).input_ids[
                            0
                        ]
                    ]
                    for choice in choices
                ]
            ).cuda()

            probs = torch.nn.functional.softmax(choice_logits, dim=0)

        probs = probs.detach().cpu().numpy()
        pred = choices[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)

    log.info("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def load_safe_rep_model(cfg):
    model = SafeRepModel.from_pretrained(
        cfg.model_path,
        polytope_weight_path=cfg.polytope_weight_path,
        use_backup_response=cfg.use_backup_response,
        projection=cfg.projection,
        steer_layer=cfg.steer_layer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        lambda_weight=cfg.lambda_weight,
        steer_first_n_tokens=cfg.steer_first_n_tokens,
        safe_violation_weight=cfg.safe_violation_weight,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
    )
    return model, tokenizer


def setup_defense(model, defense_method):
    """Setup defense wrapper for the model"""
    if defense_method == "backtranslation":
        config = BacktranslationConfig()
    elif defense_method == "semantic_smoothing":
        config = SemanticSmoothConfig()
    elif defense_method == "response_check":
        config = ResponseCheckConfig()
    elif defense_method == "smoothllm":
        config = SmoothLLMConfig()
    elif defense_method == "paraphrase_prompt":
        config = ParaphraseDefenseConfig()
    elif defense_method == "icl":
        config = ICLDefenseConfig()
    elif defense_method == "self_reminder":
        config = SelfReminderConfig()
    else:
        raise ValueError(f"Unknown defense method: {defense_method}")

    defense = load_defense(config)
    defended_model = DefendedTargetLM(model, defense)
    return defended_model


def detect_model_name_from_path(model_path: str) -> str:
    """
    Detect the model name from the model path.
    Currently supports detecting Llama-2 models.
    """
    model_path = model_path.lower()

    # Llama 2 detection
    if "llama-2" in model_path or "llama2" in model_path:
        if "7b" in model_path:
            return "llama-2-7b"

    if "ministral-8b" in model_path:
        return "ministral-8b"

    if "qwen2-1.5b" in model_path:
        return "qwen2-1.5b"

    return os.path.basename(model_path.rstrip("/"))


def load_model_for_mmlu(cfg):
    """Load model with optional defense wrapper for MMLU evaluation"""
    if cfg.use_safe_rep_model:
        model, tokenizer = load_safe_rep_model(cfg)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path,
        )

        if cfg.use_defense:
            model_name = detect_model_name_from_path(cfg.model_path)
            print(f"Detected model name: {model_name}")
            print(f"Setting up model with {cfg.defense_method} defense")

            target_model = TargetLM(
                model_name=model_name,  # Pass the detected model name
                max_n_tokens=2048,
            )
            model = setup_defense(target_model, cfg.defense_method)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

    return model, tokenizer


def oom_safe_eval(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("OOM error occurred. Skipping this evaluation.")
        return None, None, None


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="eval_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    model, tokenizer = load_model_for_mmlu(cfg)

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(cfg.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    save_dir = os.getcwd()
    results_dir = os.path.join(
        save_dir, f"results_{os.path.basename(cfg.model_path)}"
    )
    os.makedirs(results_dir, exist_ok=True)

    all_cors = []
    subcat_cors: Dict[str, List[np.ndarray]] = {
        subcat: []
        for subcat_lists in subcategories.values()
        for subcat in subcat_lists
    }
    cat_cors: Dict[str, List[np.ndarray]] = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(cfg.data_dir, "dev", f"{subject}_dev.csv"),
            header=None,
        )[: cfg.ntrain]
        test_df = pd.read_csv(
            os.path.join(cfg.data_dir, "test", f"{subject}_test.csv"),
            header=None,
        )

        oom_safe_eval_func = partial(oom_safe_eval, eval)
        cors, acc, probs = oom_safe_eval_func(
            cfg, subject, model, tokenizer, dev_df, test_df
        )

        if cors is not None:
            subcats = subcategories[subject]
            for subcat in subcats:
                subcat_cors[subcat].append(cors)
                for key in categories.keys():
                    if subcat in categories[key]:
                        cat_cors[key].append(cors)
            all_cors.append(cors)

            test_df[f"{os.path.basename(cfg.model_path)}_correct"] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df[
                    f"{os.path.basename(cfg.model_path)}_choice{choice}_probs"
                ] = probs[:, j]
            test_df.to_csv(
                os.path.join(results_dir, f"{subject}.csv"), index=None
            )
        else:
            log.warning(f"Skipped subject {subject} due to OOM error")

    # Calculate accuracies only for non-empty lists
    for subcat, cors_list in subcat_cors.items():
        if cors_list:
            subcat_acc = np.mean(np.concatenate(cors_list))
            log.info("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
        else:
            log.warning(f"No data for subcat {subcat}")

    for cat, cors_list in cat_cors.items():
        if cors_list:
            cat_acc = np.mean(np.concatenate(cors_list))
            log.info("Average accuracy {:.3f} - {}".format(cat_acc, cat))
        else:
            log.warning(f"No data for category {cat}")

    if all_cors:
        weighted_acc = np.mean(np.concatenate(all_cors))
        log.info("Average accuracy: {:.3f}".format(weighted_acc))
    else:
        log.warning("No successful evaluations to calculate overall accuracy")


if __name__ == "__main__":
    main()
