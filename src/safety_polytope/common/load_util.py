import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safety_polytope.polytope.safe_rep_model import SafeRepModel


def load_representations(rep_save_dir):
    # Walk through the directory and load all representations
    directions = {}
    signs = {}
    for root, _, files in os.walk(rep_save_dir):
        for file in files:
            if file.endswith("directions.pth"):
                rep = torch.load(os.path.join(root, file))
                category = file[: file.index("_directions.pth")]
                directions[category] = rep
            elif file.endswith("signs.pth"):
                sign = torch.load(os.path.join(root, file))
                category = file[: file.index("_signs.pth")]
                signs[category] = sign
    return directions, signs


def load_model_and_tokenizer(model_path, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        **kwargs,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        # use_fast=False, padding_side="left", padding="max_length"
    )
    tokenizer.pad_token = (
        tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_guard_model_and_tokenizer(model_path, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    return model, tokenizer


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from sklearn.utils import check_random_state

    check_random_state(seed)


def get_model_name(model_path: str) -> str:
    """Extract standardized model name from model path."""
    if "Ministral-8B" in model_path:
        return "ministral-8b"
    elif "llama-2-7b" in model_path:
        return "llama2-7b"
    elif "Qwen2-1.5B-Instruct" in model_path:
        return "qwen2-1.5b-instruct"
    elif "Qwen2-1.5B" in model_path:
        return "qwen2-1.5b"
    elif "Qwen3-4B-Instruct-2507" in model_path:
        return "qwen3-4b-instruct"
    elif "Qwen3-4B-Base" in model_path:
        return "qwen3-4b"
    elif "gpt-oss-20b" in model_path:
        return "gpt-oss-20b"
    elif "alpaca-7b" in model_path:
        return "alpaca-7b"
    else:
        raise NotImplementedError(
            f"Please manually configure a model name for {model_path}."
        )
