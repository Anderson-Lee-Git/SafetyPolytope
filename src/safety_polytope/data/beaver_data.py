import os
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


def get_beaver_dataset(
    dataset_name, split="30k_train", category=None, hint_category=True
):
    train_dataset = load_dataset(dataset_name, split=split)
    test_split = "30k_test" if split == "30k_train" else "330k_test"
    test_dataset = load_dataset(dataset_name, split=test_split)

    train_dataset = reformat_beaver_dataset(train_dataset, hint_category)
    test_dataset = reformat_beaver_dataset(test_dataset, hint_category)

    if category is not None:
        train_dataset = filter_by_category(train_dataset, category)
        test_dataset = filter_by_category(test_dataset, category)

    return train_dataset, test_dataset


def reformat_beaver_dataset(dataset, hint_category=True):
    """Reformat the BeaverTail dataset to be compatible with the SafetyDataset
    class."""
    prompt = dataset["prompt"]
    response = dataset["response"]
    is_safe = dataset["is_safe"]
    category_str = []

    for category_dict in dataset["category"]:
        if hint_category:
            cat_str = [k for k, v in category_dict.items() if v]
            if len(cat_str) == 0:
                cat_str = ["safety,_ethics,_and_legality"]
        else:
            cat_str = ["safety,_ethics,_and_legality"]
        category_str.append(cat_str[0])

    return {
        "prompt": prompt,
        "response": response,
        "is_safe": is_safe,
        "category": category_str,
    }


def filter_by_category(dataset, category):
    """Filter the dataset to include only entries that match the specified
    category."""
    filtered_prompt = []
    filtered_response = []
    filtered_is_safe = []
    filtered_category = []

    for i in range(len(dataset["prompt"])):
        if (
            category
            in dataset["category"][i]
            # or dataset["category"][i] == "safety,_ethics,_and_legality"
        ):
            filtered_prompt.append(dataset["prompt"][i])
            filtered_response.append(dataset["response"][i])
            filtered_is_safe.append(dataset["is_safe"][i])
            filtered_category.append(dataset["category"][i])

    return {
        "prompt": filtered_prompt,
        "response": filtered_response,
        "is_safe": filtered_is_safe,
        "category": filtered_category,
    }


def get_categories() -> List[str]:
    """Define list of all categories."""
    return [
        "animal_abuse",
        "child_abuse",
        "controversial_topics,politics",
        "discrimination,stereotype,injustice",
        "drug_abuse,weapons,banned_substance",
        "financial_crime,property_crime,theft",
        "hate_speech,offensive_language",
        "misinformation_regarding_ethics,laws_and_safety",
        "non_violent_unethical_behavior",
        "privacy_violation",
        "self_harm",
        "sexually_explicit,adult_content",
        "terrorism,organized_crime",
        "violence,aiding_and_abetting,incitement",
        "safety,_ethics,_and_legality",
    ]
