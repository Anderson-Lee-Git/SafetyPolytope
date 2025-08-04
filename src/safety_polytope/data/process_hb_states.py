import argparse
import glob
import json
import os
import re

import torch
from torch.utils.data import random_split


def load_hidden_states(hidden_states_file):
    return torch.load(hidden_states_file, weights_only=False)


def load_labels(labels_file):
    with open(labels_file, "r") as f:
        labels_dict = json.load(f)
    return labels_dict


def match_hidden_states_and_labels(
    hidden_states, labels_dict, max_tokens=None
):
    all_hidden_states = []
    all_labels = []

    for idx, (key, entries) in enumerate(labels_dict.items()):
        for entry in entries:
            # Safety label is the opposite of "attack success" label
            sentence_label = int(not entry["label"])
            hidden_state = hidden_states[idx]
            # Truncate if max_tokens is specified
            if max_tokens is not None:
                hidden_state = hidden_state[:max_tokens]
            all_hidden_states.append(hidden_state)
            all_labels.extend([sentence_label] * hidden_state.shape[0])

    all_hidden_states = torch.cat(all_hidden_states)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    assert all_hidden_states.shape[0] == all_labels.shape[0]
    return all_hidden_states, all_labels


def split_dataset(all_hidden_states, all_labels, test_size=0.2):
    dataset = list(zip(all_hidden_states, all_labels))
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train_dataset, test_dataset


def process_method(root, method, model, max_tokens=None):
    hidden_states_pattern = f"hidden_states_{model}_layer20_*.pth"
    label_filename_pattern = f"{model}_*.json"
    if method in ["DirectRequest", "HumanJailbreaks"]:
        hidden_states_dir = os.path.join(
            root, method, "default", "hidden_states"
        )
        labels_dir = os.path.join(root, method, "default", "results")
    else:
        hidden_states_dir = os.path.join(root, method, model, "hidden_states")
        labels_dir = os.path.join(root, method, model, "results")

    # Find the matching hidden states file
    hidden_states_files = glob.glob(
        os.path.join(hidden_states_dir, hidden_states_pattern)
    )

    # Filter out files with "part" in their names
    hidden_states_files = [
        f for f in hidden_states_files if not re.search(r"_part\d+\.pth$", f)
    ]

    if not hidden_states_files:
        raise FileNotFoundError(
            f"No matching hidden states file found in {hidden_states_dir}"
        )

    # Sort files by modification time (most recent first) and take the first
    # one
    if len(hidden_states_files) > 1:
        print(
            f"Warning: Multiple hidden states files found in {hidden_states_dir}. Using the most recent one."
        )
        print(f"Hidden states files: {hidden_states_files}")
        print(f"Using file: {hidden_states_files[0]}")

    hidden_states_file = max(hidden_states_files, key=os.path.getmtime)

    hidden_states = load_hidden_states(hidden_states_file)

    # Find the matching label file
    labels_file_pattern = os.path.join(labels_dir, label_filename_pattern)
    matching_files = glob.glob(labels_file_pattern)
    if not matching_files:
        raise FileNotFoundError(
            f"No matching label file found in {labels_dir}"
        )
    labels_file = matching_files[
        0
    ]  # Take the first matching file if multiple exist

    labels_dict = load_labels(labels_file)
    all_hidden_states, all_labels = match_hidden_states_and_labels(
        hidden_states, labels_dict, max_tokens
    )

    return all_hidden_states, all_labels


def main(args):
    root = args.root
    methods = args.methods.split(",")
    model = args.model
    data_save_path = args.output
    max_tokens = args.max_tokens

    all_hidden_states = []
    all_labels = []

    for method in methods:
        print(f"Processing method: {method}")
        hidden_states, labels = process_method(root, method, model, max_tokens)
        all_hidden_states.append(hidden_states)
        all_labels.append(labels)

    # Concatenate all hidden states and labels
    all_hidden_states = torch.cat(all_hidden_states)
    all_labels = torch.cat(all_labels)

    # Ensure all data is on CPU
    all_hidden_states = all_hidden_states.cpu().float()
    all_labels = all_labels.cpu().float()

    train_dataset, test_dataset = split_dataset(all_hidden_states, all_labels)

    # Save train and test data
    data = {
        "train": {
            "hidden_states": torch.stack([x[0] for x in train_dataset]),
            "labels": torch.stack([x[1] for x in train_dataset]),
        },
        "test": {
            "hidden_states": torch.stack([x[0] for x in test_dataset]),
            "labels": torch.stack([x[1] for x in test_dataset]),
        },
    }

    os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
    torch.save(data, data_save_path)
    print(f"Data saved to {data_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process hidden states from multiple methods."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./HarmBench/results",
        help="Root directory for results",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="GCG,DirectRequest,HumanJailbreaks",
        help="Comma-separated list of methods to process",
    )
    parser.add_argument(
        "--model", type=str, default="llama2_7b", help="Model name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/hb_word_states_data.pth",
        help="Output file path",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to keep per sentence.",
    )
    args = parser.parse_args()
    main(args)
