from typing import Optional

import numpy as np
import torch.multiprocessing
import tqdm
from torch.utils.data import DataLoader, Dataset

from safety_polytope.data.beaver_data import get_beaver_dataset
from safety_polytope.data.wildguard_data import get_wildguard_dataset
from safety_polytope.data.model_instructions import get_instruction


def get_dataset(
    dataset_name_or_path: str,
    dataset_cfg,
    seed: Optional[int] = 1,
):
    split = dataset_cfg.split
    subset = dataset_cfg.subset

    if hasattr(dataset_cfg, "category") and dataset_cfg.category != "None":
        category = dataset_cfg.category
    else:
        category = None

    hint_category = dataset_cfg.get("hint_category", True)

    torch.multiprocessing.set_sharing_strategy("file_system")

    if "BeaverTails" in dataset_name_or_path:
        train_data, test_data = get_beaver_dataset(
            dataset_name_or_path, split, category, hint_category
        )
    elif "wildguard" in dataset_name_or_path:
        train_data, test_data = get_wildguard_dataset(category, split)
    else:
        raise ValueError(f"Dataset {dataset_name_or_path} not supported.")
    return train_data, test_data


def format_prompt_response_plain(instruction, category, prompt, response):
    output = f"{prompt}\n"
    output += f"{response}\n"
    return output


def format_prompt_plain(instruction, category, prompt):
    output = f"{prompt}\n"
    return output


def get_format_fn(dataset_type):
    if dataset_type == "prompt_response":
        return format_prompt_response_plain
    elif dataset_type == "prompt":
        return format_prompt_plain
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported.")


class PromptResponseDataset(Dataset):
    def __init__(self, data, instruction, format_fn, subset_size=1.0):
        self.data = []
        self.label = []
        self.category = []
        self.method = []
        self.subset_size = subset_size
        self.generate_data(data, instruction, format_fn)
        print(f"Example data: {self.data[0]}")

    def __len__(self):
        return int(len(self.data) * self.subset_size)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.category[idx]

    def generate_data(self, data, instruction, format_fn):
        desc = "Preparing data"
        with tqdm.tqdm(total=len(data["prompt"]), desc=desc) as pbar:
            for idx in range(len(data["prompt"])):
                pbar.update(1)
                prompt = data["prompt"][idx]
                is_safe = data["is_safe"][idx]
                response = data["response"][idx]
                category = data["category"][idx]

                self.data.append(format_fn(instruction, category, prompt, response))
                self.label.append(is_safe)
                self.category.append(category)

                if "method" in data.keys():
                    self.method.append(data["method"][idx])


class PromptDataset(Dataset):
    def __init__(self, data, instruction, format_fn, subset_size=1.0):
        self.data = []
        self.label = []
        self.category = []
        self.method = []
        self.subset_size = subset_size
        self.generate_data(data, instruction, format_fn)
        print(f"Example data: {self.data[0]}")

    def __len__(self):
        return int(len(self.data) * self.subset_size)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.category[idx]

    def generate_data(self, data, instruction, format_fn):
        desc = "Preparing data"
        with tqdm.tqdm(total=len(data["prompt"]), desc=desc) as pbar:
            for idx in range(len(data["prompt"])):
                pbar.update(1)
                prompt = data["prompt"][idx]
                is_safe = data["is_safe"][idx]
                category = data["category"][idx]

                self.data.append(format_fn(instruction, category, prompt))
                self.label.append(is_safe)
                self.category.append(category)

                if "method" in data.keys():
                    self.method.append(data["method"][idx])


class HiddenStatesDataset(Dataset):
    def __init__(self, data, subset_size=1.0):
        self.data = data["hidden_states"]
        self.label = data["labels"]
        self.category = data["categories"]

        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data)
        if isinstance(self.label, np.ndarray):
            self.label = torch.from_numpy(self.label)
        self.data = self.data.to(dtype=torch.float32)
        self.label = self.label.to(dtype=torch.float32)
        self.category = self.category
        self.total_categories = np.unique(self.category).tolist()
        print(f"Type of category: {type(self.category)}")
        if subset_size < 1.0:
            indices = np.random.choice(
                len(self.data), size=int(len(self.data) * subset_size), replace=False
            )
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.category = [self.category[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.category[idx]

    def sample_with_label_distribution(
        self, label_distribution: dict, num_samples: int
    ):
        """
        Subsample the dataset so that the number of samples is num_samples,
        with the proportions of each label as specified in label_distribution.

        Args:
            label_distribution: dict with keys 0, 1 and values summing to 1.
            num_samples: int, total number of samples in the subset.
        """
        # Convert tensors to numpy for easier indexing if needed
        if isinstance(self.label, torch.Tensor):
            labels_np = self.label.cpu().numpy()
        else:
            labels_np = np.array(self.label)
        # Get indices for each label
        indices_by_label = {
            label: np.where(labels_np == label)[0] for label in label_distribution
        }
        # Compute the number of samples per label
        samples_per_label = {
            label: int(round(ratio * num_samples))
            for label, ratio in label_distribution.items()
        }

        # To ensure the sum equals num_samples due to possible rounding
        sum_samples = sum(samples_per_label.values())
        if sum_samples != num_samples:
            # Fix the difference by adding/removing from the largest group
            diff = num_samples - sum_samples
            # Choose the label with the largest ratio (fallback: label 1)
            if diff != 0:
                sorted_labels = sorted(label_distribution.items(), key=lambda x: -x[1])
                label_to_adjust = sorted_labels[0][0]
                samples_per_label[label_to_adjust] += diff

        # Randomly choose samples for each label
        chosen_indices = []
        rng = np.random.default_rng()
        for label, count in samples_per_label.items():
            available = indices_by_label[label]
            if len(available) < count:
                raise ValueError(
                    f"Not enough samples for label {label} to satisfy requested distribution."
                )
            chosen = rng.choice(available, size=count, replace=False)
            chosen_indices.extend(chosen.tolist())

        # Shuffle selected indices to mix all labels
        rng.shuffle(chosen_indices)

        # Filter self.data, self.label, self.category
        if isinstance(self.data, torch.Tensor):
            self.data = self.data[chosen_indices]
        else:
            self.data = [self.data[i] for i in chosen_indices]
        if isinstance(self.label, torch.Tensor):
            self.label = self.label[chosen_indices]
        else:
            self.label = [self.label[i] for i in chosen_indices]
        if isinstance(self.category, torch.Tensor):
            self.category = self.category[chosen_indices]
        else:
            self.category = [self.category[i] for i in chosen_indices]

    def summary_stats(self):
        category_distribution = {k: 0 for k in self.total_categories}
        label_distribution = {k: 0 for k in [0, 1]}
        for label in self.label.tolist():
            label_distribution[label] += 1
        for cat in self.category:
            category_distribution[cat] += 1
        return {
            "num_samples": len(self.data),
            "num_categories": len(self.total_categories),
            "category_distribution": category_distribution,
            "label_distribution": label_distribution,
        }


def get_hidden_states_dataloader(
    hidden_states_data, batch_size=32, shuffle=True, num_workers=1, subset_size=1.0
):
    dataloader = DataLoader(
        HiddenStatesDataset(hidden_states_data, subset_size=subset_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


def get_safety_dataloader(
    dataset_name_or_path,
    data,
    model=None,
    format_fn=format_prompt_response_plain,
    batch_size=32,
    shuffle=True,
    dataset_type="prompt_response",
    subset_size=1.0,
):
    instruction = get_instruction(model)

    if (
        "BeaverTails" not in dataset_name_or_path
        and "wildguard" not in dataset_name_or_path
    ):
        raise ValueError(f"Dataset {dataset_name_or_path} not supported.")

    if dataset_type == "prompt_response":
        dataset = PromptResponseDataset(data, instruction, format_fn, subset_size)
    elif dataset_type == "prompt":
        dataset = PromptDataset(data, instruction, format_fn, subset_size)
    else:
        raise ValueError("Dataset not supported.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader
