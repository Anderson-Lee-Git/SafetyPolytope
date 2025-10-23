import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from safety_polytope.common.load_util import load_model_and_tokenizer, get_model_name
from safety_polytope.data.safety_data import (
    get_format_fn,
    get_dataset,
    get_safety_dataloader,
)
from safety_polytope.polytope.lm_constraints import PolytopeConstraint
from safety_polytope.data.utils import merge_hidden_states

from accelerate import Accelerator

accelerator = Accelerator()

log = logging.getLogger("polytope")


def get_hidden_states_dict(cfg, dataset_cfg, data, model, tokenizer):
    dataloader = get_safety_dataloader(
        dataset_cfg.name_or_path,
        data,
        format_fn=get_format_fn(dataset_cfg.dataset_type),
        model="",
        batch_size=cfg.batch_size,
        shuffle=False,
        dataset_type=dataset_cfg.dataset_type,
    )

    hidden_states_list = []
    labels_list = []
    input_texts_list = []
    categories_list = []

    for batch in tqdm(dataloader, desc="Processing Batches"):
        text, label, category = batch
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            layer_hs = output.hidden_states[cfg.layer_number]
            last_token_hs = layer_hs[:, -1, :]

        hidden_states_list.append(last_token_hs.float().cpu().numpy())
        labels_list.append(label.float().cpu().numpy())
        input_texts_list.extend(text)
        categories_list.extend(category)

        if len(hidden_states_list) * cfg.batch_size >= cfg.total_datapoints:
            break

    result = {
        "hidden_states": np.concatenate(hidden_states_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0),
        "input_texts": input_texts_list,
        "categories": categories_list,
    }
    return result


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="save_hs_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    model_name = get_model_name(cfg.model_path)

    save_dir = f"{cfg.save_hs_root_dir}/{cfg.dataset.short_name}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Saving hidden states to {save_dir}")

    # Resolve dataset category handling
    categories_to_process = []
    get_categories_fn = None

    # Select category list provider based on dataset
    if "BeaverTails" in cfg.dataset.name_or_path:
        from safety_polytope.data.beaver_data import get_categories as beaver_categories

        get_categories_fn = beaver_categories
    elif "wildguard" in cfg.dataset.name_or_path:
        from safety_polytope.data.wildguard_data import (
            get_categories as wildguard_categories,
        )

        get_categories_fn = wildguard_categories
    else:
        raise ValueError(
            f"Dataset {cfg.dataset.name_or_path} not supported for categories."
        )

    if cfg.get("single_category"):
        categories_to_process = [cfg.single_category]
    elif cfg.get("iterate_categories", True):
        categories_to_process = list(get_categories_fn())
    else:
        # Fallback to using provided category or 'all'
        categories_to_process = [cfg.dataset.get("category", None) or "all"]

    # Load model/tokenizer once
    model, tokenizer = load_model_and_tokenizer(cfg.model_path)

    for category in categories_to_process:
        category_for_filename = category if category is not None else "all"
        pth_save_path = os.path.join(
            save_dir,
            f"hidden_states_{category_for_filename}.pth",
        )

        if os.path.exists(pth_save_path) and cfg.get("use_cache", True):
            log.info(f"Hidden states already exist at {pth_save_path}, skipping.")
            continue

        # Build a per-category dataset config without mutating the global cfg
        dataset_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.dataset, resolve=True)
        )
        # Use None for 'all' so dataset loaders don't filter
        dataset_cfg.category = None if category_for_filename == "all" else category

        train_data, test_data = get_dataset(cfg.dataset.name_or_path, dataset_cfg)

        train_dict = get_hidden_states_dict(
            cfg, dataset_cfg, train_data, model, tokenizer
        )
        test_dict = get_hidden_states_dict(
            cfg, dataset_cfg, test_data, model, tokenizer
        )

        torch.save({"train": train_dict, "test": test_dict}, pth_save_path)
        log.info(f"Hidden states, labels, and inputs saved to {pth_save_path}")

    # Optionally merge after generation when iterating categories
    if cfg.get("merge_after_generation", True) and cfg.get("iterate_categories", True):
        try:
            merge_hidden_states(
                base_path=cfg.save_hs_root_dir,
                dataset_name=cfg.dataset.short_name,
                model_file_name=model_name,
                get_categories=get_categories_fn,
            )
        except Exception as exc:
            log.warning(f"Merging hidden states failed: {exc}")


if __name__ == "__main__":
    main()
