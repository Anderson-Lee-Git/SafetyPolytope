import logging
import os
import json
import hydra
from pathlib import Path
import torch
from omegaconf import DictConfig

from safety_polytope.common.load_util import (
    load_safe_rep_model,
)
from safety_polytope.data.safety_data import (
    get_dataset,
    get_safety_dataloader,
    get_format_fn,
)

log = logging.getLogger("test")


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="eval_generation_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    _, test_data = get_dataset(cfg.dataset.name_or_path, cfg.dataset)
    test_dataloader = get_safety_dataloader(
        cfg.dataset.name_or_path,
        test_data,
        batch_size=32,
        format_fn=get_format_fn(cfg.dataset.dataset_type),
        dataset_type=cfg.dataset.dataset_type,
        shuffle=False,
        subset_size=cfg.subset_size,
    )
    safety_model, safety_tokenizer = load_safe_rep_model(cfg)

    original_responses = []
    safe_responses = []
    prompts = []
    labels = []
    categories = []

    for batch in test_dataloader:
        text, label, category = batch
        input_length = 256
        messages = [
            [
                {"role": "user", "content": text[i]},
            ]
            for i in range(len(text))
        ]
        inputs = safety_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = safety_tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
            max_length=input_length,
        ).to(safety_model.device)
        input_length = model_inputs["input_ids"].shape[1]
        original_outputs = safety_model.lm_model.generate(
            **model_inputs, max_new_tokens=256
        )
        original_response = safety_tokenizer.batch_decode(
            original_outputs[:, input_length:],
            skip_special_tokens=True,
        )
        outputs = safety_model.generate(**model_inputs, max_new_tokens=256)
        response = safety_tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
        )
        log.info(f"Prompt: {text[0]}")
        log.info(f"Original Response: {original_response[0]}")
        log.info(f"Response: {response[0]}")
        original_responses.append(original_response)
        safe_responses.append(response)
        prompts.extend(text)
        labels.extend([l.item() for l in label])
        categories.extend(category)

    del safety_model, safety_tokenizer
    torch.cuda.empty_cache()

    out_dir = Path(os.getcwd())
    flattened_original_responses = []
    flattened_safe_responses = []
    for r in original_responses:
        flattened_original_responses.extend(r)
    for r in safe_responses:
        flattened_safe_responses.extend(r)
    output_data = {
        "prompts": prompts,
        "original_responses": flattened_original_responses,
        "safe_responses": flattened_safe_responses,
        "labels": labels,
        "categories": categories,
    }
    with open(out_dir / "output_data.json", "w") as f:
        json.dump(output_data, f)


if "__main__" == __name__:
    main()
