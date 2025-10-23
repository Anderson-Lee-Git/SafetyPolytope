import logging
import os
import json
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig

from safety_polytope.common.load_util import load_model_and_tokenizer
from safety_polytope.common.outputs import evaluate_model
from safety_polytope.data.safety_data import (
    get_dataset,
    get_hidden_states_dataloader,
    get_safety_dataloader,
    get_format_fn,
)
from safety_polytope.polytope.lm_constraints import PolytopeConstraint

log = logging.getLogger("test")


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="test_model_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    if cfg.dataset.hidden_states_path != "None":
        log.info(f"Loading hidden states from {cfg.dataset.hidden_states_path}")
        hs_data = torch.load(cfg.dataset.hidden_states_path)
        test_dataloader = get_hidden_states_dataloader(hs_data["test"], shuffle=False)

        safety_model = PolytopeConstraint(
            model=None,
            tokenizer=None,
            num_phi=cfg.dataset.num_phi,
            train_on_hs=True,
        )
    else:
        _, test_data = get_dataset(cfg.dataset.name_or_path, cfg.dataset)
        model, tokenizer = load_model_and_tokenizer(cfg.model_path)
        safety_model = PolytopeConstraint(model, tokenizer)
        test_dataloader = get_safety_dataloader(
            cfg.dataset.name_or_path,
            test_data,
            batch_size=64,
            format_fn=get_format_fn(cfg.dataset.dataset_type),
            dataset_type=cfg.dataset.dataset_type,
            shuffle=False,
        )

    if cfg.run_summary_path:
        with open(cfg.run_summary_path, "r") as f:
            run_summary = json.load(f)
        trained_weights_path = run_summary["weights_save_path"]
    else:
        trained_weights_path = cfg.dataset.trained_weights_path

    trained_weights = torch.load(trained_weights_path, weights_only=False)
    safety_model.phi = trained_weights.phi.to(safety_model.device)
    safety_model.threshold = trained_weights.threshold.to(safety_model.device)
    safety_model.feature_extractor = trained_weights.feature_extractor.to(
        safety_model.device
    )

    test_result = evaluate_model(safety_model, test_dataloader, plot_id="test")
    log.info(f"Test acc: {test_result.accuracy:.3f}")
    log.info(f"Test fpr: {test_result.false_positive_rate:.3f}")
    log.info(f"Test fnr: {test_result.false_negative_rate:.3f}")

    if cfg.output_run_summary:
        run_summary["test_model_result"] = {
            "test_acc": test_result.accuracy,
            "test_fpr": test_result.false_positive_rate,
            "test_fnr": test_result.false_negative_rate,
            "test_f1": test_result.f1_score,
        }
        with open(Path(os.getcwd()) / "run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=4)

    torch.save(test_result, "test_outputs.pth")


if "__main__" == __name__:
    main()
