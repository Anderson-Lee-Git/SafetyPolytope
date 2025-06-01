import logging
import os

import hydra
import numpy as np
import torch
import torch.optim as optim
import tqdm
from omegaconf import DictConfig

from safety_polytope.common.load_util import set_seed
from safety_polytope.common.outputs import ModelResult, evaluate_model
from safety_polytope.data.safety_data import get_hidden_states_dataloader
from safety_polytope.polytope.lm_constraints import (
    BaselineMLP,
    PolytopeConstraint,
)

log = logging.getLogger("polytope")


def report_loss_callback(epoch, batch, avg_loss):
    loss_str = f"loss: {avg_loss:.3f}"
    log.info(f"[Epoch {epoch+1}, Batch {batch+1}] {loss_str}")


def train_model(
    model,
    dataloader,
    optimizer,
    num_epochs=1,
    disable_tqdm=False,
    callback_every=100,
    callback_fn=report_loss_callback,
):
    model.train()
    model.to(model.device)
    latest_loss = 0
    running_additional_params = {}

    for epoch in range(num_epochs):
        running_loss = []
        with tqdm.tqdm(total=len(dataloader), disable=disable_tqdm) as pbar:
            for i, (inputs, labels) in enumerate(dataloader):
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(model.device)
                labels = labels.float().to(model.device)
                outputs = model(inputs, labels)
                loss = outputs.loss
                additional_params = outputs.additional_params

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

                running_loss.append(loss.item())

                # Update running averages for additional params
                if additional_params:
                    for key, value in additional_params.items():
                        if key not in running_additional_params:
                            running_additional_params[key] = []
                        running_additional_params[key].append(value)

                if callback_fn and i % callback_every == 0:
                    latest_loss = np.mean(running_loss)
                    callback_fn(epoch, i, latest_loss)

                    # Print running averages of additional params
                    log.info("Running averages of additional parameters:")
                    for key, values in running_additional_params.items():
                        avg_value = np.mean(values)
                        log.info(f"  {key}: {avg_value:.4f}")

                    # Reset running averages after reporting
                    running_loss = []
                    running_additional_params = {}

    latest_loss = np.mean(running_loss)
    return latest_loss.item()


def get_model_input_dim(model_path):
    model_path = model_path.lower()
    if "llama" in model_path or "mistral" in model_path:
        return 4096
    elif "qwen" in model_path:
        return 1536
    else:
        raise NotImplementedError(
            f"Input dimension not defined for model: {model_path}"
        )


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="polytope_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    log.info(f"Loading hidden states from {cfg.dataset.hidden_states_path}")
    hs_data = torch.load(cfg.dataset.hidden_states_path, weights_only=False)
    dataloader = get_hidden_states_dataloader(hs_data["train"])
    test_dataloader = get_hidden_states_dataloader(hs_data["test"])

    if cfg.model_type == "polytope":
        safety_model = PolytopeConstraint(
            model=None,
            tokenizer=None,
            num_phi=cfg.dataset.num_phi,
            entropy_weight=cfg.entropy_weight,
            valid_edges_threshold=cfg.valid_edges_threshold,
            unsafe_weight=cfg.unsafe_weight,
            train_on_hs=True,
            use_nonlinear=cfg.use_nonlinear,
            entropy_assignment=cfg.entropy_assignment,
            feature_dim=cfg.feature_dim,
            f_l1_weight=cfg.f_l1_weight,
            phi_l1_weight=cfg.phi_l1_weight,
            margin=cfg.margin,
        )
        safety_model.rand_init_phi_theta(
            cfg.dataset.num_phi, x=hs_data["train"]["hidden_states"]
        )
    else:  # model_type == "mlp"
        safety_model = BaselineMLP(
            input_dim=hs_data["train"]["hidden_states"].shape[-1],
            hidden_dim=cfg.feature_dim,
            num_edges=cfg.dataset.num_phi,
        )

    optimizer = optim.Adam(safety_model.parameters(), lr=cfg.learning_rate)
    loss = train_model(
        safety_model, dataloader, optimizer, num_epochs=cfg.dataset.num_epochs
    )

    result_dir = os.getcwd()
    weight_save_path = os.path.join(result_dir, "weights.pth")

    if cfg.model_type == "polytope":
        model_result = ModelResult(
            safety_model.phi,
            safety_model.threshold,
            safety_model.feature_extractor,
        )
    else:  # model_type == "mlp"
        model_result = safety_model

    torch.save(model_result, weight_save_path)

    plot_const_freq = True if cfg.model_type == "polytope" else False
    test_result = evaluate_model(
        safety_model,
        test_dataloader,
        plot_id="test",
        plot_const_freq=plot_const_freq,
    )
    log.info(f"Test acc: {test_result.accuracy:.3f}")
    log.info(f"Test fpr: {test_result.false_positive_rate:.3f}")
    log.info(f"Test fnr: {test_result.false_negative_rate:.3f}")
    log.info(f"Test f1: {test_result.f1_score:.3f}")

    return test_result.accuracy


if "__main__" == __name__:
    main()
