import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from safety_polytope.common.outputs import ConstraintOutputs


def get_model_hidden_states(model, tokenizer, inputs):
    encoded_inputs = tokenizer(inputs, return_tensors="pt")
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model(
            **encoded_inputs.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        del encoded_inputs
        del outputs
        return hidden_states


def get_model_hidden_states_loop(
    model, tokenizer, inputs, to_np=True, disable_tqdm=False
):
    hidden_states = []
    with tqdm(total=len(inputs), disable=disable_tqdm) as pbar:
        for in_text in inputs:
            hs = get_model_hidden_states(model, tokenizer, in_text)
            if to_np:
                hs = [hs.cpu().numpy() for hs in hs]
            hidden_states.append(hs)
            pbar.update(1)
    return hidden_states


class PolytopeConstraint(torch.nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        learn_phi=True,
        num_phi=10,
        entropy_weight=1.0,
        train_on_hs=False,
        valid_edges_threshold=0,
        unsafe_weight=1.0,
        feature_dim=256,
        use_nonlinear=False,
        entropy_assignment=True,
        f_l1_weight=0.1,
        phi_l1_weight=0.0001,
        margin=1.0,
        num_feature_extractor_layers=1,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.phi = None
        self.threshold = None
        self.phi_categories = []
        self.learn_phi = learn_phi
        self.entropy_weight = entropy_weight
        self.num_phi = num_phi
        self.train_on_hs = train_on_hs
        self.valid_edges_threshold = valid_edges_threshold
        self.unsafe_weight = unsafe_weight
        self.entropy_assignment = entropy_assignment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_feature_extractor_layers = num_feature_extractor_layers
        self.feature_dim = feature_dim
        self.use_nonlinear = use_nonlinear
        self.feature_extractor = None
        self.f_l1_weight = f_l1_weight
        self.phi_l1_weight = phi_l1_weight
        self.margin = margin

        if not self.train_on_hs:
            self.rand_init_phi_theta(num_phi)

    def rand_init_phi_theta(self, num_phi, x="random input"):
        hs_rep = self.get_hidden_states_representation(x)
        rep_dim = hs_rep.shape[1]

        if self.use_nonlinear:
            if self.num_feature_extractor_layers > 1:
                layers = [
                    nn.Linear(rep_dim, self.feature_dim),
                    nn.ReLU(),
                ]
                for _ in range(self.num_feature_extractor_layers - 1):
                    layers.append(nn.Linear(self.feature_dim, self.feature_dim))
                    layers.append(nn.ReLU())
                self.feature_extractor = nn.Sequential(*layers).to(self.device)
            else:
                self.feature_extractor = nn.Sequential(
                    nn.Linear(rep_dim, self.feature_dim),
                    nn.ReLU(),
                ).to(self.device)

            phi_dim = self.feature_dim
        else:
            self.feature_extractor = nn.Sequential(nn.Identity()).to(self.device)
            phi_dim = rep_dim

        self.phi = torch.nn.Parameter(torch.randn(num_phi, phi_dim, device=self.device))
        self.threshold = torch.nn.Parameter(torch.randn(num_phi, device=self.device))

        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def get_hidden_states_representation(self, x):
        if self.train_on_hs:
            return x

        hidden_states = get_model_hidden_states_loop(
            self.model, self.tokenizer, x, to_np=False, disable_tqdm=True
        )
        hs_rep = torch.stack([hs[-1][:, -1, :] for hs in hidden_states], dim=1)
        hs_rep = hs_rep.squeeze(0)
        return hs_rep

    def get_safety_prediction(self, hs_rep, return_cost=False):
        features = self.feature_extractor(hs_rep)

        # Apply polytope edges
        cost = torch.matmul(features, self.phi.t())
        is_safe = torch.all(cost < self.threshold, dim=1)

        if return_cost:
            return is_safe, cost
        return is_safe

    def calculate_entropy(self, violation_edges, label):
        num_unsafe = torch.sum(label == 0).float()
        total_num_edges = self.phi.shape[0]

        # Ensure violation_edges and label have the same first dimension
        batch_size = min(violation_edges.shape[0], label.shape[0])
        violation_edges = violation_edges[:batch_size]
        label = label[:batch_size]

        num_constraint_violations = torch.zeros(
            total_num_edges, device=violation_edges.device
        )
        num_constraint_violations.scatter_add_(
            0,
            violation_edges[label == 0].long(),
            torch.ones(torch.sum(label == 0), device=violation_edges.device),
        )

        distribution = num_constraint_violations / (
            num_unsafe + 1e-10
        )  # Add small epsilon to avoid division by zero
        entropy = -torch.sum(distribution * torch.log2(distribution + 1e-10))

        return entropy

    def violation_entropy_assignment(
        self, violations, label, entropy_threshold=None, max_attempts=100
    ):
        batch_size, num_edges = violations.shape
        max_violations, max_violation_edges = torch.max(violations, dim=1)

        if entropy_threshold is None:
            entropy_threshold = 0.5 * np.log2(num_edges)

        # Create new tensors instead of modifying in-place
        new_max_violations = max_violations.clone()
        new_max_violation_edges = max_violation_edges.clone()

        # Get indices of unsafe examples
        unsafe_indices = torch.where(label == 0)[0]

        # If there are no unsafe examples, return original values
        if len(unsafe_indices) == 0:
            current_entropy = self.calculate_entropy(new_max_violation_edges, label)
            return new_max_violations, new_max_violation_edges, current_entropy

        current_entropy = self.calculate_entropy(new_max_violation_edges, label)
        entropy = current_entropy
        attempts = 0
        num_valid_edges = 0

        while entropy < entropy_threshold and attempts < max_attempts:
            # Randomly pick one unsafe batch item
            batch_idx = unsafe_indices[
                random.randint(0, len(unsafe_indices) - 1)
            ].item()

            sorted_violations, sorted_indices = torch.sort(
                violations[batch_idx], descending=True
            )
            valid_edges = sorted_indices[sorted_violations > self.valid_edges_threshold]
            num_valid_edges += len(valid_edges)

            if len(valid_edges) <= 1:
                chosen_edge = sorted_indices[1].item()
            else:
                chosen_edge = valid_edges[
                    random.randint(1, len(valid_edges) - 1)
                ].item()

            # Try reassigning this edge
            temp_edges = new_max_violation_edges.clone()
            temp_edges[batch_idx] = chosen_edge
            new_entropy = self.calculate_entropy(temp_edges, label)

            if new_entropy > entropy:
                new_max_violations[batch_idx] = violations[batch_idx, chosen_edge]
                new_max_violation_edges[batch_idx] = chosen_edge
                entropy = new_entropy

            attempts += 1

        return new_max_violations, new_max_violation_edges, entropy

    def forward(self, x, label=None, reduction="sum"):
        assert (
            self.phi is not None and self.threshold is not None
        ), "Please initialize phi and threshold first."

        hs_rep = self.get_hidden_states_representation(x)
        dtype = hs_rep.dtype
        device = hs_rep.device

        is_safe, cost = self.get_safety_prediction(hs_rep, return_cost=True)
        feature = self.feature_extractor(hs_rep)

        violations = cost - self.threshold.to(dtype)
        violation_idx = torch.nonzero(torch.relu(violations))

        batch_size = hs_rep.shape[0]
        probs = torch.zeros(batch_size, dtype=dtype, device=device)

        loss, entropy_loss, additional_params = None, None, None

        if label is not None:
            if self.entropy_assignment:
                max_violations, max_violation_edges, entropy = (
                    self.violation_entropy_assignment(
                        violations, label, entropy_threshold=None
                    )
                )
            else:
                max_violations, max_violation_edges = torch.max(violations, dim=1)
            safe_violations = torch.sum(torch.relu(self.margin + violations), axis=1)
            unsafe_violations = torch.relu(self.margin - max_violations)

            # Calculate regularization losses
            f_l1_loss = torch.norm(feature, p=1, dim=1).mean()
            phi_l1_loss = torch.norm(self.phi, p=1, dim=1).mean()

            f_l1_term = self.f_l1_weight * f_l1_loss
            phi_l1_term = self.phi_l1_weight * phi_l1_loss

            entropy_loss = self.calculate_entropy(max_violation_edges, label)
            edge_entropy_loss = self.entropy_weight * entropy_loss

            # Apply reduction based on parameter
            if reduction == "sum" or reduction == "mean":
                # Sum the losses
                safe_loss = safe_violations[label == 1]
                unsafe_loss = unsafe_violations[label == 0]
                loss = torch.sum(safe_loss) + self.unsafe_weight * torch.sum(
                    unsafe_loss
                )
                # Add regularization terms (already scalar values)
                loss = loss + f_l1_term + phi_l1_term - edge_entropy_loss
                if reduction == "mean":
                    loss = loss / batch_size
            elif reduction == "none" or reduction is None:
                # Create a tensor with batch_size elements
                loss = torch.zeros(batch_size, device=device, dtype=dtype)

                # Distribute regularization terms evenly across all samples
                reg_term = (f_l1_term + phi_l1_term - edge_entropy_loss) / batch_size

                # Set loss for safe samples
                safe_mask = label == 1
                if torch.any(safe_mask):
                    loss[safe_mask] = safe_violations[safe_mask] + reg_term

                # Set loss for unsafe samples (with unsafe weight)
                unsafe_mask = label == 0
                if torch.any(unsafe_mask):
                    loss[unsafe_mask] = (
                        self.unsafe_weight * unsafe_violations[unsafe_mask] + reg_term
                    )
            else:
                raise ValueError(f"Unsupported reduction mode: {reduction}")

            # Calculate average number of edges with violations >
            # valid_edges_threshold
            unsafe_mask = label == 0
            num_edges_with_violations_unsafe = torch.sum(
                violations[unsafe_mask] > 0, dim=1
            ).float()
            avg_edges_with_violations_unsafe = (
                torch.mean(num_edges_with_violations_unsafe).item()
                if torch.sum(unsafe_mask) > 0
                else 0.0
            )

            additional_params = {
                "safe_loss": (
                    torch.mean(safe_violations[label == 1]).item()
                    if torch.any(label == 1)
                    else 0.0
                ),
                "unsafe_loss": (
                    torch.mean(unsafe_violations[label == 0]).item()
                    if torch.any(label == 0)
                    else 0.0
                ),
                "entropy": entropy_loss.item(),
                "max_violation": max_violations.max().item(),
                "min_violation": max_violations.min().item(),
                "avg_violation": max_violations.mean().item(),
                "avg_activated_edges": avg_edges_with_violations_unsafe,
                "edge_entropy_loss": edge_entropy_loss.item(),
                "f_l1_loss": f_l1_loss.item(),
                "phi_l1_loss": phi_l1_loss.item(),
            }

        return ConstraintOutputs(
            is_safe=is_safe,
            probs=probs,
            violations=violations,
            loss=loss,
            entropy_loss=entropy_loss,
            additional_params=additional_params,
            violation_idx=violation_idx,
        )


class BaselineMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=16384,
        num_edges=50,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(self.device)

        # Hidden layer matching phi dimensions
        self.hidden_layer = nn.Linear(hidden_dim, num_edges).to(self.device)

        # Fixed classifier layer with output dimension 1
        self.classifier = nn.Linear(num_edges, 1, bias=False).to(self.device)

        # Initialize weights
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)

        # Initialize and freeze classifier weights
        with torch.no_grad():
            nn.init.xavier_uniform_(self.classifier.weight)
            self.classifier.weight.requires_grad = False

    def forward(self, x, label=None):
        x = x.to(self.device)
        features = self.feature_extractor(x)
        hidden = F.relu(self.hidden_layer(features))
        logits = self.classifier(hidden)
        probs = torch.sigmoid(logits)

        loss = None
        if label is not None:
            label = label.to(self.device).float().unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, label)

        return ConstraintOutputs(
            is_safe=(probs >= 0.5).squeeze(1),
            probs=probs.squeeze(1),  # probability of being safe
            loss=loss,
            entropy_loss=None,
            additional_params={"bce_loss": loss.item() if loss is not None else None},
            violations=None,
            violation_idx=None,
        )
