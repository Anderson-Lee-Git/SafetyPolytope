import time
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import find_executable_batch_size
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)


def get_elapsed_time(start_time: float) -> str:
    return f"[{time.time() - start_time:.2f}s]"


def print_memory_stats(prefix=""):
    """Print current GPU memory usage with an optional prefix."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(
        f"{prefix} GPU Memory: {allocated:.2f}GB allocated,"
        f"{reserved:.2f}GB reserved"
    )


class SafeRepModel(PreTrainedModel, GenerationMixin):
    requires_grad_for_generation = True

    def __init__(
        self,
        pretrained_model_name_or_path=None,
        steer_layer=20,
        steer_first_n_tokens=20,
        use_backup_response=False,
        lambda_weight=2.0,
        safe_violation_weight=0.0001,
        projection=False,
        backup_response="Sorry, I can't answer your request.",
        device_map="auto",
        **model_kwargs,
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        super().__init__(config)
        self.model_name_or_path = pretrained_model_name_or_path

        if pretrained_model_name_or_path:
            self.lm_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                device_map=device_map,
                **model_kwargs,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
            )
            self.model = self.lm_model.model
            self.lm_head = self.lm_model.lm_head

        self.lm_head_device = next(self.lm_head.parameters()).device

        # Move backup_ids to the same device as lm_head
        if use_backup_response:
            self.backup_ids = self.tokenizer(
                backup_response, return_tensors="pt"
            ).input_ids.to(self.lm_head_device)

        self.phi, self.threshold, self.feature_extractor = None, None, None
        self.rejection_logit = None
        self.backup_logit = None

        self.use_backup_response = use_backup_response
        self.projection = projection
        self.lambda_weight = lambda_weight
        self.safe_violation_weight = safe_violation_weight
        if use_backup_response:
            self.backup_ids = self.tokenizer(
                backup_response, return_tensors="pt"
            ).input_ids
            self.backup_logit = None

        self.safe = None
        if steer_layer is None:
            self.steer_layer = len(self.model.layers) - 1
        else:
            self.steer_layer = steer_layer
        self.steer_first_n_tokens = steer_first_n_tokens
        self.tokens_steered = 0
        self.is_new_generation = True

    def _apply_backup_response(
        self,
        logits: torch.Tensor,
        safe_mask: torch.Tensor,
        num_logits_to_keep: int,
        start_time: float,
    ) -> torch.Tensor:
        """Apply backup response logits to unsafe samples.

        Args:
            logits: Model output logits of shape [batch_size, seq_len,
                vocab_size]
            safe_mask: Boolean mask indicating which samples are unsafe
            num_logits_to_keep: Number of last logits to keep
            start_time: Start time for logging
        """
        if num_logits_to_keep > 0:
            logits = logits[:, -num_logits_to_keep:, :]
            current_token_idx = num_logits_to_keep - 1
        else:
            current_token_idx = -1

        # Get the current position in the backup response
        backup_position = self.tokens_steered
        backup_token_id = self.backup_ids[0, backup_position]

        current_logits = torch.zeros(
            self.model.config.vocab_size,
            device=logits.device,
            dtype=logits.dtype,
        )
        current_logits[backup_token_id] = 100.0

        # Apply modified logits
        logits[safe_mask, current_token_idx, :] = current_logits

        self.tokens_steered += 1
        if self.tokens_steered == self.backup_ids.shape[1] - 1:
            self.tokens_steered = 0
        return logits

    def _apply_optimization(
        self,
        hidden_states: torch.Tensor,
        safe_mask: torch.Tensor,
        start_time: float,
    ) -> torch.Tensor:
        """Apply hidden state optimization for unsafe samples."""
        print("Starting optimization...")
        optimized_states = self.optimize_hidden_states(
            hidden_states, safe_mask
        )
        self.tokens_steered += 1
        return optimized_states

    def edit_logit_forward(self, input_ids, attention_mask, **kwargs):
        start_time = time.time()

        # Ensure input_ids and attention_mask are on the correct device
        input_device = input_ids.device

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=kwargs.get("position_ids"),
            past_key_values=kwargs.get("past_key_values"),
            inputs_embeds=kwargs.get("inputs_embeds"),
            use_cache=kwargs.get("use_cache"),
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=True,
            return_dict=True,
            cache_position=kwargs.get("cache_position"),
        )

        # Get hidden states and move to lm_head device if needed
        hidden_states = outputs.hidden_states[self.steer_layer]
        token_hidden_states = hidden_states[:, -1, :].to(self.lm_head_device)

        # Check safety using features on the correct device
        features = self.feature_extractor(token_hidden_states)
        unsafe_mask = ~self.check_constraint(features)

        # Move last_hidden_state to lm_head device for logits computation
        last_hidden = outputs.last_hidden_state.to(self.lm_head_device)
        logits = self.lm_head(last_hidden)

        if unsafe_mask.any():
            logits = self._apply_backup_response(
                logits,
                unsafe_mask,
                kwargs.get("num_logits_to_keep", 0),
                start_time,
            )

        if kwargs.get("num_logits_to_keep", 0) > 0:
            logits = logits[:, -kwargs["num_logits_to_keep"] :, :]

        # Move outputs back to input device if needed
        if logits.device != input_device:
            logits = logits.to(input_device)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def steer_forward(self, input_ids, attention_mask, verbose=False, **args):
        position_ids = args.get("position_ids")
        past_key_values = args.get("past_key_values")
        inputs_embeds = args.get("inputs_embeds")
        use_cache = args.get("use_cache")
        output_attentions = args.get("output_attentions")
        output_hidden_states = args.get("output_hidden_states")
        return_dict = args.get("return_dict")
        cache_position = args.get("cache_position")
        num_logits_to_keep = args.get("num_logits_to_keep", 0)

        start_time = time.time()
        steered = False

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.model.config.use_cache
        )

        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the",
                "same time, and must specify either one",
            )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if inputs_embeds.shape[1] != 1:
            self.is_new_generation = True
            self.tokens_steered = 0

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache:
            if not isinstance(past_key_values, Cache):
                return_legacy_cache = True
                if past_key_values is None:
                    past_key_values = DynamicCache()
                else:
                    past_key_values = DynamicCache.from_legacy_cache(
                        past_key_values
                    )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()  # type: ignore
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if (
            "llama" in self.model_name_or_path.lower()
            or "qwen" in self.model_name_or_path.lower()
        ):
            causal_mask = self.model._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions,
            )
        elif "mistral" in self.model_name_or_path.lower():
            causal_mask = self.model._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                use_cache,
                output_attentions,
            )
        else:
            raise NotImplementedError(
                f"Model {self.model_name_or_path} not supported"
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = (
            () if output_hidden_states else None
        )
        all_self_attns: Optional[Tuple[torch.FloatTensor, ...]] = (
            () if output_attentions else None
        )
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.model.layers):
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            # Apply steering at the specified layer
            if idx == self.steer_layer:
                if self.tokens_steered < self.steer_first_n_tokens:
                    features = self.feature_extractor(hidden_states[:, -1, :])
                    safe_mask = ~self.check_constraint(features)

                    if safe_mask.any():
                        hidden_states[:, -1, :] = self._apply_optimization(
                            hidden_states[:, -1, :], safe_mask, start_time
                        )
                        steered = True

                    del features
                    del safe_mask

            if use_cache:
                next_decoder_cache = layer_outputs[
                    2 if output_attentions else 1
                ]

            if output_attentions and all_self_attns is not None:
                all_self_attns = all_self_attns + (layer_outputs[1],)

        hidden_states = self.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()  # type: ignore

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        logits = self.lm_head(
            hidden_states[:, -num_logits_to_keep:, :]
        ).float()

        if steered and verbose:
            baseline_outputs = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
            )
            baseline_hidden = baseline_outputs.hidden_states
            baseline_logits = baseline_outputs.logits

            print("\n=== Hidden States Comparison ===")
            hidden_diff = (
                (all_hidden_states[-1] - baseline_hidden[-1])
                .abs()
                .mean()
                .item()
            )
            print(f"Last layer mean abs difference: {hidden_diff:.6f}")
            steered_diff = (
                (
                    all_hidden_states[self.steer_layer]
                    - baseline_hidden[self.steer_layer]
                )
                .abs()
                .mean()
                .item()
            )
            print(f"Steer layer mean abs difference: {steered_diff:.6f}")

            print("\n=== Logits Comparison ===")
            print(f"Shape: {baseline_logits.shape}")
            logits_diff = (logits - baseline_logits).abs().mean().item()
            print(f"Average absolute difference: {logits_diff:.6f}")

            if num_logits_to_keep > 0:
                baseline_top5 = torch.topk(baseline_logits[0, -1], 5)
                steered_top5 = torch.topk(logits[0, -1], 5)

                print("\nTop 5 token predictions:")
                print("Baseline:")
                for score, idx in zip(
                    baseline_top5.values, baseline_top5.indices
                ):
                    token = self.tokenizer.decode([idx])
                    print(f"  {token}: {score.item():.3f}")

                print("After steering:")
                for score, idx in zip(
                    steered_top5.values, steered_top5.indices
                ):
                    token = self.tokenizer.decode([idx])
                    print(f"  {token}: {score.item():.3f}")

        # Clear unnecessary outputs
        if not args.get("output_hidden_states", False):
            outputs.hidden_states = None
        if not args.get("output_attentions", False):
            outputs.attentions = None

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[
            Union[Cache, List[torch.FloatTensor]]
        ] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if self.use_backup_response:
            forward_fn = self.edit_logit_forward
        else:
            forward_fn = self.steer_forward

        output = forward_fn(
            input_ids,
            attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        return output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        return self.lm_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

    def check_constraint(self, features):
        if (
            hasattr(self, "phi")
            and self.phi is not None
            and hasattr(self, "threshold")
            and self.threshold is not None
        ):
            # Polytope constraint
            phi = self.phi.to(features.device).to(features.dtype)
            xi = self.threshold.to(features.device).to(features.dtype)
            constraints = torch.matmul(features, phi.T) <= xi
            return constraints.all(dim=1)
        else:
            # MLP classifier
            logits = self.classifier(F.relu(self.hidden_layer(features)))
            probs = torch.sigmoid(logits)
            return (probs >= 0.5).squeeze(1)

    def optimize_hidden_states(
        self,
        hidden_state,
        mask,
        num_iterations=1,
        verbose=True,
        initial_batch_size=32,
    ):
        # Add a guard to ensure we're not in inference mode
        if torch._C._get_tracing_state() or torch.is_inference_mode_enabled():
            warnings.warn(
                "optimize_hidden_states called in inference mode or while tracing. "
                "Returning original hidden states without optimization."
            )
            return hidden_state.clone()  # Return unmodified hidden states

        optimized_hidden_states = hidden_state.clone()

        unsafe_indices = torch.nonzero(mask).view(-1)

        device = hidden_state.device
        dtype = hidden_state.dtype

        # Check if we're using polytope or MLP
        is_polytope = (
            hasattr(self, "phi")
            and self.phi is not None
            and hasattr(self, "threshold")
            and self.threshold is not None
        )

        if is_polytope:
            phi = self.phi.to(device).to(dtype)
            threshold = self.threshold.to(device).to(dtype)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.eval()

        @find_executable_batch_size(starting_batch_size=initial_batch_size)
        def optimize_batch(batch_size):
            nonlocal optimized_hidden_states, unsafe_indices

            # Get total number of unsafe samples
            num_unsafe = unsafe_indices.size(0)
            rep_dim = hidden_state.size(-1)

            for start_idx in range(0, num_unsafe, batch_size):
                end_idx = min(start_idx + batch_size, num_unsafe)
                batch_indices = unsafe_indices[start_idx:end_idx]

                # Initialize optimization for this batch
                batch_hs = (
                    hidden_state[batch_indices]
                    .clone()
                    .detach()
                    .requires_grad_(True)
                )
                original_hs = hidden_state[batch_indices].clone().detach()
                optimizer = torch.optim.SGD([batch_hs], lr=0.01)

                # Optimization loop for this batch
                for iter_idx in range(num_iterations):
                    optimizer.zero_grad()
                    features = self.feature_extractor(batch_hs)
                    # Note: This line below is very important - It forces
                    # PyTorch to create a new memory for features instead of
                    # reusing the cached one, so the loss can actually go down.

                    if is_polytope:
                        features_reshaped = features.view(1, -1)
                        violations = (
                            torch.matmul(features_reshaped, phi.T) - threshold
                        )
                        safety_violation = violations.sum()
                        positive_violations = torch.relu(violations)
                        positive_violation_penalty = torch.sum(
                            positive_violations
                        )
                    else:
                        # MLP optimization
                        hidden_output = F.relu(self.hidden_layer(features))
                        logits = self.classifier(hidden_output)
                        probs = torch.sigmoid(logits)
                        # For MLP, we want to maximize the probability of being safe
                        safety_violation = -torch.sum(probs)
                        positive_violations = (
                            -logits
                        )  # Negative logits indicate unsafe predictions
                        positive_violation_penalty = torch.sum(
                            torch.relu(positive_violations)
                        )

                    distance = torch.sum(torch.abs(batch_hs - original_hs))
                    # distance = torch.norm(batch_hs - original_hs, p=2)

                    loss = (
                        distance / rep_dim
                        + self.safe_violation_weight * safety_violation
                        + self.lambda_weight * positive_violation_penalty
                    )

                    print(f"Loss: {loss.item()}")

                    # Check if loss is inf or NaN before backpropagation
                    if torch.isinf(loss).any() or torch.isnan(loss).any():
                        print(
                            f"Warning: Detected inf/NaN in loss at iteration {iter_idx}. Skipping optimization."
                        )
                        # Reset this batch to original hidden states
                        with torch.no_grad():
                            batch_hs.copy_(original_hs)
                        break  # Exit the optimization loop for this batch
                    else:
                        loss.backward()
                        optimizer.step()

                # After optimization, update the main tensor
                with torch.no_grad():
                    features = self.feature_extractor(batch_hs)

                    if is_polytope:
                        final_violations = (
                            torch.matmul(features, phi.T) - threshold
                        )
                        final_positive_violations = torch.relu(
                            final_violations
                        )
                        final_safety_violation = final_violations.sum()
                    else:
                        hidden_output = F.relu(self.hidden_layer(features))
                        logits = self.classifier(hidden_output)
                        probs = torch.sigmoid(logits)
                        final_violations = -logits
                        final_positive_violations = torch.relu(
                            final_violations
                        )
                        final_safety_violation = -torch.sum(probs)

                    final_distance = torch.sum(
                        torch.abs(batch_hs - original_hs)
                    )

                    optimized_hidden_states[batch_indices] = batch_hs.detach()

                    del features, final_violations, final_positive_violations
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return optimized_hidden_states

        with torch.set_grad_enabled(True):
            result = optimize_batch()
            return result

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        polytope_weight_path,
        **kwargs,
    ):
        # Create the base model first
        safe_model = cls(pretrained_model_name_or_path, **kwargs)

        # Load the saved safety model
        safety_model = torch.load(polytope_weight_path, weights_only=False)

        # Check if we loaded a PolytopeConstraint or BaselineMLP
        if hasattr(safety_model, "phi") and safety_model.phi is not None:
            # Polytope model
            safe_model.phi = safety_model.phi
            safe_model.threshold = safety_model.threshold
            safe_model.feature_extractor = safety_model.feature_extractor
            print("Loaded polytope constraint model")
        else:
            # MLP model
            safe_model.feature_extractor = safety_model.feature_extractor
            safe_model.hidden_layer = safety_model.hidden_layer
            safe_model.classifier = safety_model.classifier
            print("Loaded MLP classifier model")

        # Move all components to the correct device and dtype
        device = next(safe_model.model.parameters()).device
        dtype = next(safe_model.model.parameters()).dtype

        # Move feature extractor
        if hasattr(safe_model, "feature_extractor") and isinstance(
            safe_model.feature_extractor, nn.Module
        ):
            safe_model.feature_extractor.to(device=device, dtype=dtype)

        # Move phi and threshold for polytope models
        if hasattr(safe_model, "phi") and safe_model.phi is not None:
            safe_model.phi = nn.Parameter(
                safe_model.phi.to(device=device, dtype=dtype)
            )
        if (
            hasattr(safe_model, "threshold")
            and safe_model.threshold is not None
        ):
            safe_model.threshold = nn.Parameter(
                safe_model.threshold.to(device=device, dtype=dtype)
            )

        # Move hidden_layer and classifier for MLP models
        if hasattr(safe_model, "hidden_layer"):
            safe_model.hidden_layer.to(device=device, dtype=dtype)
        if hasattr(safe_model, "classifier"):
            safe_model.classifier.to(device=device, dtype=dtype)

        # Store device information for easier access
        safe_model.lm_head_device = device

        print(f"Feature extractor type: {type(safe_model.feature_extractor)}")
        print(f"Polytope weight path: {polytope_weight_path}")
        print(f"Model device: {device}")

        return safe_model


class SafeRepConfig(PretrainedConfig):
    model_type = "safe_rep_model"

    def __init__(
        self,
        pretrained_model_path=None,
        polytope_weight_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_model_path = pretrained_model_path
        self.polytope_weight_path = polytope_weight_path
        self.model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
