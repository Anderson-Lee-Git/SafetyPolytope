import csv
import difflib
import logging
import os
import re
import string
from collections import Counter, defaultdict

import hydra
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from omegaconf import DictConfig
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety_polytope.data.safety_data import get_hidden_states_dataloader
from safety_polytope.interpret.plot_kld import plot_edge_kl_divergences
from safety_polytope.polytope.lm_constraints import PolytopeConstraint

log = logging.getLogger(__name__)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


def plot_constraint_violation_distributions(
    safety_model, dataloader, output_folder="constraint_violation_plots"
):
    os.makedirs(output_folder, exist_ok=True)
    safety_model.eval()
    num_edges = safety_model.phi.shape[0]
    all_violations = [[] for _ in range(num_edges)]

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="Processing inputs for violation distributions"
        ):
            inputs, _ = batch
            inputs = inputs.to(safety_model.device)

            hs_rep = safety_model.get_hidden_states_representation(inputs)
            features = safety_model.feature_extractor(hs_rep)
            violations = torch.matmul(
                safety_model.phi, features.T
            ) - safety_model.threshold.unsqueeze(1)

            for edge in range(num_edges):
                edge_violations = violations[edge].cpu().numpy()
                all_violations[edge].extend(edge_violations)

    for edge in range(num_edges):
        edge_violations = np.array(all_violations[edge])

        # Filter for positive violations only
        positive_violations = edge_violations[edge_violations > 0]

        if len(positive_violations) == 0:
            log.info(f"No positive violations for Edge {edge}. Skipping plot.")
            continue

        plt.figure(figsize=(10, 6))

        # Calculate the histogram
        counts, bins, _ = plt.hist(positive_violations, bins=50, alpha=0.7)

        plt.title(
            f"Positive Constraint Violation Distribution for Edge {edge}"
        )
        plt.xlabel("Constraint Violation")
        plt.ylabel("Number of Data Points")
        plt.grid(True, alpha=0.3)

        # Add statistics to the plot
        mean = np.mean(positive_violations)
        median = np.median(positive_violations)
        std = np.std(positive_violations)
        plt.text(
            0.05,
            0.95,
            f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std:.2f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
        )

        y_max = 1.5 * np.max(counts)
        plt.ylim(0, y_max)
        plt.savefig(
            os.path.join(
                output_folder,
                f"edge_{edge}_positive_violation_distribution.png",
            )
        )
        plt.close()

    log.info(
        f"Saved constraint violation distribution plots to {output_folder}"
    )


def plot_cosine_similarity_heatmap(similarity_matrix, save_path):
    """Plot and save a heatmap of the cosine similarity matrix."""
    plt.figure(figsize=(40, 38))
    sns.heatmap(
        similarity_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1
    )
    plt.title("Cosine Similarity of Polytope Edges")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def extract_relevant_text(input_text):
    """Extract the relevant part of the text between 'Human:' and '[/INST]'."""
    if input_text is None:
        return ""  # Return empty string if input is None
    match = re.search(r"Human:(.*?)\[/INST\]", input_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return input_text.strip()  # Return the original text if no match is found


class UniqueViolationList:
    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.items = []
        self.texts = set()

    def add(self, violation, text):
        if text not in self.texts:
            self.texts.add(text)
            self.items.append((violation, text))
            self.items.sort(key=lambda x: x[0], reverse=True)
            if self.maxlen and len(self.items) > self.maxlen:
                removed_text = self.items.pop()[1]
                self.texts.remove(removed_text)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


def find_top_violations(safety_model, dataloader, input_texts, num_top=10):
    """Find top unique text inputs that lead to maximum constraint violation
    for each edge."""
    safety_model.eval()
    num_edges = safety_model.phi.shape[0]
    top_violations = [
        UniqueViolationList(maxlen=num_top) for _ in range(num_edges)
    ]

    input_text_counter = Counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Processing inputs")
        ):
            inputs, _ = batch
            inputs = inputs.to(safety_model.device)

            hs_rep = safety_model.get_hidden_states_representation(inputs)
            features = safety_model.feature_extractor(hs_rep)
            violations = torch.matmul(
                safety_model.phi, features.T
            ) - safety_model.threshold.unsqueeze(1)

            for edge in range(num_edges):
                edge_violations = violations[edge].cpu().numpy()
                for idx, violation in enumerate(edge_violations):
                    input_text = input_texts[
                        batch_idx * dataloader.batch_size + idx
                    ]
                    formatted_text = extract_relevant_text(input_text)

                    input_text_counter[formatted_text] += 1
                    top_violations[edge].add(violation, formatted_text)

    log.info(f"Total unique input texts: {len(input_text_counter)}")
    log.info(f"Most common input texts: {input_text_counter.most_common(5)}")

    return top_violations


def load_data_and_model(cfg):
    log.info(
        f"Loading trained weights from {cfg.dataset.trained_weights_path}"
    )
    trained_weights = torch.load(cfg.dataset.trained_weights_path)

    log.info(f"Loading dataset from {cfg.dataset.hidden_states_path}")
    hs_data = torch.load(cfg.dataset.hidden_states_path)
    dataloader = get_hidden_states_dataloader(hs_data["test"], shuffle=True)
    input_texts = hs_data["test"]["input_texts"]

    safety_model = PolytopeConstraint(
        model=None,
        tokenizer=None,
        train_on_hs=True,
    )
    safety_model.phi = trained_weights.phi
    safety_model.threshold = trained_weights.threshold
    safety_model.feature_extractor = trained_weights.feature_extractor
    safety_model.to(safety_model.device)

    return safety_model, dataloader, input_texts


def analyze_cosine_similarity(safety_model):
    phi = safety_model.phi.detach().cpu().numpy()

    log.info("Calculating cosine similarity among polytope edges...")
    similarity_matrix = cosine_similarity(phi)

    avg_cosine_similarity = np.mean(
        similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    )
    log.info(f"Average cosine similarity: {avg_cosine_similarity:.4f}")

    log.info("Plotting cosine similarity heatmap...")
    plot_save_path = "edge_cosine_similarity_heatmap.png"
    plot_cosine_similarity_heatmap(similarity_matrix, plot_save_path)
    log.info(f"Cosine similarity heatmap saved to {plot_save_path}")

    np.save("edge_cosine_similarity_matrix.npy", similarity_matrix)


def analyze_violations(safety_model, dataloader, input_texts, num_top):
    log.info("Finding text inputs with maximum constraint violations...")
    top_violations = find_top_violations(
        safety_model, dataloader, input_texts, num_top=num_top
    )

    all_violations = set()
    for edge_violations in top_violations:
        for _, text in edge_violations:
            all_violations.add(text)

    log.info(
        f"Total unique violations across all edges: {len(all_violations)}"
    )

    return top_violations


def save_top_violations(top_violations):
    log.info("Saving top violations to file...")
    with open("top_violations.txt", "w") as f:
        for edge, violations in enumerate(top_violations):
            f.write(f"Edge {edge}:\n")
            for i, (violation, input_text) in enumerate(violations, 1):
                f.write(f"  {i}. Violation: {violation:.4f}\n")
                f.write(f"     {input_text}\n\n")
            f.write("\n")


def clean_word(word):
    """Remove punctuation and spaces, and convert to lowercase."""
    # Remove punctuation and spaces
    word = re.sub(f"[{re.escape(string.punctuation)}]", "", word)
    # Convert to lowercase
    return word.lower()


def find_word_index(words, cleaned_word):
    """Find the index of a cleaned word in a list of uncleaned words."""
    for i, word in enumerate(words):
        if clean_word(word) == cleaned_word:
            return i
    return -1  # Return -1 if the word is not found


def get_violations(safety_model, model, tokenizer, text, cfg):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(
        safety_model.device
    )
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        layer_hs = outputs.hidden_states[cfg.layer_number]
        last_token_hs = layer_hs[:, -1, :]
        # Convert to float32 before passing to feature extractor
        last_token_hs = last_token_hs.to(torch.float32)
        features = safety_model.feature_extractor(last_token_hs)
        violations = torch.matmul(
            safety_model.phi, features.T
        ) - safety_model.threshold.unsqueeze(1)
    return violations.squeeze().cpu().numpy()


def mask_word_with_context(text, word, context_size=3):
    if not text:
        return (
            "",
            False,
        )  # Return empty string and False if text is empty or None

    words = text.split()
    lemma = lemmatizer.lemmatize(word)
    pattern = re.compile(
        rf"\b{re.escape(lemma)}(?:s|ed|ing)?\b", re.IGNORECASE
    )
    masked_words = words.copy()
    word_found = False

    for i, w in enumerate(words):
        if pattern.search(w):
            word_found = True
            start = max(0, i - context_size)
            end = min(len(words), i + context_size + 1)
            for j in range(start, end):
                masked_words[j] = "######"

    return " ".join(masked_words), word_found


def analyze_masked_violations(
    safety_model, input_texts, model, tokenizer, cfg, words_to_mask
):
    log.info("Analyzing violations with contextual word importance...")
    num_edges = safety_model.phi.shape[0]
    masked_violations = [[] for _ in range(num_edges)]
    original_violations = [[] for _ in range(num_edges)]
    log.info(f"Total number of input texts: {len(input_texts)}")
    log.info(f"Number of unique input texts: {len(set(input_texts))}")

    for idx, input_text in enumerate(
        tqdm(input_texts, desc="Processing inputs")
    ):
        text_violations = get_violations(
            safety_model, model, tokenizer, input_text, cfg
        )

        for edge in range(num_edges):
            original_violation = text_violations[edge]

            # Skip if original violation is negative
            if original_violation < 0:
                continue

            original_violations[edge].append((original_violation, input_text))

            relevant_text = extract_relevant_text(input_text)

            for word in words_to_mask:
                # Mask the word and its context
                masked_relevant_text, word_found = mask_word_with_context(
                    relevant_text, word
                )

                # Skip if the word is not in the text
                if not word_found:
                    continue

                # Reconstruct the full text with the masked relevant part
                masked_text = input_text.replace(
                    relevant_text, masked_relevant_text
                )

                masked_violations_array = get_violations(
                    safety_model, model, tokenizer, masked_text, cfg
                )
                masked_violation = masked_violations_array[edge]

                masked_violations[edge].append(
                    (
                        original_violation,
                        masked_violation,
                        word,
                        masked_text,
                    )
                )

        if idx >= cfg.total_datapoints:
            break

    return masked_violations, original_violations


def calculate_kl_divergence(masked_violations):
    kl_divergences = []

    for edge in range(len(masked_violations)):
        edge_kl = {}

        # Group both original and masked violations by word
        word_violations = defaultdict(lambda: {"original": [], "masked": []})

        if len(masked_violations[edge]) == 0:
            kl_divergences.append(edge_kl)
            continue

        for pairs in masked_violations[edge]:
            orig_v, masked_v, word, _ = pairs
            word_violations[word]["original"].append(orig_v)
            word_violations[word]["masked"].append(masked_v)

        for word, violations in word_violations.items():
            orig_violations = np.array(violations["original"])
            updated_violations = np.array(violations["masked"])

            # Skip calculation if there's no data for this word
            if len(orig_violations) == 0 or len(updated_violations) == 0:
                continue

            assert orig_violations.shape == updated_violations.shape

            # Calculate probabilities
            orig_prob = softmax(orig_violations)
            masked_prob = softmax(updated_violations)

            # Calculate KL divergence
            kl_div = entropy(orig_prob, masked_prob)

            edge_kl[word] = kl_div

        kl_divergences.append(edge_kl)

    return kl_divergences


def plot_kl_divergences(kl_divergences, save_dir="./kld"):
    # Convert the nested dictionary to a DataFrame
    data = []
    for edge, word_dict in enumerate(kl_divergences):
        for word, kl_div in word_dict.items():
            data.append({"Edge": edge, "Word": word, "KL Divergence": kl_div})
    df = pd.DataFrame(data)

    # Get unique words
    words = df["Word"].unique()

    # Create a plot for each word
    for word in words:
        # Filter data for the current word
        word_data = df[df["Word"] == word]

        # Sort by KL Divergence from max to min
        word_data = word_data.sort_values("KL Divergence", ascending=False)

        # Create the plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Edge", y="KL Divergence", data=word_data)

        plt.title(f'KL Divergence for "{word}" across Edges')
        plt.xlabel("Edge")
        plt.ylabel("KL Divergence")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent cutting off labels
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(save_dir, f"kl_divergence_{word}.png"))
        plt.close()

    log.info(f"KL divergence plots saved to {save_dir}")


def save_violations_and_kl(masked_violations, kl_divergences, save_dir="."):
    violations_path = os.path.join(save_dir, "violations.csv")
    kl_path = os.path.join(save_dir, "kl_divergences.csv")

    # Save violations
    log.info(f"Saving violations to {violations_path}")
    with open(violations_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Edge",
                "Word",
                "Original Violation",
                "Masked Violation",
                "Relevant Text",
            ]
        )

        for edge, edge_violations in enumerate(masked_violations):
            for orig_v, masked_v, word, masked_text in edge_violations:
                relevant_text = extract_relevant_text(masked_text)
                writer.writerow([edge, word, orig_v, masked_v, relevant_text])

    log.info(f"Violations saved to {violations_path}")

    # Save KL divergences
    log.info(f"Saving KL divergences to {kl_path}")
    with open(kl_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Edge", "Word", "KL Divergence"])

        for edge, edge_kl in enumerate(kl_divergences):
            for word, kl_div in edge_kl.items():
                writer.writerow([edge, word, kl_div])

    log.info(f"KL divergences saved to {kl_path}")
    kl_plot_dir = os.path.join(save_dir, "kld_plots")
    plot_edge_kl_divergences(kl_path, kl_plot_dir)


def clean_and_lemmatize_word(word):
    """Clean the word and return its lemma."""
    # Clean the word (remove punctuation and convert to lowercase)
    cleaned_word = clean_word(word)
    # Lemmatize the word
    return lemmatizer.lemmatize(cleaned_word)


def guess_words_to_mask(model, tokenizer, top_texts, top_k=100):
    model.eval()
    all_words = []

    template = "[INST] The following text activates a safety filter: [TEXT] "
    template += "Which words in this text might be concerning? Repeat the "
    template += "text and replace the "
    template += "potentially problematic words with [MASK]. [/INST]"

    # Flatten the list of violations across all edges and remove duplicates
    all_violations = set()
    for edge_violations in top_texts:
        for violation, text in edge_violations:
            all_violations.add(text)
    unique_texts = list(all_violations)

    log.info(f"Processing {len(unique_texts)} unique violations")

    for text in tqdm(unique_texts, desc="Processing violations"):
        # Prepare input using the template
        input_text = template.replace("[TEXT]", text)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode the generated output
        generated_text = tokenizer.decode(
            outputs[0], skip_special_tokens=False
        )

        # Extract the model's response (everything after the input)
        response = generated_text.split(input_text)[-1].strip()

        # Compare the original text with the response
        original_words = word_tokenize(text)
        response_words = word_tokenize(response)

        diff = list(difflib.ndiff(original_words, response_words))

        for i, s in enumerate(diff):
            if s.startswith("- "):
                # This word was in the original text but not in the
                # response (potentially masked)
                word = s[2:]
                if word and word != tokenizer.mask_token:
                    all_words.append(clean_and_lemmatize_word(word))
            elif s.startswith("+ ") and s[2:] == tokenizer.mask_token:
                # A mask token was added in the response
                if i > 0 and diff[i - 1].startswith("- "):
                    # The word before this mask was removed, so it's
                    # likely the masked word
                    word = diff[i - 1][2:]
                    if word:
                        all_words.append(clean_and_lemmatize_word(word))

    # Get the most common words across all texts
    word_counts = Counter(all_words)
    words_to_mask = [word for word, _ in word_counts.most_common(top_k)]

    return words_to_mask


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs", config_name="interpret_config"
)
def main(cfg: DictConfig):
    safety_model, dataloader, input_texts = load_data_and_model(cfg)

    # Load the Mistral model using AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure the model's config is updated with the pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Add [MASK] token if it's not already in the tokenizer
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        model.resize_token_embeddings(len(tokenizer))

    print(f"Mask token: {tokenizer.mask_token}")
    print(f"Mask token ID: {tokenizer.mask_token_id}")

    analyze_cosine_similarity(safety_model)

    plot_constraint_violation_distributions(safety_model, dataloader)

    top_violations = analyze_violations(
        safety_model, dataloader, input_texts, cfg.num_top
    )
    save_top_violations(top_violations)

    # Step 2: Guess words to mask
    # words_to_mask = guess_words_to_mask(model, tokenizer, top_violations)
    # words_to_mask = ["fake", "kill", "dead", "bomb", "eat"]
    words_to_mask = ["fight", "prank", "cheat", "lie", "steal", "sex"]

    log.info(f"Words to mask: {words_to_mask}")

    # Step 3: Analyze masked violations
    masked_violations, original_violations = analyze_masked_violations(
        safety_model, input_texts, model, tokenizer, cfg, words_to_mask
    )

    # Calculate KL divergence
    kl_divergences = calculate_kl_divergence(masked_violations)

    # Save violations and KL divergences
    save_violations_and_kl(
        masked_violations,
        kl_divergences,
    )

    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
