"""
WikiText-2: Original Mamba vs State-Dependent Mamba (model_developing_v2.py)
=======================================================================

Experiment:
- 5 independent random seeds
- Same training windows for both models within each seed
- Same initialization seed for paired models
- Full validation and test evaluation
- Best validation checkpoint selected for each run
- Mean, standard deviation, paired t-test and confidence interval

Run in Google Colab with GPU enabled.

Where to put the files:
- model.py             -> original Mamba (in the cloned repo)
- model_developing_v2.py   -> the modified model (in the repo folder, OR upload it
                          to /content/ and the script copies it into the repo)
"""

import os
import sys
import time
import random
import shutil
import subprocess
import importlib
import importlib.util
import csv
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

REPO_URL = "https://github.com/amirho3inz1/mamba-state-dependent.git"
REPO_DIR = Path("/content/mamba-state-dependent")

ORIGINAL_MODULE_NAME = "model"
MODIFIED_MODULE_NAME = "model_developing_v2"      # <-- the modified model file

# Extra ModelArgs passed ONLY to the modified model.
# h_pool: mean | learned | window   |   h_norm: True/False
MODIFIED_ARGS = dict(h_pool="window", h_norm=True, pool_cut=10.0, pool_sharp=1.0)

SEQ_LEN = 64
BATCH_SIZE = 8

D_MODEL = 32
N_LAYER = 1
D_STATE = 8

EPOCHS = 3
SEEDS = [11, 22, 33, 44, 55]

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

# None means use the complete training split each epoch.
# For a quick pilot only, you can set this to 1000.
MAX_STEPS_PER_EPOCH = None

MAX_VAL_BATCHES = None

RESULTS_FILE = "/content/wikitext2_mamba_results.csv"


# ============================================================
# 2. INSTALL MISSING DEPENDENCIES
# ============================================================

def install_if_missing(module_name, package_name=None):

    if importlib.util.find_spec(module_name) is None:

        package_name = package_name or module_name

        print(f"Installing {package_name} ...")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            package_name
        ])


# ============================================================
# 3. RANDOM SEEDS
# ============================================================

def set_seed(seed):

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 4. REPOSITORY
# ============================================================

def prepare_repository():
    """Find model.py and model_developing_v2.py.

    Priority:
      1. Both files uploaded directly to /content  -> use them, no GitHub needed.
      2. Otherwise clone the repo and use its model.py; copy the modified
         file from /content into the repo if it is not already there.
    """

    content = Path("/content")

    local_original = content / f"{ORIGINAL_MODULE_NAME}.py"
    local_modified = content / f"{MODIFIED_MODULE_NAME}.py"

    # Option 1: everything uploaded to /content
    if local_original.exists() and local_modified.exists():

        print("Using model files uploaded to /content (no clone needed).")

        if str(content) not in sys.path:
            sys.path.insert(0, str(content))

        return

    # Option 2: clone the repository
    if not REPO_DIR.exists():

        REPO_DIR.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"Cloning {REPO_URL} ...")

        subprocess.check_call([
            "git",
            "clone",
            REPO_URL,
            str(REPO_DIR)
        ])

    else:

        print(f"Using repository: {REPO_DIR}")

    original_file = REPO_DIR / f"{ORIGINAL_MODULE_NAME}.py"
    modified_file = REPO_DIR / f"{MODIFIED_MODULE_NAME}.py"

    if not original_file.exists():

        if local_original.exists():
            shutil.copy(local_original, original_file)
            print(f"Copied {local_original} -> {original_file}")
        else:
            raise FileNotFoundError(
                f"{ORIGINAL_MODULE_NAME}.py not found in {REPO_DIR} "
                f"or /content. Upload it to /content."
            )

    if not modified_file.exists():

        if local_modified.exists():
            shutil.copy(local_modified, modified_file)
            print(f"Copied {local_modified} -> {modified_file}")
        else:
            raise FileNotFoundError(
                f"{MODIFIED_MODULE_NAME}.py not found in {REPO_DIR} "
                f"or /content. Upload it to /content "
                f"(Colab left sidebar > Files)."
            )

    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))


# ============================================================
# 5. DOWNLOAD AND TOKENIZE WIKITEXT-2
# ============================================================

def prepare_data():

    from datasets import load_dataset
    from transformers import AutoTokenizer
    import torch

    print("\nDownloading WikiText-2 raw ...")

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1"
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def encode_split(name):

        text = "\n".join(dataset[name]["text"])

        ids = tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]

        return torch.tensor(
            ids,
            dtype=torch.long
        )

    train = encode_split("train")
    valid = encode_split("validation")
    test = encode_split("test")

    print(f"\nVocabulary size: {tokenizer.vocab_size:,}")
    print(f"Train tokens:    {len(train):,}")
    print(f"Validation:      {len(valid):,}")
    print(f"Test:            {len(test):,}")

    return train, valid, test, tokenizer.vocab_size


# ============================================================
# 6. VALIDATION AND TEST BATCHES
# ============================================================

def make_eval_batches(tokens, seq_len, batch_size):

    usable = (
        (len(tokens) - 1) // seq_len
    ) * seq_len

    if usable <= 0:
        raise ValueError("Not enough tokens for evaluation.")

    x = tokens[:usable].view(-1, seq_len)

    y = tokens[1:usable + 1].view(-1, seq_len)

    batches = []

    for i in range(0, x.size(0), batch_size):

        batches.append((
            x[i:i + batch_size],
            y[i:i + batch_size]
        ))

    return batches


# ============================================================
# 7. LANGUAGE MODEL WRAPPER
# ============================================================

def build_lm(module, vocab_size, extra_args=None):

    import torch.nn as nn

    extra_args = extra_args or {}

    class LanguageModel(nn.Module):

        def __init__(self):

            super().__init__()

            self.args = module.ModelArgs(
                d_model=D_MODEL,
                n_layer=N_LAYER,
                vocab_size=vocab_size,
                d_state=D_STATE,
                **extra_args,
            )

            self.embedding = nn.Embedding(
                self.args.vocab_size,
                D_MODEL
            )

            self.blocks = nn.ModuleList([
                module.ResidualBlock(self.args)
                for _ in range(N_LAYER)
            ])

            self.norm = nn.LayerNorm(D_MODEL)

            self.lm_head = nn.Linear(
                D_MODEL,
                vocab_size,
                bias=False
            )

        def forward(self, token_ids):

            x = self.embedding(token_ids)

            for block in self.blocks:
                x = block(x)

            x = self.norm(x)

            logits = self.lm_head(x)

            return logits

    return LanguageModel()


# ============================================================
# 8. EVALUATION
# ============================================================

def evaluate(model, batches, device, max_batches=None):

    import torch
    import torch.nn.functional as F

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():

        for batch_index, (x, y) in enumerate(batches):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            if logits.ndim != 3 or logits.shape[:2] != y.shape:

                raise RuntimeError(
                    f"Unexpected output shape: {tuple(logits.shape)}"
                )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += y.numel()

    mean_loss = total_loss / total_tokens

    perplexity = torch.exp(
        torch.tensor(mean_loss)
    ).item()

    return mean_loss, perplexity


# ============================================================
# 9. TRAIN ONE MODEL
# ============================================================

def train_one(
    name,
    module,
    train_tokens,
    valid_batches,
    test_batches,
    vocab_size,
    device,
    batch_starts_by_epoch,
    seed,
    extra_args=None
):

    import torch
    import torch.nn.functional as F

    print("\n" + "=" * 70)
    print(f"Training: {name} | Seed: {seed}")
    if extra_args:
        print(f"Extra model args: {extra_args}")
    print("=" * 70)

    # Reset RNG so paired models start from the same random seed.
    set_seed(seed)

    model = build_lm(
        module,
        vocab_size,
        extra_args
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Smoke test
    sample_x = train_tokens[:SEQ_LEN].unsqueeze(0).to(device)

    with torch.no_grad():
        sample_logits = model(sample_x)

    expected_shape = (
        1,
        SEQ_LEN,
        vocab_size
    )

    if tuple(sample_logits.shape) != expected_shape:

        raise RuntimeError(
            f"Expected {expected_shape}, "
            f"got {tuple(sample_logits.shape)}"
        )

    print(f"Parameters: {n_params:,}")
    print(f"Output shape: {tuple(sample_logits.shape)}")

    best_val_ppl = float("inf")
    best_state = None

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        total_tokens = 0

        epoch_start = time.time()

        for step, starts in enumerate(
            batch_starts_by_epoch[epoch - 1]
        ):

            x = torch.stack([
                train_tokens[s:s + SEQ_LEN]
                for s in starts
            ])

            y = torch.stack([
                train_tokens[s + 1:s + SEQ_LEN + 1]
                for s in starts
            ])

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            # Progress display
            if (step + 1) % 200 == 0:

                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Step {step + 1}/"
                    f"{len(batch_starts_by_epoch[epoch - 1])}"
                )

        train_loss = total_loss / total_tokens

        train_ppl = torch.exp(
            torch.tensor(train_loss)
        ).item()

        val_loss, val_ppl = evaluate(
            model,
            valid_batches,
            device,
            MAX_VAL_BATCHES
        )

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train PPL: {train_ppl:.2f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"Time: {epoch_time / 60:.2f} min"
        )

        # Save best validation checkpoint
        if val_ppl < best_val_ppl:

            best_val_ppl = val_ppl

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    elapsed = time.time() - start_time

    # Restore best validation checkpoint
    if best_state is not None:

        model.load_state_dict(best_state)

    # Final full validation and test evaluation
    final_val_loss, final_val_ppl = evaluate(
        model,
        valid_batches,
        device,
        max_batches=None
    )

    test_loss, test_ppl = evaluate(
        model,
        test_batches,
        device,
        max_batches=None
    )

    print(f"\n{name} results:")
    print(f"Validation PPL: {final_val_ppl:.3f}")
    print(f"Test PPL:       {test_ppl:.3f}")
    print(f"Parameters:     {n_params:,}")
    print(f"Training time:  {elapsed / 60:.2f} min")

    del model
    del optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "val_ppl": final_val_ppl,
        "test_ppl": test_ppl,
        "params": n_params,
        "minutes": elapsed / 60
    }


# ============================================================
# 10. SAVE RESULTS AFTER EACH SEED
# ============================================================

def save_results(results):

    with open(
        RESULTS_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Seed",
            "Original_Val_PPL",
            "Modified_Val_PPL",
            "Original_Test_PPL",
            "Modified_Test_PPL",
            "Original_Params",
            "Modified_Params",
            "Original_Minutes",
            "Modified_Minutes"
        ])

        for r in results:

            writer.writerow([
                r["seed"],
                r["original"]["val_ppl"],
                r["modified"]["val_ppl"],
                r["original"]["test_ppl"],
                r["modified"]["test_ppl"],
                r["original"]["params"],
                r["modified"]["params"],
                r["original"]["minutes"],
                r["modified"]["minutes"]
            ])

    print(f"\nResults saved to: {RESULTS_FILE}")


# ============================================================
# 11. FINAL STATISTICAL ANALYSIS
# ============================================================

def statistical_analysis(results):

    import numpy as np
    from scipy.stats import ttest_rel, t

    original_test = np.array([
        r["original"]["test_ppl"]
        for r in results
    ])

    modified_test = np.array([
        r["modified"]["test_ppl"]
        for r in results
    ])

    # Positive means modified model is better.
    differences = original_test - modified_test

    improvements = (
        differences / original_test
    ) * 100.0

    n = len(differences)

    mean_difference = differences.mean()
    std_difference = differences.std(ddof=1)

    standard_error = std_difference / np.sqrt(n)

    t_critical = t.ppf(
        0.975,
        df=n - 1
    )

    ci_low = (
        mean_difference
        - t_critical * standard_error
    )

    ci_high = (
        mean_difference
        + t_critical * standard_error
    )

    # Two-sided paired t-test
    test_result = ttest_rel(
        original_test,
        modified_test
    )

    print("\n\n" + "=" * 90)
    print("FINAL RESULTS — WIKITEXT-2")
    print("=" * 90)

    print(
        f"{'Seed':<10}"
        f"{'Original PPL':>18}"
        f"{'Modified PPL':>18}"
        f"{'Improvement':>18}"
    )

    print("-" * 68)

    for r, improvement in zip(results, improvements):

        print(
            f"{r['seed']:<10}"
            f"{r['original']['test_ppl']:>18.3f}"
            f"{r['modified']['test_ppl']:>18.3f}"
            f"{improvement:>17.2f}%"
        )

    print("-" * 68)

    print(
        f"{'Mean':<10}"
        f"{original_test.mean():>18.3f}"
        f"{modified_test.mean():>18.3f}"
        f"{improvements.mean():>17.2f}%"
    )

    print(
        f"{'Std':<10}"
        f"{original_test.std(ddof=1):>18.3f}"
        f"{modified_test.std(ddof=1):>18.3f}"
        f"{improvements.std(ddof=1):>17.2f}%"
    )

    print("\nPaired comparison:")
    print(f"Mean PPL difference: {mean_difference:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Paired t-test p-value: {test_result.pvalue:.6f}")

    wins = np.sum(differences > 0)

    print(
        f"Modified model wins: {wins}/{n} seeds"
    )

    print("\nMean relative improvement:")
    print(f"{improvements.mean():.3f}%")

    print("\nInterpretation:")

    if mean_difference > 0 and ci_low > 0:

        print(
            "The modified model has lower mean test PPL, "
            "and the paired 95% confidence interval "
            "for the PPL difference is above zero."
        )

    elif mean_difference > 0:

        print(
            "The modified model has lower mean test PPL, "
            "but the confidence interval includes zero. "
            "The evidence is inconclusive."
        )

    else:

        print(
            "The modified model does not have lower "
            "mean test PPL in this experiment."
        )

    print(
        "\nImportant: five seeds provide preliminary evidence, "
        "not definitive proof of generalization."
    )


# ============================================================
# 12. MAIN EXPERIMENT
# ============================================================

def main():

    install_if_missing("einops")
    install_if_missing("datasets")
    install_if_missing("transformers")
    install_if_missing("scipy")

    import torch
    import numpy as np

    # GPU verification
    if not torch.cuda.is_available():

        raise RuntimeError(
            "GPU not detected. In Colab, select "
            "Runtime > Change runtime type > GPU."
        )

    device = torch.device("cuda")

    print("\nDevice:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    # Enable faster matrix operations on supported GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True

    # Repository and model imports
    prepare_repository()

    importlib.invalidate_caches()

    original_module = importlib.import_module(ORIGINAL_MODULE_NAME)
    modified_module = importlib.import_module(MODIFIED_MODULE_NAME)

    # If you edit model_developing_v2.py and re-run in the same session,
    # reload so the changes are picked up.
    modified_module = importlib.reload(modified_module)

    # Dataset
    train_tokens, val_tokens, test_tokens, vocab_size = (
        prepare_data()
    )

    valid_batches = make_eval_batches(
        val_tokens,
        SEQ_LEN,
        BATCH_SIZE
    )

    test_batches = make_eval_batches(
        test_tokens,
        SEQ_LEN,
        BATCH_SIZE
    )

    # Training window configuration
    max_start = len(train_tokens) - SEQ_LEN - 1

    steps_per_epoch = len(train_tokens) // (
        BATCH_SIZE * SEQ_LEN
    )

    if MAX_STEPS_PER_EPOCH is not None:

        steps_per_epoch = min(
            steps_per_epoch,
            MAX_STEPS_PER_EPOCH
        )

    print("\nExperiment configuration:")
    print("Modified module:", MODIFIED_MODULE_NAME)
    print("Modified args:", MODIFIED_ARGS)
    print("Sequence length:", SEQ_LEN)
    print("Batch size:", BATCH_SIZE)
    print("Model dimension:", D_MODEL)
    print("State dimension:", D_STATE)
    print("Epochs:", EPOCHS)
    print("Seeds:", SEEDS)
    print("Steps per epoch:", steps_per_epoch)
    print("Full validation:", MAX_VAL_BATCHES is None)

    all_results = []

    # Run paired experiments
    for seed in SEEDS:

        print("\n\n" + "#" * 80)
        print(f"STARTING SEED {seed}")
        print("#" * 80)

        # Generate identical training windows for both models.
        rng = np.random.default_rng(seed)

        batch_starts_by_epoch = []

        for epoch in range(EPOCHS):

            epoch_batches = []

            for step in range(steps_per_epoch):

                starts = rng.integers(
                    0,
                    max_start + 1,
                    size=BATCH_SIZE
                ).tolist()

                epoch_batches.append(starts)

            batch_starts_by_epoch.append(epoch_batches)

        # Original
        original_result = train_one(
            "Original Mamba",
            original_module,
            train_tokens,
            valid_batches,
            test_batches,
            vocab_size,
            device,
            batch_starts_by_epoch,
            seed
        )

        # Modified (model_developing_v2.py)
        modified_result = train_one(
            "Modified Mamba (state-dependent B, C)",
            modified_module,
            train_tokens,
            valid_batches,
            test_batches,
            vocab_size,
            device,
            batch_starts_by_epoch,
            seed,
            extra_args=MODIFIED_ARGS
        )

        all_results.append({
            "seed": seed,
            "original": original_result,
            "modified": modified_result
        })

        # Save intermediate results in case the Colab session ends.
        save_results(all_results)

        improvement = (
            original_result["test_ppl"]
            - modified_result["test_ppl"]
        ) / original_result["test_ppl"] * 100

        print(
            f"\nSeed {seed} improvement: {improvement:.2f}%"
        )

    # Final comparison
    statistical_analysis(all_results)

    print("\nExperiment completed.")
    print("Results file:", RESULTS_FILE)


if __name__ == "__main__":
    main()
