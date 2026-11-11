"""
One-file WikiText-2 language-modeling experiment for original vs modified Mamba.

What it does:
1. Installs missing lightweight Python dependencies (PyTorch must already be installed).
2. Clones the user's public repository if it is not already present.
3. Downloads WikiText-2 raw and the GPT-2 tokenizer.
4. Trains original and modified Mamba with the same data, hyperparameters,
   batch order, and evaluation procedure.
5. Reports validation/test perplexity, parameter counts, and training time.

Run in Colab:
    !python run_wikitext2.py

If your repository URL changes, edit REPO_URL below.
"""

import os
import sys
import time
import random
import subprocess
import importlib.util
from pathlib import Path

REPO_URL = "https://github.com/amirho3inz1/mamba-state-dependent.git"
REPO_DIR = Path("/content/mamba-state-dependent")
SEQ_LEN = 64
BATCH_SIZE = 8
D_MODEL = 32
N_LAYER = 1
D_STATE = 8
EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
SEED = 42
MAX_VAL_BATCHES = None  # None = evaluate the complete validation/test split


def install_if_missing(module_name, package_name=None):
    if importlib.util.find_spec(module_name) is None:
        package_name = package_name or module_name
        print(f"Installing {package_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_repository():
    if not REPO_DIR.exists():
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {REPO_URL} ...")
        subprocess.check_call(["git", "clone", REPO_URL, str(REPO_DIR)])
    else:
        print(f"Using existing repository: {REPO_DIR}")

    if not (REPO_DIR / "model.py").exists():
        raise FileNotFoundError(f"Could not find {REPO_DIR / 'model.py'}")
    if not (REPO_DIR / "model_modified.py").exists():
        raise FileNotFoundError(f"Could not find {REPO_DIR / 'model_modified.py'}")

    sys.path.insert(0, str(REPO_DIR))


def prepare_data():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import torch

    print("\nDownloading WikiText-2 raw ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def encode_split(name):
        # Keep the original line/document boundaries as newline tokens.
        text = "\n".join(ds[name]["text"])
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    train = encode_split("train")
    valid = encode_split("validation")
    test = encode_split("test")

    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Train tokens:    {len(train):,}")
    print(f"Validation:      {len(valid):,}")
    print(f"Test tokens:     {len(test):,}")
    return train, valid, test, tokenizer.vocab_size


def make_eval_batches(tokens, seq_len, batch_size):
    """
    Sequential, non-overlapping evaluation chunks. Each target is the
    next-token-shifted version of its input. Incomplete final chunks are skipped.
    """
    import torch
    usable = ((len(tokens) - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError("Not enough tokens for evaluation.")
    x = tokens[:usable].view(-1, seq_len)
    y = tokens[1:usable + 1].view(-1, seq_len)
    batches = []
    for i in range(0, x.size(0), batch_size):
        batches.append((x[i:i + batch_size], y[i:i + batch_size]))
    return batches


def build_lm(module, vocab_size):
    import torch
    import torch.nn as nn

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.args = module.ModelArgs(
                d_model=D_MODEL,
                n_layer=N_LAYER,
                vocab_size=vocab_size,
                d_state=D_STATE,
            )
            # ModelArgs may pad vocab_size to its required multiple.
            self.embedding = nn.Embedding(self.args.vocab_size, D_MODEL)
            self.blocks = nn.ModuleList(
                [module.ResidualBlock(self.args) for _ in range(N_LAYER)]
            )
            self.norm = nn.LayerNorm(D_MODEL)
            self.lm_head = nn.Linear(D_MODEL, vocab_size, bias=False)

        def forward(self, token_ids):
            x = self.embedding(token_ids)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return self.lm_head(x)

    return LanguageModel()


def evaluate(model, batches, device, max_batches=None):
    import torch
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_index, (x, y) in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if logits.ndim != 3 or logits.shape[:2] != y.shape:
                raise RuntimeError(
                    "Unexpected model output shape. Expected [batch, seq_len, vocab]. "
                    f"Got {tuple(logits.shape)} for targets {tuple(y.shape)}."
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    mean_loss = total_loss / total_tokens
    perplexity = float(torch.exp(torch.tensor(mean_loss)))
    return mean_loss, perplexity


def train_one(name, module, train_tokens, valid_batches, test_batches,
              vocab_size, device, batch_starts_by_epoch):
    import torch
    import torch.nn.functional as F

    print(f"\n{'=' * 68}\nTraining: {name}\n{'=' * 68}")
    set_seed(SEED)
    model = build_lm(module, vocab_size).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Smoke test catches incompatible model APIs before a long training run.
    sample_x = train_tokens[:SEQ_LEN].unsqueeze(0).to(device)
    with torch.no_grad():
        sample_logits = model(sample_x)
    expected_shape = (1, SEQ_LEN, vocab_size)
    if tuple(sample_logits.shape) != expected_shape:
        raise RuntimeError(
            f"{name}: expected output shape {expected_shape}, "
            f"got {tuple(sample_logits.shape)}."
        )
    print(f"Parameters: {n_params:,}")
    print(f"Smoke test output: {tuple(sample_logits.shape)}")

    best_val_ppl = float("inf")
    best_state = None
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for starts in batch_starts_by_epoch[epoch - 1]:
            x = torch.stack([train_tokens[s:s + SEQ_LEN] for s in starts])
            y = torch.stack([train_tokens[s + 1:s + SEQ_LEN + 1] for s in starts])
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

        train_loss = total_loss / total_tokens
        train_ppl = float(torch.exp(torch.tensor(train_loss)))
        val_loss, val_ppl = evaluate(
            model, valid_batches, device, MAX_VAL_BATCHES
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | train PPL {train_ppl:.2f} | "
            f"val loss {val_loss:.4f} | val PPL {val_ppl:.2f}"
        )

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            # Keep the best validation checkpoint in memory.
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    elapsed = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_ppl = evaluate(model, valid_batches, device)
    test_loss, test_ppl = evaluate(model, test_batches, device)

    print(
        f"\n{name} best-checkpoint results:\n"
        f"  Validation perplexity: {final_val_ppl:.2f}\n"
        f"  Test perplexity:       {test_ppl:.2f}\n"
        f"  Parameters:            {n_params:,}\n"
        f"  Training time:         {elapsed / 60:.2f} minutes"
    )

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "val_ppl": final_val_ppl,
        "test_ppl": test_ppl,
        "params": n_params,
        "minutes": elapsed / 60,
    }


def main():
    install_if_missing("einops")
    install_if_missing("datasets")
    install_if_missing("transformers")

    import torch
    import numpy as np

    prepare_repository()
    import model as original_module
    import model_modified as modified_module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected. The modified model may run slowly.")

    train_tokens, val_tokens, test_tokens, vocab_size = prepare_data()

    if len(train_tokens) <= SEQ_LEN + 1:
        raise ValueError("Training split is too short.")

    valid_batches = make_eval_batches(val_tokens, SEQ_LEN, BATCH_SIZE)
    test_batches = make_eval_batches(test_tokens, SEQ_LEN, BATCH_SIZE)

    # Precompute exactly the same random training windows for both models.
    # This prevents differences in batch order from biasing the comparison.
    rng = np.random.default_rng(SEED)
    max_start = len(train_tokens) - SEQ_LEN - 1
    steps_per_epoch = max(1, len(train_tokens) // (BATCH_SIZE * SEQ_LEN))
    batch_starts_by_epoch = []
    for _ in range(EPOCHS):
        epoch_batches = []
        for _ in range(steps_per_epoch):
            starts = rng.integers(0, max_start + 1, size=BATCH_SIZE).tolist()
            epoch_batches.append(starts)
        batch_starts_by_epoch.append(epoch_batches)

    original_result = train_one(
        "Original Mamba", original_module, train_tokens, valid_batches,
        test_batches, vocab_size, device, batch_starts_by_epoch
    )
    modified_result = train_one(
        "Modified Mamba", modified_module, train_tokens, valid_batches,
        test_batches, vocab_size, device, batch_starts_by_epoch
    )

    print("\n\n" + "=" * 76)
    print("FINAL COMPARISON — WIKITEXT-2")
    print("=" * 76)
    print(f"{'Metric':<24}{'Original':>18}{'Modified':>18}")
    print("-" * 60)
    print(
        f"{'Validation PPL':<24}"
        f"{original_result['val_ppl']:>18.2f}"
        f"{modified_result['val_ppl']:>18.2f}"
    )
    print(
        f"{'Test PPL':<24}"
        f"{original_result['test_ppl']:>18.2f}"
        f"{modified_result['test_ppl']:>18.2f}"
    )
    print(
        f"{'Parameters':<24}"
        f"{original_result['params']:>18,}"
        f"{modified_result['params']:>18,}"
    )
    print(
        f"{'Training time (min)':<24}"
        f"{original_result['minutes']:>18.2f}"
        f"{modified_result['minutes']:>18.2f}"
    )

    delta = (
        (original_result["test_ppl"] - modified_result["test_ppl"])
        / original_result["test_ppl"]
    ) * 100.0
    print("-" * 60)
    if delta > 0:
        print(f"Modified model test perplexity is {delta:.2f}% lower.")
    else:
        print(f"Modified model test perplexity is {abs(delta):.2f}% higher.")
    print("\nLower perplexity is better. One run is preliminary; repeat with")
    print("multiple seeds before drawing a research conclusion.")


if __name__ == "__main__":
    main()
