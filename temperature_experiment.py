
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import model as original
import model_modified as modified


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

SEQ_LEN = 12
NUM_TRAJECTORIES = 600
TRAJECTORY_LENGTH = 35

D_MODEL = 8
D_STATE = 4
N_LAYER = 1

BATCH_SIZE = 128
EPOCHS = 12
LEARNING_RATE = 1e-3

DEVICE = torch.device("cpu")

torch.set_num_threads(2)


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 3. Generate temperature data
# ============================================================

def generate_trajectories():

    trajectories = []

    for _ in range(NUM_TRAJECTORIES):

        # Random initial temperature
        temperature = np.random.uniform(18, 32)

        # Each trajectory has a heating or cooling trend
        direction = np.random.choice([-1, 1])
        slope = direction * np.random.uniform(0.04, 0.20)

        true_values = [temperature]

        for t in range(TRAJECTORY_LENGTH - 1):

            process_noise = np.random.normal(0, 0.025)

            temperature = temperature + slope + process_noise

            true_values.append(temperature)

        true_values = np.array(true_values)

        # Add measurement noise
        measured_values = (
            true_values + np.random.normal(0, 0.08, len(true_values))
        )

        trajectories.append((measured_values, true_values))

    return trajectories


def create_dataset(trajectories):

    X = []
    Y = []

    for measured, true_values in trajectories:

        for i in range(len(measured) - SEQ_LEN):

            # Input: previous temperature measurements
            x = measured[i:i + SEQ_LEN]

            # Target: next true temperature
            y = true_values[i + SEQ_LEN]

            X.append(x)
            Y.append(y)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # Normalize temperatures
    X = (X - 25.0) / 10.0
    Y = (Y - 25.0) / 10.0

    X = torch.tensor(X).unsqueeze(-1)
    Y = torch.tensor(Y)

    return X, Y


def prepare_data():

    trajectories = generate_trajectories()

    # Split by complete trajectories to avoid data leakage
    indices = np.random.permutation(len(trajectories))

    n_train = int(0.70 * len(indices))
    n_val = int(0.15 * len(indices))

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_trajectories = [trajectories[i] for i in train_indices]
    val_trajectories = [trajectories[i] for i in val_indices]
    test_trajectories = [trajectories[i] for i in test_indices]

    X_train, Y_train = create_dataset(train_trajectories)
    X_val, Y_val = create_dataset(val_trajectories)
    X_test, Y_test = create_dataset(test_trajectories)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=BATCH_SIZE
    )

    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=BATCH_SIZE
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 4. Continuous-input Mamba forecasting model
# ============================================================

class TemperatureMamba(nn.Module):

    def __init__(self, variant):

        super().__init__()

        args = original.ModelArgs(
            d_model=D_MODEL,
            n_layer=N_LAYER,
            vocab_size=8,
            d_state=D_STATE,
            expand=2,
            dt_rank="auto",
            d_conv=3
        )

        self.input_projection = nn.Linear(1, D_MODEL)

        if variant == "original":

            block_class = original.ResidualBlock
            norm_class = original.RMSNorm

        elif variant == "modified":

            block_class = modified.ResidualBlock
            norm_class = modified.RMSNorm

        else:

            raise ValueError("Unknown model variant")

        self.layers = nn.ModuleList([
            block_class(args) for _ in range(N_LAYER)
        ])

        self.norm = norm_class(D_MODEL)

        self.output_projection = nn.Linear(D_MODEL, 1)

    def forward(self, x):

        # Continuous temperature input
        x = self.input_projection(x)

        # Mamba residual blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Predict the next temperature using the final position
        x = x[:, -1, :]

        output = self.output_projection(x)

        return output.squeeze(-1)


# ============================================================
# 5. Evaluation
# ============================================================

def evaluate(model, loader):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            predictions.append(pred.cpu())
            targets.append(y.cpu())

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    # Convert normalized values back to degrees Celsius
    predictions = predictions * 10.0 + 25.0
    targets = targets * 10.0 + 25.0

    mae = torch.mean(torch.abs(predictions - targets)).item()

    rmse = torch.sqrt(
        torch.mean((predictions - targets) ** 2)
    ).item()

    return mae, rmse


# ============================================================
# 6. Training
# ============================================================

def train_model(variant, train_loader, val_loader, test_loader):

    print("\n" + "=" * 60)
    print("Training:", variant.upper())
    print("=" * 60)

    set_seed(SEED)

    model = TemperatureMamba(variant).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(x)

            loss = criterion(prediction, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_mae, val_rmse = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val MAE: {val_mae:.4f} C"
        )

        if val_mae < best_val_loss:

            best_val_loss = val_mae

            best_weights = copy.deepcopy(model.state_dict())

    training_time = time.perf_counter() - start_time

    # Restore the best validation model
    model.load_state_dict(best_weights)

    test_mae, test_rmse = evaluate(model, test_loader)

    num_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("\nResults for", variant.upper())

    print(f"Test MAE:       {test_mae:.5f} C")
    print(f"Test RMSE:      {test_rmse:.5f} C")
    print(f"Parameters:     {num_parameters}")
    print(f"Training time:  {training_time:.2f} seconds")

    return {
        "model": variant,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "parameters": num_parameters,
        "training_time": training_time
    }


# ============================================================
# 7. Main experiment
# ============================================================

def main():

    set_seed(SEED)

    print("Preparing temperature dataset...")

    train_loader, val_loader, test_loader = prepare_data()

    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    results = []

    # Original Mamba
    result_original = train_model(
        "original",
        train_loader,
        val_loader,
        test_loader
    )

    results.append(result_original)

    # Modified Mamba
    result_modified = train_model(
        "modified",
        train_loader,
        val_loader,
        test_loader
    )

    results.append(result_modified)

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print(
        f"{'Model':<15}"
        f"{'MAE (C)':<15}"
        f"{'RMSE (C)':<15}"
        f"{'Parameters':<15}"
        f"{'Time (s)':<10}"
    )

    for r in results:

        print(
            f"{r['model']:<15}"
            f"{r['test_mae']:<15.5f}"
            f"{r['test_rmse']:<15.5f}"
            f"{r['parameters']:<15}"
            f"{r['training_time']:<10.2f}"
        )


if __name__ == "__main__":

    main()
