import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
from LSTM_Model import SimpleLSTM
from SimpleNuScenesDataset import SimpleNuScenesDataset

def train_model(model, train_loader, val_loader, device, epochs=100, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for past, future in train_loader:
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            output = model(past)
            loss = criterion(output, future)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for past, future in val_loader:
                past, future = past.to(device), future.to(device)
                output = model(past)
                loss = criterion(output, future)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, Epochs No Improve: {epochs_no_improve}")

        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_lstm_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step(val_loss)

    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

def visualize_prediction(model, dataset, sample_idx=0, device="cpu"):
    if len(dataset) == 0:
        print("Validation dataset is empty. Skipping visualization.")
        return
    model.eval()
    past, future = dataset[sample_idx]
    past = past.unsqueeze(0).to(device)
    future = future.numpy()
    with torch.no_grad():
        pred = model(past).cpu().numpy().squeeze(0)
    pred = dataset.denormalize(pred)
    future = dataset.denormalize(future)

    plt.figure()
    past_pos = np.zeros((len(past), 2))
    true_pos = np.zeros((len(future), 2))
    pred_pos = np.zeros((len(pred), 2))
    past = past.cpu().numpy().squeeze(0)
    past_pos = past[:, :2] * dataset.pos_std + dataset.pos_mean
    last_pos = past_pos[-1]
    for i in range(len(future)):
        true_pos[i] = last_pos + future[i] * 0.5
        pred_pos[i] = last_pos + pred[i] * 0.5
        last_pos = true_pos[i]

    plt.plot(past_pos[:, 0], past_pos[:, 1], 'b-', label="Past")
    plt.plot(true_pos[:, 0], true_pos[:, 1], 'g-', label="True Future")
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', label="Predicted Future")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.savefig("prediction_visualization.png")
    plt.close()

def main():
    DATA_ROOT = "D:/HACKATHON/wires_boxes"
    SEQUENCE_LENGTH = 3
    PREDICTION_HORIZON = 6
    BATCH_SIZE = 32
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(DATA_ROOT, "v0.1", "scene.json"), "r") as f:
        scene_data = json.load(f)
    all_scenes = [scene['name'] for scene in scene_data]
    print(f"Total scenes: {len(all_scenes)}")
    print("Scenes:", all_scenes)

    train_size = int(0.7 * len(all_scenes))
    train_scenes = all_scenes[:train_size]
    val_scenes = all_scenes[train_size:]
    print(f"Training scenes: {train_scenes}")
    print(f"Validation scenes: {val_scenes}")

    train_dataset = SimpleNuScenesDataset(DATA_ROOT, train_scenes, SEQUENCE_LENGTH, PREDICTION_HORIZON, split="train", data_augmentation=True)
    val_dataset = SimpleNuScenesDataset(DATA_ROOT, val_scenes, SEQUENCE_LENGTH, PREDICTION_HORIZON, split="val", data_augmentation=False)

    for dataset, split in [(train_dataset, "train"), (val_dataset, "val")]:
        print(f"\nSequences per scene in {split} split:")
        scene_sequences = {}
        for scene_name in dataset.scenes:
            temp_dataset = SimpleNuScenesDataset(DATA_ROOT, [scene_name], SEQUENCE_LENGTH, PREDICTION_HORIZON, split=split)
            scene_sequences[scene_name] = len(temp_dataset.sequences)
        for scene_name, count in scene_sequences.items():
            print(f"Scene {scene_name}: {count} sequences")

    all_sequences = train_dataset.sequences + val_dataset.sequences
    if all_sequences:
        all_past = np.stack([seq[0] for seq in all_sequences], axis=0)
        all_future = np.stack([seq[1] for seq in all_sequences], axis=0)
        positions = all_past[:, :, :2].reshape(-1, 2)
        velocities_past = all_past[:, :, 2:4].reshape(-1, 2)
        headings = all_past[:, :, 4].reshape(-1)
        velocities_future = all_future.reshape(-1, 2)
        velocities = np.concatenate([velocities_past, velocities_future], axis=0)
        pos_mean = positions.mean(axis=0)
        pos_std = positions.std(axis=0) + 1e-6
        vel_mean = velocities.mean(axis=0)
        vel_std = velocities.std(axis=0) + 1e-6
        heading_mean = headings.mean()
        heading_std = headings.std() + 1e-6
    else:
        pos_mean = np.array([0.0, 0.0])
        pos_std = np.array([1.0, 1.0])
        vel_mean = np.array([0.0, 0.0])
        vel_std = np.array([1.0, 1.0])
        heading_mean = 0.0
        heading_std = 1.0

    train_dataset.set_normalization_stats(pos_mean, pos_std, vel_mean, vel_std, heading_mean, heading_std)
    val_dataset.set_normalization_stats(pos_mean, pos_std, vel_mean, vel_std, heading_mean, heading_std)

    normalization_stats = {
        "pos_mean": pos_mean.tolist(),
        "pos_std": pos_std.tolist(),
        "vel_mean": vel_mean.tolist(),
        "vel_std": vel_std.tolist(),
        "heading_mean": float(heading_mean),
        "heading_std": float(heading_std)
    }
    with open(os.path.join(DATA_ROOT, "normalization_stats.json"), "w") as f:
        json.dump(normalization_stats, f)

    print(f"Number of training sequences: {len(train_dataset)}")
    print(f"Number of validation sequences: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleLSTM(input_size=5, hidden_size=32, num_layers=1, prediction_horizon=PREDICTION_HORIZON, dropout=0.3).to(DEVICE)

    train_losses, val_losses = train_model(model, train_loader, val_loader, DEVICE, epochs=EPOCHS, patience=10)
    plot_losses(train_losses, val_losses)
    visualize_prediction(model, val_dataset, sample_idx=0, device=DEVICE)

if __name__ == "__main__":
    main()