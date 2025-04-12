import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
from Hybrid_CNN_LSTM import HybridCNNLSTM
from tqdm import tqdm
from SimpleNuScenesDataset import SimpleNuScenesDataset
import glob




def train_model(model, train_loader, val_loader, device, epochs=100, patience=10, resume=False, checkpoint_path="best_lstm_model.pth"):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 0

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        print(f"Resumed training from epoch {start_epoch}, best validation loss: {best_val_loss:.4f}")
    elif resume:
        print(f"No checkpoint found at {checkpoint_path}. Starting fresh training.")

    for epoch in range(start_epoch, epochs):
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

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }, checkpoint_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
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

def verify_scenes(data_root, scenes, sample_data_metadata, sample_map):
    valid_scenes = []
    scene_to_samples = {}
    for scene in scenes:
        scene_to_samples[scene['name']] = set()
        sample_token = scene['first_sample_token']
        while sample_token:
            scene_to_samples[scene['name']].add(sample_token)
            sample = sample_map.get(sample_token, {})
            sample_token = sample.get('next')
    
    lidar_files = set(os.path.basename(f) for f in glob.glob(os.path.join(data_root, "samples/LIDAR_TOP/*.bin")))
    for scene_name, sample_tokens in scene_to_samples.items():
        for sd in sample_data_metadata:
            if sd['sample_token'] in sample_tokens and 'LIDAR_TOP' in sd['filename']:
                if os.path.basename(sd['filename']) in lidar_files:
                    valid_scenes.append(scene_name)
                    break

    # Save valid scenes to a JSON file
    valid_scenes_file = os.path.join(data_root, "valid_scenes.json")
    with open(valid_scenes_file, "w") as f:
        json.dump({"valid_scenes": valid_scenes}, f, indent=4)
    print(f"Saved valid scenes to {valid_scenes_file}")
    

    return valid_scenes



def main(resume_training=False):
    DATA_ROOT = "/content/nuscenes"
    SEQUENCE_LENGTH = 3
    PREDICTION_HORIZON = 6
    BATCH_SIZE = 32
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = "best_lstm_model.pth"

    meta_dir = os.path.join(DATA_ROOT, "v1.0-trainval")
    with open(os.path.join(meta_dir, "scene.json"), "r") as f:
        scene_data = json.load(f)
    with open(os.path.join(meta_dir, "sample.json"), "r") as f:
        sample_data = json.load(f)
    with open(os.path.join(meta_dir, "sample_data.json"), "r") as f:
        sample_data_metadata = json.load(f)

    sample_map = {sample['token']: sample for sample in sample_data}
    all_scenes = [scene for scene in scene_data]
    print(f"Total scenes: {len(all_scenes)}")

    #valid_scenes = verify_scenes(DATA_ROOT, all_scenes, sample_data_metadata, sample_map)
    #print(f"Valid scenes with data: {len(valid_scenes)}")
    #print("Valid scenes:", valid_scenes)
    # Check for existing valid_scenes.json
    valid_scenes_file = os.path.join(DATA_ROOT, "valid_scenes.json")
    if os.path.exists(valid_scenes_file):
        try:
            with open(valid_scenes_file, "r") as f:
                valid_scenes = json.load(f)["valid_scenes"]
            print(f"Loaded valid scenes from {valid_scenes_file}")
        except Exception as e:
            print(f"Error reading {valid_scenes_file}: {e}. Running scene verification.")
            valid_scenes = verify_scenes(DATA_ROOT, all_scenes, sample_data_metadata, sample_map)
    else:
        print(f"No valid_scenes.json found. Running scene verification.")
        valid_scenes = verify_scenes(DATA_ROOT, all_scenes, sample_data_metadata, sample_map)

    print(f"Valid scenes with data: {len(valid_scenes)}")
    print("Valid scenes:", valid_scenes)


    if not valid_scenes:
        print("No valid scenes found with data. Exiting.")
        return

    train_size = int(0.7 * len(valid_scenes))
    train_scenes = valid_scenes[:train_size]
    val_scenes = valid_scenes[train_size:]
    print(f"Training scenes: {train_scenes}")
    print(f"Validation scenes: {val_scenes}")

    train_dataset = SimpleNuScenesDataset(DATA_ROOT, train_scenes, SEQUENCE_LENGTH, PREDICTION_HORIZON, split="train", data_augmentation=True)
    val_dataset = SimpleNuScenesDataset(DATA_ROOT, val_scenes, SEQUENCE_LENGTH, PREDICTION_HORIZON, split="val", data_augmentation=False)

    for dataset, split in [(train_dataset, "train"), (val_dataset, "val")]:
        print(f"\nSequences per scene in {split} split:")
        for scene_name in dataset.valid_scenes:
            count = dataset.scene_sequence_counts.get(scene_name, 0)
            print(f"Scene {scene_name}: {count} sequences")

    all_sequences = train_dataset.sequences + val_dataset.sequences
    if all_sequences:
        all_past = np.stack([seq[0] for seq in all_sequences], axis=0)
        all_future = np.stack([seq[1] for seq in all_sequences], axis=0)
        positions = all_past[:, :, :2].reshape(-1, 2)
        velocities_past = all_past[:, :, 2:4].reshape(-1, 2)
        headings = all_past[:, :, 4].reshape(-1)
        distances = all_past[:, :, 5].reshape(-1)
        velocities_future = all_future.reshape(-1, 2)
        velocities = np.concatenate([velocities_past, velocities_future], axis=0)
        pos_mean = positions.mean(axis=0)
        pos_std = positions.std(axis=0) + 1e-6
        vel_mean = velocities.mean(axis=0)
        vel_std = velocities.std(axis=0) + 1e-6
        heading_mean = headings.mean()
        heading_std = headings.std() + 1e-6
        dist_mean = distances.mean()
        dist_std = distances.std() + 1e-6
    else:
        pos_mean = np.array([0.0, 0.0])
        pos_std = np.array([1.0, 1.0])
        vel_mean = np.array([0.0, 0.0])
        vel_std = np.array([1.0, 1.0])
        heading_mean = 0.0
        heading_std = 1.0
        dist_mean = 0.0
        dist_std = 1.0

    train_dataset.set_normalization_stats(pos_mean, pos_std, vel_mean, vel_std, heading_mean, heading_std, dist_mean, dist_std)
    val_dataset.set_normalization_stats(pos_mean, pos_std, vel_mean, vel_std, heading_mean, heading_std, dist_mean, dist_std)

    normalization_stats = {
        "pos_mean": pos_mean.tolist(),
        "pos_std": pos_std.tolist(),
        "vel_mean": vel_mean.tolist(),
        "vel_std": vel_std.tolist(),
        "heading_mean": float(heading_mean),
        "heading_std": float(heading_std),
        "dist_mean": float(dist_mean),
        "dist_std": float(dist_std)
    }
    with open(os.path.join(DATA_ROOT, "normalization_stats.json"), "w") as f:
        json.dump(normalization_stats, f)

    print(f"Number of training sequences: {len(train_dataset)}")
    print(f"Number of validation sequences: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridCNNLSTM(input_size=6, hidden_size=32, num_layers=1, prediction_horizon=PREDICTION_HORIZON, dropout=0.3).to(DEVICE)

    train_losses, val_losses = train_model(model, train_loader, val_loader, DEVICE, epochs=EPOCHS, patience=10, resume=resume_training, checkpoint_path=CHECKPOINT_PATH)
    plot_losses(train_losses, val_losses)
    visualize_prediction(model, val_dataset, sample_idx=0, device=DEVICE)

if __name__ == "__main__":
    main(resume_training=False)