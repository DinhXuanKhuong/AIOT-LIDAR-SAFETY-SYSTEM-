import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import os

# --- Dataset ---
class NuScenesMotionDataset(Dataset):
    def __init__(self, nusc, scenes, sequence_length, prediction_horizon):
        self.nusc = nusc
        self.scenes = scenes  # List of scene names (e.g., 'scene-0061')
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.sequences = []
        # For normalization
        self.pos_mean = None
        self.pos_std = None
        self.vel_mean = None
        self.vel_std = None
        self._build_sequences()
        self._compute_normalization_stats()

    def _build_sequences(self):
        for scene_name in self.scenes:
            # Find the scene token by matching the scene name
            scene = None
            for s in self.nusc.scene:
                if s['name'] == scene_name:
                    scene = s
                    break
            if scene is None:
                print(f"Scene {scene_name} not found in dataset. Skipping.")
                continue

            # Process each sample in the scene
            first_sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', first_sample_token)
            instances = {}  # instance_token -> list of (timestamp, [x, y, vx, vy])

            while sample:
                timestamp = sample['timestamp']
                anns = [self.nusc.get('sample_annotation', ann_token) for ann_token in sample['anns']]
                for ann in anns:
                    instance_token = ann['instance_token']
                    if instance_token not in instances:
                        instances[instance_token] = []
                    pos = ann['translation'][:2]  # [x, y]
                    vel = self._estimate_velocity(instance_token, timestamp)
                    state = list(pos) + list(vel)  # Ensure [x, y, vx, vy]
                    if len(state) != 4:
                        print(f"Warning: State has incorrect length {len(state)} for instance {instance_token} at timestamp {timestamp}")
                        continue
                    instances[instance_token].append((timestamp, state))
                sample = self.nusc.get('sample', sample['next']) if sample['next'] else None

            # Build sequences for each instance
            for instance_token, states in instances.items():
                states = sorted(states, key=lambda x: x[0])  # Sort by timestamp
                for i in range(len(states) - self.sequence_length - self.prediction_horizon + 1):
                    past = [state for _, state in states[i:i + self.sequence_length]]
                    future = [state[2:] for _, state in states[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]]
                    if len(past) == self.sequence_length and len(future) == self.prediction_horizon:
                        past_array = np.array(past, dtype=np.float32)  # Shape: (sequence_length, 4)
                        future_array = np.array(future, dtype=np.float32)  # Shape: (prediction_horizon, 2)
                        self.sequences.append((past_array, future_array))

    def _compute_normalization_stats(self):
        if not self.sequences:
            return
        # Stack all past and future data to compute mean and std
        all_past = np.stack([seq[0] for seq in self.sequences], axis=0)  # Shape: (num_sequences, sequence_length, 4)
        all_future = np.stack([seq[1] for seq in self.sequences], axis=0)  # Shape: (num_sequences, prediction_horizon, 2)
        # Positions: [x, y] (first two elements of past)
        positions = all_past[:, :, :2].reshape(-1, 2)  # Shape: (num_sequences * sequence_length, 2)
        # Velocities: [vx, vy] (last two elements of past, and all of future)
        velocities_past = all_past[:, :, 2:].reshape(-1, 2)  # Shape: (num_sequences * sequence_length, 2)
        velocities_future = all_future.reshape(-1, 2)  # Shape: (num_sequences * prediction_horizon, 2)
        velocities = np.concatenate([velocities_past, velocities_future], axis=0)
        # Compute mean and std
        self.pos_mean = positions.mean(axis=0)  # Shape: (2,)
        self.pos_std = positions.std(axis=0) + 1e-6  # Shape: (2,), avoid division by zero
        self.vel_mean = velocities.mean(axis=0)  # Shape: (2,)
        self.vel_std = velocities.std(axis=0) + 1e-6  # Shape: (2,)
        # Normalize the sequences
        for i in range(len(self.sequences)):
            past, future = self.sequences[i]
            # Normalize positions in past
            past[:, :2] = (past[:, :2] - self.pos_mean) / self.pos_std
            # Normalize velocities in past and future
            past[:, 2:] = (past[:, 2:] - self.vel_mean) / self.vel_std
            future = (future - self.vel_mean) / self.vel_std
            self.sequences[i] = (torch.from_numpy(past), torch.from_numpy(future))

    def denormalize(self, velocities):
        # Denormalize velocities for visualization
        return velocities * self.vel_std + self.vel_mean

    def _estimate_velocity(self, instance_token, timestamp):
        instance = self.nusc.get('instance', instance_token)
        ann_tokens = []
        current_ann_token = instance['first_annotation_token']
        while current_ann_token:
            ann = self.nusc.get('sample_annotation', current_ann_token)
            ann_tokens.append(current_ann_token)
            current_ann_token = ann['next'] if ann['next'] else None

        velocities = []
        for i, ann_token in enumerate(ann_tokens):
            ann = self.nusc.get('sample_annotation', ann_token)
            sample = self.nusc.get('sample', ann['sample_token'])
            if sample['timestamp'] == timestamp:
                if i > 0:
                    prev_ann = self.nusc.get('sample_annotation', ann_tokens[i-1])
                    prev_sample = self.nusc.get('sample', prev_ann['sample_token'])
                    delta_time = (timestamp - prev_sample['timestamp']) / 1e6  # Convert microseconds to seconds
                    if delta_time > 0:
                        pos = np.array(ann['translation'][:2])
                        prev_pos = np.array(prev_ann['translation'][:2])
                        velocity = (pos - prev_pos) / delta_time
                        velocities.append(velocity)
        return velocities[-1] if velocities else [0.0, 0.0]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# --- Model ---
class MotionPredictionLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=3, prediction_horizon=2):
        super(MotionPredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon  # Store prediction_horizon
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2 * prediction_horizon)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(batch_size, self.prediction_horizon, 2)  # Use stored prediction_horizon
        return out

# --- Training ---
def train_model(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for past, future in train_loader:
            past, future = past.to(DEVICE), future.to(DEVICE)
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
                past, future = past.to(DEVICE), future.to(DEVICE)
                output = model(past)
                loss = criterion(output, future)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_motion_prediction_model.pth")

    return train_losses, val_losses

# --- Visualization ---
def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

def visualize_prediction(model, dataset, sample_idx=0):
    if len(dataset) == 0:
        print("Validation dataset is empty. Skipping visualization.")
        return
    model.eval()
    past, future = dataset[sample_idx]
    past = past.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(past).cpu().numpy().squeeze(0)
    past = past.cpu().numpy().squeeze(0)
    future = future.numpy()
    # Denormalize for visualization
    pred = dataset.denormalize(pred)
    future = dataset.denormalize(future)

    plt.figure()
    past_pos = np.zeros((len(past), 2))
    true_pos = np.zeros((len(future), 2))
    pred_pos = np.zeros((len(pred), 2))
    last_pos = past[-1, :2] * dataset.pos_std + dataset.pos_mean  # Denormalize past positions
    for i in range(len(past)):
        past_pos[i] = past[i, :2] * dataset.pos_std + dataset.pos_mean
    for i in range(len(future)):
        true_pos[i] = last_pos + future[i] * 0.5  # 0.5s per timestep
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

# --- Main ---
def main():
    # Dataset parameters
    DATA_ROOT = "D:/HACKATHON/wires_boxes/v1.0-mini"
    VERSION = "v1.0-mini"
    SEQUENCE_LENGTH = 3
    PREDICTION_HORIZON = 2
    BATCH_SIZE = 32
    global EPOCHS
    EPOCHS = 50
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load nuScenes dataset
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)

    # Get all scene names
    all_scenes = [scene['name'] for scene in nusc.scene]
    print(f"Total scenes in {VERSION}: {len(all_scenes)}")
    print("Scenes:", all_scenes)

    # Split scenes into train and validation (80-20 split)
    train_size = int(0.8 * len(all_scenes))
    train_scenes = all_scenes[:train_size]
    val_scenes = all_scenes[train_size:]
    print(f"Training scenes: {train_scenes}")
    print(f"Validation scenes: {val_scenes}")

    # Initialize datasets
    train_dataset = NuScenesMotionDataset(nusc, train_scenes, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    val_dataset = NuScenesMotionDataset(nusc, val_scenes, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    print(f"Number of training sequences: {len(train_dataset)}")
    print(f"Number of validation sequences: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = MotionPredictionLSTM(input_size=4, hidden_size=128, num_layers=3, prediction_horizon=PREDICTION_HORIZON).to(DEVICE)

    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    plot_losses(train_losses, val_losses)
    visualize_prediction(model, val_dataset, sample_idx=0)

if __name__ == "__main__":
    main()