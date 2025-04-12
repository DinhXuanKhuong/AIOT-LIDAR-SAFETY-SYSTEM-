import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob

class HybridCNNLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=1, prediction_horizon=6, dropout=0.3):
        super(HybridCNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # Compute CNN output size (assuming sequence_length=3)
        cnn_output_size = 32 * (3 // 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(cnn_output_size, hidden_size, num_layers, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 2 * prediction_horizon)

    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN expects input as (batch, channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        # CNN layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last timestep's output
        out = self.dropout_lstm(out[:, -1, :])
        
        # Fully connected layer
        out = self.fc(out)
        out = out.view(batch_size, self.prediction_horizon, 2)
        return out

########################################################################################################
class SimpleNuScenesDataset(Dataset):
    def __init__(self, data_root, scenes, sequence_length, prediction_horizon, split="train", data_augmentation=False):
        self.data_root = data_root
        self.meta_dir = os.path.join(data_root, "v1.0-trainval")
        self.scenes = scenes
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.split = split
        self.data_augmentation = data_augmentation and split == "train"
        self.sequences = []
        self.pos_mean = None
        self.pos_std = None
        self.vel_mean = None
        self.vel_std = None
        self.heading_mean = None
        self.heading_std = None
        self.dist_mean = None
        self.dist_std = None
        self.valid_scenes = []
        self.scene_sequence_counts = {}
        self._load_metadata()
        self._precompute_velocities()
        self._build_sequences()

    def _load_metadata(self):
        with open(os.path.join(self.meta_dir, "scene.json"), "r") as f:
            self.scene_data = json.load(f)
        with open(os.path.join(self.meta_dir, "sample.json"), "r") as f:
            self.sample_data = json.load(f)
        with open(os.path.join(self.meta_dir, "sample_annotation.json"), "r") as f:
            self.annotation_data = json.load(f)
        with open(os.path.join(self.meta_dir, "instance.json"), "r") as f:
            self.instance_data = json.load(f)
        with open(os.path.join(self.meta_dir, "category.json"), "r") as f:
            self.category_data = json.load(f)
        with open(os.path.join(self.meta_dir, "sample_data.json"), "r") as f:
            self.sample_data_metadata = json.load(f)

        self.category_map = {cat['token']: cat['name'] for cat in self.category_data}
        self.sample_map = {sample['token']: sample for sample in self.sample_data}
        self.annotation_map = {}
        for ann in self.annotation_data:
            sample_token = ann['sample_token']
            if sample_token not in self.annotation_map:
                self.annotation_map[sample_token] = []
            self.annotation_map[sample_token].append(ann)
        self.instance_map = {inst['token']: inst for inst in self.instance_data}
        self.instance_category_map = {}
        for inst in self.instance_data:
            category_token = inst.get('category_token')
            category_name = self.category_map.get(category_token) if category_token else None
            self.instance_category_map[inst['token']] = category_name
        self.instance_annotations = {}
        for ann in self.annotation_data:
            instance_token = ann['instance_token']
            if instance_token not in self.instance_annotations:
                self.instance_annotations[instance_token] = []
            self.instance_annotations[instance_token].append(ann)
        self.sample_data_map = {sd['token']: sd for sd in self.sample_data_metadata}

    def _verify_scene_data(self, scene):
        first_sample_token = scene['first_sample_token']
        sample = self.sample_map.get(first_sample_token)
        sample_tokens = set()
        while sample:
            sample_tokens.add(sample['token'])
            sample_token = sample['next']
            sample = self.sample_map.get(sample_token)
        
        for sd in self.sample_data_metadata:
            if sd['sample_token'] in sample_tokens and 'LIDAR_TOP' in sd['filename']:
                file_path = os.path.join(self.data_root, sd['filename'])
                if os.path.exists(file_path):
                    return True
        return False

    def _precompute_velocities(self):
        self.instance_velocities = {}
        for instance_token in self.instance_annotations:
            anns = self.instance_annotations.get(instance_token, [])
            ann_tokens = []
            for ann in anns:
                sample = self.sample_map.get(ann['sample_token'])
                if sample:
                    ann_tokens.append((ann, sample['timestamp']))
            ann_tokens = sorted(ann_tokens, key=lambda x: x[1])
            
            velocities = {}
            for i, (ann, timestamp) in enumerate(ann_tokens):
                if i > 0:
                    prev_ann, prev_timestamp = ann_tokens[i-1]
                    delta_time = (timestamp - prev_timestamp) / 1e6
                    if delta_time > 0.01:
                        pos = np.array(ann['translation'][:2])
                        prev_pos = np.array(prev_ann['translation'][:2])
                        velocity = (pos - prev_pos) / delta_time
                        if i > 1:
                            prev_velocity = velocities[ann_tokens[i-1][1]]
                            velocity = 0.7 * velocity + 0.3 * prev_velocity
                        velocities[timestamp] = velocity
            self.instance_velocities[instance_token] = velocities

    def _compute_relative_distance(self, ann, anns):
        ego_pos = np.array(ann['translation'][:2])
        min_dist = float('inf')
        for other_ann in anns:
            if other_ann['instance_token'] == ann['instance_token']:
                continue
            if self._get_category_name(other_ann['instance_token']) != 'vehicle.car':
                continue
            other_pos = np.array(other_ann['translation'][:2])
            dist = np.linalg.norm(ego_pos - other_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist if min_dist != float('inf') else 10.0

    def _build_sequences(self):
        for scene_name in self.scenes:
            scene = None
            for s in self.scene_data:
                if s['name'] == scene_name:
                    scene = s
                    break
            if scene is None:
                print(f"Scene {scene_name} not found in dataset. Skipping.")
                continue

            if not self._verify_scene_data(scene):
                print(f"Scene {scene_name} has no valid data files (e.g., LiDAR). Skipping.")
                continue

            self.valid_scenes.append(scene_name)
            self.scene_sequence_counts[scene_name] = 0
            first_sample_token = scene['first_sample_token']
            sample = self.sample_map.get(first_sample_token)
            instances = {}

            while sample:
                timestamp = sample['timestamp']
                anns = self.annotation_map.get(sample['token'], [])
                for ann in anns:
                    instance_token = ann['instance_token']
                    category_name = self._get_category_name(instance_token)
                    if category_name != 'vehicle.car':
                        continue
                    if instance_token not in instances:
                        instances[instance_token] = []
                    pos = ann['translation'][:2]
                    vel = self.instance_velocities.get(instance_token, {}).get(timestamp, [0.0, 0.0])
                    rotation = ann['rotation']
                    w, x, y, z = rotation
                    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    rel_dist = self._compute_relative_distance(ann, anns)
                    state = list(pos) + list(vel) + [yaw, rel_dist]
                    if len(state) != 6 or any(np.isnan(state)) or any(np.isinf(state)):
                        print(f"Warning: Invalid state for instance {instance_token} at timestamp {timestamp}")
                        continue
                    instances[instance_token].append((timestamp, state))
                sample_token = sample['next']
                sample = self.sample_map.get(sample_token)

            for instance_token, states in instances.items():
                states = sorted(states, key=lambda x: x[0])
                for i in range(len(states) - self.sequence_length - self.prediction_horizon + 1):
                    past = [state for _, state in states[i:i + self.sequence_length]]
                    future = [state[2:4] for _, state in states[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]]
                    if len(past) == self.sequence_length and len(future) == self.prediction_horizon:
                        past_array = np.array(past, dtype=np.float32)
                        future_array = np.array(future, dtype=np.float32)
                        self.sequences.append((past_array, future_array))
                        self.scene_sequence_counts[scene_name] += 1

    def _get_category_name(self, instance_token):
        return self.instance_category_map.get(instance_token)

    def set_normalization_stats(self, pos_mean, pos_std, vel_mean, vel_std, heading_mean, heading_std, dist_mean, dist_std):
        self.pos_mean = pos_mean
        self.pos_std = pos_std
        self.vel_mean = vel_mean
        self.vel_std = vel_std
        self.heading_mean = heading_mean
        self.heading_std = heading_std
        self.dist_mean = dist_mean
        self.dist_std = dist_std
        for i in range(len(self.sequences)):
            past, future = self.sequences[i]
            past[:, :2] = (past[:, :2] - self.pos_mean) / self.pos_std
            past[:, 2:4] = (past[:, 2:4] - self.vel_mean) / self.vel_std
            past[:, 4] = (past[:, 4] - self.heading_mean) / self.heading_std
            past[:, 5] = (past[:, 5] - self.dist_mean) / self.dist_std
            future = (future - self.vel_mean) / self.vel_std
            self.sequences[i] = (torch.from_numpy(past), torch.from_numpy(future))

    def denormalize(self, velocities):
        return velocities * self.vel_std + self.vel_mean

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        past, future = self.sequences[idx]
        if self.data_augmentation:
            past = past.clone()
            noise_pos = torch.randn_like(past[:, :2]) * 0.2
            noise_vel = torch.randn_like(past[:, 2:4]) * 0.1
            past[:, :2] += noise_pos
            past[:, 2:4] += noise_vel
            if torch.rand(1) < 0.5:
                past = torch.roll(past, shifts=1, dims=0)
                past[0] = past[1]
        return past, future

########################################################################################################


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