import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os


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