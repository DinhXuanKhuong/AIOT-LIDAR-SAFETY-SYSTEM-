import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os

class SimpleNuScenesDataset(Dataset):
    def __init__(self, data_root, scenes, sequence_length, prediction_horizon, split="train", data_augmentation=False):
        self.data_root = data_root
        self.meta_dir = os.path.join(data_root, "v0.1")
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
        self._load_metadata()
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

    def _get_category_name(self, instance_token):
        return self.instance_category_map.get(instance_token)

    def _estimate_velocity(self, instance_token, timestamp):
        instance = self.instance_map.get(instance_token)
        if not instance:
            return [0.0, 0.0]

        anns = self.instance_annotations.get(instance_token, [])
        ann_tokens = []
        for ann in anns:
            sample = self.sample_map.get(ann['sample_token'])
            if sample:
                ann_tokens.append((ann, sample['timestamp']))

        velocities = []
        for i, (ann, ann_timestamp) in enumerate(ann_tokens):
            if ann_timestamp == timestamp:
                if i > 0:
                    prev_ann, prev_timestamp = ann_tokens[i-1]
                    delta_time = (timestamp - prev_timestamp) / 1e6
                    if delta_time > 0.01:
                        pos = np.array(ann['translation'][:2])
                        prev_pos = np.array(prev_ann['translation'][:2])
                        velocity = (pos - prev_pos) / delta_time
                        velocities.append(velocity)
                        if len(velocities) > 1:
                            velocities[-1] = 0.7 * velocities[-1] + 0.3 * velocities[-2]
        return velocities[-1] if velocities else [0.0, 0.0]

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
                    vel = self._estimate_velocity(instance_token, timestamp)
                    rotation = ann['rotation']
                    w, x, y, z = rotation
                    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    state = list(pos) + list(vel) + [yaw]
                    if len(state) != 5:
                        print(f"Warning: State has incorrect length {len(state)} for instance {instance_token} at timestamp {timestamp}")
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

    def set_normalization_stats(self, pos_mean, pos_std, vel_mean, vel_std, heading_mean, heading_std):
        self.pos_mean = pos_mean
        self.pos_std = pos_std
        self.vel_mean = vel_mean
        self.vel_std = vel_std
        self.heading_mean = heading_mean
        self.heading_std = heading_std
        for i in range(len(self.sequences)):
            past, future = self.sequences[i]
            past[:, :2] = (past[:, :2] - self.pos_mean) / self.pos_std
            past[:, 2:4] = (past[:, 2:4] - self.vel_mean) / self.vel_std
            past[:, 4] = (past[:, 4] - self.heading_mean) / self.heading_std
            future = (future - self.vel_mean) / self.vel_std
            self.sequences[i] = (torch.from_numpy(past), torch.from_numpy(future))

    def denormalize(self, velocities):
        return velocities * self.vel_std + self.vel_mean

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        past, future = self.sequences[idx]
        if self.data_augmentation:
            noise_pos = torch.randn_like(past[:, :2]) * 0.2
            noise_vel = torch.randn_like(past[:, 2:4]) * 0.1
            past = past.clone()
            past[:, :2] += noise_pos
            past[:, 2:4] += noise_vel
        return past, future