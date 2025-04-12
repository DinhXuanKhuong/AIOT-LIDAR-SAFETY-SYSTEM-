import sys
import numpy as np
import tkinter as tk
import open3d as o3d
from sklearn.cluster import DBSCAN
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pygame
import os
import time
import glob
from bisect import bisect_left
from Hybrid_CNN_LSTM import HybridCNNLSTM  # Updated import
import torch
import json
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


# Load pretrained model
model = HybridCNNLSTM(input_size=6, hidden_size=32, num_layers=1, prediction_horizon=6, dropout=0.3)
model_path = "best_lstm_model_new.pth"
try:
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
except RuntimeError as e:
    print(f"Error loading model: {e}")
    sys.exit(1)
model.eval()

# Load normalization stats
with open("normalization_stats_new.json", "r") as f:
    norm_stats = json.load(f)

def prepare_lstm_input(past_seq, stats):
    past_seq = np.array(past_seq)
    past_seq[:, :2] = (past_seq[:, :2] - stats["pos_mean"]) / stats["pos_std"]
    past_seq[:, 2:4] = (past_seq[:, 2:4] - stats["vel_mean"]) / stats["vel_std"]
    past_seq[:, 4] = (past_seq[:, 4] - stats["heading_mean"]) / stats["heading_std"]
    past_seq[:, 5] = (past_seq[:, 5] - stats["dist_mean"]) / stats["dist_std"]
    return torch.tensor(past_seq, dtype=torch.float32).unsqueeze(0)

obstacle_histories = {}

def compute_relative_distance(centroid, obstacles):
    """Compute distance to nearest other obstacle."""
    min_dist = float('inf')
    for other_id, other_obs in obstacles.items():
        other_centroid = other_obs["centroid"]
        dist = np.linalg.norm(centroid[:2] - other_centroid[:2])
        if dist < min_dist and dist > 0:
            min_dist = dist
    return min_dist if min_dist != float('inf') else 10.0

def predict_motion_for_obstacles(obstacles, timestamp, stats):
    predictions = {}
    for obs_id, obs in obstacles.items():
        centroid = obs["centroid"]
        cluster_points = obs["points"]
        if len(cluster_points) < 2:
            velocity = [0.0, 0.0]
        else:
            delta = cluster_points[-1] - cluster_points[0]
            velocity = delta[:2] / 0.1
        yaw = np.arctan2(velocity[1], velocity[0])
        rel_dist = compute_relative_distance(centroid, obstacles)
        state = list(centroid[:2]) + list(velocity) + [yaw, rel_dist]
        if obs_id not in obstacle_histories:
            obstacle_histories[obs_id] = []
        obstacle_histories[obs_id].append((timestamp, state))
        history = sorted(obstacle_histories[obs_id], key=lambda x: x[0])
        if len(history) < 3:
            while len(history) < 3:
                history.insert(0, history[0])
        past_seq = [s for _, s in history[-3:]]
        input_tensor = prepare_lstm_input(past_seq, stats)
        with torch.no_grad():
            future_vels = model(input_tensor).squeeze(0).numpy()
        future_vels = future_vels * stats["vel_std"] + stats["vel_mean"]
        dt = 0.5
        pos = np.array(past_seq[-1][:2])
        future_positions = []
        for v in future_vels:
            pos = pos + np.array(v) * dt
            future_positions.append(pos.copy())
        predictions[obs_id] = future_positions
    return predictions

def load_nuscenes_pcd_bin(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 5))
    xyz = points[:, :3]
    intensity = points[:, 3] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    colors = np.tile(intensity[:, np.newaxis], (1, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, points[:, 4]

def center_point_cloud(pcd, ring_data):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Point cloud is empty.")
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    pcd.points = o3d.utility.Vector3dVector(points_centered)
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.1)
    num_points_downsampled = len(np.asarray(pcd_downsampled.points))
    ring_data = ring_data[:num_points_downsampled]
    return pcd_downsampled, ring_data

def remove_ground(pcd, ring_data, z_threshold=-1.5, ring_threshold=10):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mask = (points[:, 2] > z_threshold) | (ring_data > ring_threshold)
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return pcd, ring_data[mask]

def smart_split(pcd):
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points, axis=1)
    quadrants = {
        "front": (points[:, 0] >= 0) & (abs(points[:, 1]) <= points[:, 0]),
        "rear": (points[:, 0] < 0) & (abs(points[:, 1]) <= abs(points[:, 0])),
        "left": (points[:, 1] < 0) & (abs(points[:, 0]) < abs(points[:, 1])),
        "right": (points[:, 1] >= 0) & (abs(points[:, 0]) < points[:, 1])
    }
    split_pcds = {}
    for quad, mask in quadrants.items():
        for rmin, rmax in [(0, 5), (5, 10)]:
            dist_mask = (distances >= rmin) & (distances < rmax) & mask
            pcd_split = o3d.geometry.PointCloud()
            pcd_split.points = o3d.utility.Vector3dVector(points[dist_mask])
            pcd_split.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[dist_mask])
            split_pcds[f"{quad}_{rmin}-{rmax}m"] = pcd_split
    return split_pcds

def detect_obstacles(split_pcds, min_points=100, eps=1.0):
    obstacles = {}
    for key, pcd in split_pcds.items():
        points = np.asarray(pcd.points)
        if len(points) < min_points:
            continue
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
        labels = clustering.labels_
        for label in set(labels) - {-1}:
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            distance = np.linalg.norm(centroid)
            obstacles[f"{key}_cluster_{label}"] = {
                "centroid": centroid,
                "distance": distance,
                "points": cluster_points
            }
    return obstacles

def classify_dangers_and_suggest(obstacles):
    alerts = []
    suggestions = []
    for key, obs in obstacles.items():
        dist = obs["distance"]
        region = key.split("_")[0]
        if dist < 2.0:
            alert = f"Critical Danger in {region.capitalize()} ({dist:.1f}m)!"
            suggestion = "Stop Immediately"
        elif dist < 3.0:
            alert = f"Obstacle in {region.capitalize()} ({dist:.1f}m)!"
            if "left" in region:
                suggestion = "Turn Right"
            elif "right" in region:
                suggestion = "Turn Left"
            elif "rear" in region:
                suggestion = "Accelerate"
            else:
                suggestion = "Slow Down"
        else:
            continue
        alerts.append(alert)
        suggestions.append(suggestion)
    return alerts, suggestions
