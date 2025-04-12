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

class DriverAlertApp(QMainWindow):
    def __init__(self, camera_files):
        super().__init__()
        self.setWindowTitle("Driver Alert System")
        self.setGeometry(100, 100, 1400, 900)

        self.camera_files = camera_files
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        lidar_widget = QWidget()
        lidar_layout = QVBoxLayout(lidar_widget)
        self.fig = Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Top-Down LIDAR View")
        lidar_layout.addWidget(self.canvas)
        self.pred_fig = Figure(figsize=(4, 3))
        self.pred_canvas = FigureCanvas(self.pred_fig)
        self.pred_ax = self.pred_fig.add_subplot(111)
        self.pred_ax.set_title("Predicted Paths")
        self.pred_ax.set_xlabel("X (m)")
        self.pred_ax.set_ylabel("Y (m)")
        lidar_layout.addWidget(self.pred_canvas)
        main_layout.addWidget(lidar_widget, stretch=1)

        camera_widget = QWidget()
        camera_layout = QVBoxLayout(camera_widget)
        front_layout = QHBoxLayout()
        front_vbox = QVBoxLayout()
        front_title = QLabel("CAM_FRONT")
        front_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        front_title.setAlignment(Qt.AlignCenter)
        self.front_label = QLabel()
        front_vbox.addWidget(front_title)
        front_vbox.addWidget(self.front_label)
        front_layout.addLayout(front_vbox)
        front_left_vbox = QVBoxLayout()
        front_left_title = QLabel("CAM_FRONT_LEFT")
        front_left_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        front_left_title.setAlignment(Qt.AlignCenter)
        self.front_left_label = QLabel()
        front_left_vbox.addWidget(front_left_title)
        front_left_vbox.addWidget(self.front_left_label)
        front_layout.addLayout(front_left_vbox)
        front_right_vbox = QVBoxLayout()
        front_right_title = QLabel("CAM_FRONT_RIGHT")
        front_right_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        front_right_title.setAlignment(Qt.AlignCenter)
        self.front_right_label = QLabel()
        front_right_vbox.addWidget(front_right_title)
        front_right_vbox.addWidget(self.front_right_label)
        front_layout.addLayout(front_right_vbox)
        back_layout = QHBoxLayout()
        back_vbox = QVBoxLayout()
        back_title = QLabel("CAM_BACK")
        back_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        back_title.setAlignment(Qt.AlignCenter)
        self.back_label = QLabel()
        back_vbox.addWidget(back_title)
        back_vbox.addWidget(self.back_label)
        back_layout.addLayout(back_vbox)
        back_left_vbox = QVBoxLayout()
        back_left_title = QLabel("CAM_BACK_LEFT")
        back_left_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        back_left_title.setAlignment(Qt.AlignCenter)
        self.back_left_label = QLabel()
        back_left_vbox.addWidget(back_left_title)
        back_left_vbox.addWidget(self.back_left_label)
        back_layout.addLayout(back_left_vbox)
        back_right_vbox = QVBoxLayout()
        back_right_title = QLabel("CAM_BACK_RIGHT")
        back_right_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        back_right_title.setAlignment(Qt.AlignCenter)
        self.back_right_label = QLabel()
        back_right_vbox.addWidget(back_right_title)
        back_right_vbox.addWidget(self.back_right_label)
        back_layout.addLayout(back_right_vbox)
        camera_layout.addLayout(front_layout)
        camera_layout.addLayout(back_layout)
        main_layout.addWidget(camera_widget, stretch=3)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setAlignment(Qt.AlignCenter)
        title = QLabel("Driver Alert Dashboard")
        title.setStyleSheet("font-size: 18pt; font-weight: bold; color: #333;")
        right_layout.addWidget(title, alignment=Qt.AlignCenter)
        self.alert_label = QLabel("Alerts:\nInitializing...")
        self.alert_label.setStyleSheet("font-size: 14pt; color: red; background-color: #ffe6e6; padding: 10px;")
        self.alert_label.setWordWrap(True)
        self.suggestion_label = QLabel("Suggestions:\nInitializing...")
        self.suggestion_label.setStyleSheet("font-size: 14pt; color: green; background-color: #e6ffe6; padding: 10px;")
        self.suggestion_label.setWordWrap(True)
        button_layout = QHBoxLayout()
        self.dismiss_button = QPushButton("Dismiss Alerts")
        self.dismiss_button.setStyleSheet("font-size: 12pt; padding: 5px;")
        self.dismiss_button.clicked.connect(self.dismiss_alerts)
        self.view_3d_button = QPushButton("View 3D Point Cloud")
        self.view_3d_button.setStyleSheet("font-size: 12pt; padding: 5px;")
        self.view_3d_button.clicked.connect(self.show_3d_view)
        self.pause_button = QPushButton("Pause")
        self.pause_button.setStyleSheet("font-size: 12pt; padding: 5px;")
        self.pause_button.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.dismiss_button)
        button_layout.addWidget(self.view_3d_button)
        button_layout.addWidget(self.pause_button)
        right_layout.addWidget(self.alert_label)
        right_layout.addWidget(self.suggestion_label)
        right_layout.addLayout(button_layout)
        main_layout.addWidget(right_widget, stretch=1)

        self.pcd = None
        self.obstacles = {}
        self.ring_data = None
        self.is_paused = False

        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_alert_color)

    def update_2d_view(self, pcd, obstacles):
        self.ax.clear()
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        for obs in obstacles.values():
            obs_points = obs["points"]
            mask = np.isin(points, obs_points).all(axis=1)
            colors[mask] = [1, 0, 0]
        self.ax.scatter(points[:, 0], points[:, 1], c=colors, s=1)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Top-Down LIDAR View")
        self.canvas.draw()

        self.pred_ax.clear()
        predictions = predict_motion_for_obstacles(obstacles, int(time.time() * 1000), norm_stats)
        car_size = 2.0
        self.pred_ax.add_patch(plt.Rectangle((-car_size/4, -car_size/4), car_size, car_size,
                                            linewidth=1, edgecolor='deepskyblue', facecolor='lightblue', label='Ego Vehicle'))
        for obs_id, path in predictions.items():
            xs, ys = zip(*path)
            final_point = np.array(path[-1])
            start_point = np.array(path[0])
            direction = final_point - start_point
            distance = np.linalg.norm(final_point)
            if distance < 2.0:
                color = 'red'
            elif distance < 4.0:
                color = 'orange'
            else:
                color = 'green'
            angle_to_front = np.arccos(direction[0] / (np.linalg.norm(direction) + 1e-6))
            if np.abs(angle_to_front) < np.pi / 6 and distance < 4.0:
                color = 'red'
            self.pred_ax.plot(xs, ys, linestyle='--', color=color, linewidth=1)
            self.pred_ax.arrow(xs[-2], ys[-2], direction[0]*0.2, direction[1]*0.2,
                              head_width=0.3, head_length=0.5, fc=color, ec=color)
        self.pred_ax.legend(loc="upper right")
        self.pred_ax.set_xlim(-10, 10)
        self.pred_ax.set_ylim(-10, 10)
        self.pred_ax.set_aspect('equal', adjustable='box')
        self.pred_ax.grid(True)
        self.pred_ax.set_title("Predicted Paths")
        self.pred_ax.set_xlabel("X (m)")
        self.pred_ax.set_ylabel("Y (m)")
        self.pred_canvas.draw()

    def update_camera_views(self, lidar_timestamp):
        cam_dirs = {
            'CAM_FRONT': (self.front_label, self.camera_files['CAM_FRONT']),
            'CAM_FRONT_LEFT': (self.front_left_label, self.camera_files['CAM_FRONT_LEFT']),
            'CAM_FRONT_RIGHT': (self.front_right_label, self.camera_files['CAM_FRONT_RIGHT']),
            'CAM_BACK': (self.back_label, self.camera_files['CAM_BACK']),
            'CAM_BACK_LEFT': (self.back_left_label, self.camera_files['CAM_BACK_LEFT']),
            'CAM_BACK_RIGHT': (self.back_right_label, self.camera_files['CAM_BACK_RIGHT'])
        }
        for cam, (label, (cam_timestamps, cam_paths)) in cam_dirs.items():
            if not cam_timestamps:
                label.setText("No images")
                continue
            idx = bisect_left(cam_timestamps, lidar_timestamp)
            if idx == 0:
                closest_file = cam_paths[0]
            elif idx == len(cam_timestamps):
                closest_file = cam_paths[-1]
            else:
                closest_file = cam_paths[idx] if abs(cam_timestamps[idx] - lidar_timestamp) < abs(cam_timestamps[idx-1] - lidar_timestamp) else cam_paths[idx-1]
            pixmap = QPixmap(closest_file).scaled(600, 500, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)

    def show_3d_view(self):
        if self.pcd is None:
            print("No point cloud data available.")
            return
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)
        for obs in self.obstacles.values():
            obs_points = obs["points"]
            mask = np.isin(points, obs_points).all(axis=1)
            colors[mask] = [1, 0, 0]
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        car = o3d.geometry.TriangleMesh.create_box(width=2, height=1, depth=0.5)
        car.translate([-1, -0.5, -0.25])
        car.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([self.pcd, car], window_name="Point Cloud View",
                                         front=[0, -1, 0], lookat=[0, 0, 0], up=[0, 0, 1], zoom=0.1)

    def update_data(self, pcd, obstacles, alerts, suggestions, lidar_timestamp):
        self.pcd = pcd
        self.obstacles = obstacles
        self.alert_label.setText("Alerts:\n" + "\n".join(alerts if alerts else ["No dangers detected."]))
        self.suggestion_label.setText("Suggestions:\n" + "\n".join(suggestions if suggestions else ["Drive safely."]))
        self.update_2d_view(pcd, obstacles)
        self.update_camera_views(lidar_timestamp)
        if any("Critical" in a for a in alerts):
            self.blink_timer.start(500)
        else:
            self.blink_timer.stop()
        if obstacles:
            self.play_alert_sound()

    def toggle_alert_color(self):
        current = self.alert_label.styleSheet()
        new_color = "red" if "ffe6e6" in current else "#ff9999"
        self.alert_label.setStyleSheet(f"font-size: 14pt; color: {new_color}; background-color: #ffe6e6; padding: 10px;")

    def dismiss_alerts(self):
        self.alert_label.setText("Alerts:\nNo dangers detected.")
        self.suggestion_label.setText("Suggestions:\nDrive safely.")
        self.blink_timer.stop()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_button.setText("Continue" if self.is_paused else "Pause")

    def play_alert_sound(self):
        pygame.mixer.init()
        try:
            pygame.mixer.Sound("alert.wav").play()
        except FileNotFoundError:
            print("Warning: alert.wav not found. Skipping audio.")

def process_pcd_file(file_path):
    try:
        pcd, ring_data = load_nuscenes_pcd_bin(file_path)
        pcd, ring_data = center_point_cloud(pcd, ring_data)
        pcd, ring_data = remove_ground(pcd, ring_data)
        split_pcds = smart_split(pcd)
        obstacles = detect_obstacles(split_pcds, min_points=100, eps=1.0)
        alerts, suggestions = classify_dangers_and_suggest(obstacles)
        timestamp = int(file_path.split('__')[-1].replace('.pcd.bin', ''))
        return pcd, obstacles, alerts, suggestions, timestamp
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, {}, ["Error processing file."], ["Check file path and format."], 0

def load_camera_files(base_dir):
    cam_dirs = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    camera_files = {}
    for cam in cam_dirs:
        files = sorted(glob.glob(os.path.join(base_dir, cam, "*.jpg")))
        timestamps = [int(f.split('__')[-1].replace('.jpg', '')) for f in files]
        camera_files[cam] = (timestamps, files)
    return camera_files

def main():
    app = QApplication(sys.argv)
    base_dir = "v1.0-mini/samples/"
    camera_files = load_camera_files(base_dir)
    window = DriverAlertApp(camera_files)
    window.show()
    pcd_files = sorted(glob.glob(os.path.join(base_dir, "LIDAR_TOP", "*.pcd.bin")))
    if not pcd_files:
        print("No .bin files found in the specified directory.")
        window.update_data(None, {}, ["No files found."], ["Check directory path."], 0)
    else:
        for pcd_file in pcd_files:
            while window.is_paused:
                app.processEvents()
                time.sleep(0.1)
            pcd, obstacles, alerts, suggestions, timestamp = process_pcd_file(pcd_file)
            if pcd is not None:
                window.update_data(pcd, obstacles, alerts, suggestions, timestamp)
                app.processEvents()
                time.sleep(0.1)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()