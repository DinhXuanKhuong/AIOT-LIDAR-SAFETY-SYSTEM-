import sys
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pygame
import os
import time
import glob
from bisect import bisect_left

# --- Data Loading and Preprocessing ---
def load_nuscenes_pcd_bin(file_path):
    """Load a nuScenes .pcd.bin file into an Open3D point cloud with intensity."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    scan = np.fromfile(file_path, dtype=np.float32)
    points = scan.reshape((-1, 5))  # nuScenes: x, y, z, intensity, ring
    xyz = points[:, :3]
    intensity = points[:, 3] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    colors = np.tile(intensity[:, np.newaxis], (1, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, points[:, 4]

def center_point_cloud(pcd, ring_data):
    """Center the point cloud by subtracting the centroid and downsample."""
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
    """Remove ground points using z height and ring index."""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mask = (points[:, 2] > z_threshold) | (ring_data > ring_threshold)
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return pcd, ring_data[mask]

# --- Smart Splitting ---
def smart_split(pcd):
    """Split point cloud into quadrants and distance zones."""
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

# --- Obstacle Detection and Suggestions ---
def detect_obstacles(split_pcds, min_points=100, eps=1.0):
    """Detect obstacles in each split region using clustering."""
    obstacles = {}
    for key, pcd in split_pcds.items():
        points = np.asarray(pcd.points)
        if len(points) < min_points:
            continue
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
        labels = clustering.labels_
        for label in set(labels) - {-1}:  # Exclude noise
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
    """Classify obstacles and suggest actions."""
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

# --- UI/UX with PyQt5 ---
class DriverAlertApp(QMainWindow):
    def __init__(self, camera_files):
        super().__init__()
        self.setWindowTitle("Driver Alert System")
        self.setGeometry(100, 100, 1200, 800)

        # Store sorted camera files
        self.camera_files = camera_files

        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left: 2D LIDAR View
        lidar_widget = QWidget()
        lidar_layout = QVBoxLayout(lidar_widget)
        self.fig = Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Top-Down LIDAR View")
        lidar_layout.addWidget(self.canvas)
        main_layout.addWidget(lidar_widget, stretch=1)

        # Center: Camera Images
        camera_widget = QWidget()
        camera_layout = QVBoxLayout(camera_widget)
        
        # Top row: Front cameras
        front_layout = QHBoxLayout()
        # CAM_FRONT
        front_vbox = QVBoxLayout()
        front_title = QLabel("CAM_FRONT")
        front_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        front_title.setAlignment(Qt.AlignCenter)
        self.front_label = QLabel()
        front_vbox.addWidget(front_title)
        front_vbox.addWidget(self.front_label)
        front_layout.addLayout(front_vbox)
        # CAM_FRONT_LEFT
        front_left_vbox = QVBoxLayout()
        front_left_title = QLabel("CAM_FRONT_LEFT")
        front_left_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        front_left_title.setAlignment(Qt.AlignCenter)
        self.front_left_label = QLabel()
        front_left_vbox.addWidget(front_left_title)
        front_left_vbox.addWidget(self.front_left_label)
        front_layout.addLayout(front_left_vbox)
        # CAM_FRONT_RIGHT
        front_right_vbox = QVBoxLayout()
        front_right_title = QLabel("CAM_FRONT_RIGHT")
        front_right_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        front_right_title.setAlignment(Qt.AlignCenter)
        self.front_right_label = QLabel()
        front_right_vbox.addWidget(front_right_title)
        front_right_vbox.addWidget(self.front_right_label)
        front_layout.addLayout(front_right_vbox)
        
        # Bottom row: Back cameras
        back_layout = QHBoxLayout()
        # CAM_BACK
        back_vbox = QVBoxLayout()
        back_title = QLabel("CAM_BACK")
        back_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        back_title.setAlignment(Qt.AlignCenter)
        self.back_label = QLabel()
        back_vbox.addWidget(back_title)
        back_vbox.addWidget(self.back_label)
        back_layout.addLayout(back_vbox)
        # CAM_BACK_LEFT
        back_left_vbox = QVBoxLayout()
        back_left_title = QLabel("CAM_BACK_LEFT")
        back_left_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #333;")
        back_left_title.setAlignment(Qt.AlignCenter)
        self.back_left_label = QLabel()
        back_left_vbox.addWidget(back_left_title)
        back_left_vbox.addWidget(self.back_left_label)
        back_layout.addLayout(back_left_vbox)
        # CAM_BACK_RIGHT
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
        main_layout.addWidget(camera_widget, stretch=2)

        # Right: Alerts and Suggestions
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
        button_layout.addWidget(self.dismiss_button)
        button_layout.addWidget(self.view_3d_button)

        right_layout.addWidget(self.alert_label)
        right_layout.addWidget(self.suggestion_label)
        right_layout.addLayout(button_layout)
        main_layout.addWidget(right_widget, stretch=1)

        # Store data
        self.pcd = None
        self.obstacles = {}
        self.ring_data = None

        # Blinking for critical alerts
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_alert_color)

    def update_2d_view(self, pcd, obstacles):
        """Update the 2D top-down view with intensity-based colors."""
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

    def update_camera_views(self, lidar_timestamp):
        """Update camera images based on closest LIDAR timestamp."""
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
            # Find the closest timestamp
            idx = bisect_left(cam_timestamps, lidar_timestamp)
            if idx == 0:
                closest_file = cam_paths[0]
            elif idx == len(cam_timestamps):
                closest_file = cam_paths[-1]
            else:
                closest_file = cam_paths[idx] if abs(cam_timestamps[idx] - lidar_timestamp) < abs(cam_timestamps[idx-1] - lidar_timestamp) else cam_paths[idx-1]
            pixmap = QPixmap(closest_file).scaled(200, 150, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)

    def show_3d_view(self):
        """Display the 3D point cloud with a car model and intensity colors."""
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
        """Update the UI with new data."""
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
        """Toggle alert color for blinking effect."""
        current = self.alert_label.styleSheet()
        new_color = "red" if "ffe6e6" in current else "#ff9999"
        self.alert_label.setStyleSheet(f"font-size: 14pt; color: {new_color}; background-color: #ffe6e6; padding: 10px;")

    def dismiss_alerts(self):
        """Clear alerts and suggestions."""
        self.alert_label.setText("Alerts:\nNo dangers detected.")
        self.suggestion_label.setText("Suggestions:\nDrive safely.")
        self.blink_timer.stop()

    def play_alert_sound(self):
        """Play an alert sound."""
        pygame.mixer.init()
        try:
            pygame.mixer.Sound("alert.wav").play()
        except FileNotFoundError:
            print("Warning: alert.wav not found. Skipping audio.")

# --- Main Application Logic ---
def process_pcd_file(file_path):
    """Process a single .pcd.bin file and return data for UI."""
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
    """Load and sort camera files by timestamp."""
    cam_dirs = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    camera_files = {}
    for cam in cam_dirs:
        files = sorted(glob.glob(os.path.join(base_dir, cam, "*.jpg")))
        timestamps = [int(f.split('__')[-1].replace('.jpg', '')) for f in files]
        camera_files[cam] = (timestamps, files)
    return camera_files

def main():
    app = QApplication(sys.argv)

    # Load and sort camera files
    base_dir = "D:/HACKATHON/wires_boxes/v1.0-mini/samples/"
    camera_files = load_camera_files(base_dir)

    # Initialize window with camera files
    window = DriverAlertApp(camera_files)
    window.show()

    # Load and sort LIDAR files
    pcd_files = sorted(glob.glob(os.path.join(base_dir, "LIDAR_TOP", "*.pcd.bin")))
    if not pcd_files:
        print("No .bin files found in the specified directory.")
        window.update_data(None, {}, ["No files found."], ["Check directory path."], 0)
    else:
        for pcd_file in pcd_files:
            pcd, obstacles, alerts, suggestions, timestamp = process_pcd_file(pcd_file)
            if pcd is not None:
                window.update_data(pcd, obstacles, alerts, suggestions, timestamp)
                app.processEvents()
                time.sleep(0.1)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()