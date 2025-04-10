import sys
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pygame
import os
import time
import glob

# --- Data Loading and Preprocessing ---
def load_nuscenes_pcd_bin(file_path):
    """Load a nuScenes .pcd.bin file into an Open3D point cloud."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    scan = np.fromfile(file_path, dtype=np.float32)
    # nuScenes format: 5 floats per point (x, y, z, intensity, ring)
    points = scan.reshape((-1, 5))  # Reshape to N x 5
    # Use only x, y, z (first 3 columns)
    xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def center_point_cloud(pcd):
    """Center the point cloud by subtracting the centroid."""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Point cloud is empty.")
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    pcd.points = o3d.utility.Vector3dVector(points_centered)
    # Downsample for performance
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    return pcd

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
            split_pcds[f"{quad}_{rmin}-{rmax}m"] = pcd_split
    return split_pcds

# --- Obstacle Detection and Suggestions ---
def detect_obstacles(split_pcds, min_points=50):
    """Detect obstacles in each split region using clustering."""
    obstacles = {}
    for key, pcd in split_pcds.items():
        points = np.asarray(pcd.points)
        if len(points) < min_points:
            continue
        clustering = DBSCAN(eps=0.5, min_samples=min_points).fit(points)
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
        centroid = obs["centroid"]
        region = key.split("_")[0]  # e.g., "front"
        if dist < 2.0:
            alert = f"Critical Danger in {region.capitalize()} ({dist:.1f}m)!"
            suggestion = "Stop Immediately"
        elif dist < 5.0:
            alert = f"Obstacle in {region.capitalize()} ({dist:.1f}m)!"
            if "left" in region:
                suggestion = "Turn Right"
            elif "right" in region:
                suggestion = "Turn Left"
            elif "rear" in region:
                suggestion = "Accelerate"
            else:  # front
                suggestion = "Slow Down"
        else:
            continue
        alerts.append(alert)
        suggestions.append(suggestion)
    return alerts, suggestions

# --- UI/UX with PyQt5 ---
class DriverAlertApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Alert System")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left: 2D Top-Down View
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.fig = Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Top-Down View")
        left_layout.addWidget(self.canvas)
        main_layout.addWidget(left_widget, stretch=1)

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

        # Blinking for critical alerts
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_alert_color)

    def update_2d_view(self, pcd, obstacles):
        """Update the 2D top-down view."""
        self.ax.clear()
        points = np.asarray(pcd.points)
        colors = np.zeros_like(points)
        for obs in obstacles.values():
            obs_points = obs["points"]
            mask = np.isin(points, obs_points).all(axis=1)
            colors[mask] = [1, 0, 0]  # Red for obstacles
        self.ax.scatter(points[:, 0], points[:, 1], c=colors, s=1)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Top-Down View")
        self.canvas.draw()

    def show_3d_view(self):
        """Display the 3D point cloud with a car model."""
        if self.pcd is None:
            print("No point cloud data available.")
            return
        points = np.asarray(self.pcd.points)
        colors = np.zeros_like(points)
        for obs in self.obstacles.values():
            obs_points = obs["points"]
            mask = np.isin(points, obs_points).all(axis=1)
            colors[mask] = [1, 0, 0]  # Red for obstacles
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # Add car model
        car = o3d.geometry.TriangleMesh.create_box(width=2, height=1, depth=0.5)
        car.translate([-1, -0.5, -0.25])  # Center at origin
        car.paint_uniform_color([0, 0, 1])  # Blue car

        o3d.visualization.draw_geometries([self.pcd, car], window_name="Point Cloud View",
                                         front=[0, -1, 0], lookat=[0, 0, 0], up=[0, 0, 1], zoom=0.1)

    def update_data(self, pcd, obstacles, alerts, suggestions):
        """Update the UI with new data."""
        self.pcd = pcd
        self.obstacles = obstacles
        self.alert_label.setText("Alerts:\n" + "\n".join(alerts if alerts else ["No dangers detected."]))
        self.suggestion_label.setText("Suggestions:\n" + "\n".join(suggestions if suggestions else ["Drive safely."]))
        self.update_2d_view(pcd, obstacles)
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
        pcd = load_nuscenes_pcd_bin(file_path)
        pcd = center_point_cloud(pcd)
        split_pcds = smart_split(pcd)
        obstacles = detect_obstacles(split_pcds)
        alerts, suggestions = classify_dangers_and_suggest(obstacles)
        return pcd, obstacles, alerts, suggestions
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, {}, ["Error processing file."], ["Check file path and format."]

def main():
    # Initialize Qt application
    app = QApplication(sys.argv)

    # Create window with initial empty state
    window = DriverAlertApp()
    window.show()

    # Real-time simulation with multiple files
    pcd_files = sorted(glob.glob("D:/HACKATHON/wires_boxes/v1.0-mini/samples/LIDAR_TOP/*.pcd.bin"))  # Update directory
    if not pcd_files:
        print("No .bin files found in the specified directory.")
        window.update_data(None, {}, ["No files found."], ["Check directory path."])
    else:
        for pcd_file in pcd_files:
            pcd, obstacles, alerts, suggestions = process_pcd_file(pcd_file)
            if pcd is not None:
                window.update_data(pcd, obstacles, alerts, suggestions)
                app.processEvents()
                time.sleep(0.1)  # Simulate 10 FPS

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()