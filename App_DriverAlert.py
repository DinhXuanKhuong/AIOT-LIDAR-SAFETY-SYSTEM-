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
from App_Functions import (
    load_nuscenes_pcd_bin, center_point_cloud, remove_ground, smart_split,
    detect_obstacles, classify_dangers_and_suggest, predict_motion_for_obstacles
)



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