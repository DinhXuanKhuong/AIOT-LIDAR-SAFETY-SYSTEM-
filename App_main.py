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
from App_DriverAlert import DriverAlertApp



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