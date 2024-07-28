#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Partial2Complete 
@File    ：read3d.py
@IDE     ：PyCharm 
@Author  ：Wang Song
@Date    ：2024/5/14 上午10:48 
"""
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Load the data
data = np.load('./data/EPN3D/plane/complete/1d7eb22189100710ca8607f540cc62ba.npy')

# Convert to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
