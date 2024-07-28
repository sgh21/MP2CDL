#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Partial2Complete 
@File    ：modelweight.py
@IDE     ：PyCharm 
@Author  ：Wang Song
@Date    ：2024/5/14 下午9:35 
"""
import torch


def extract_and_save_weights(old_path, new_path):
    # Load the old checkpoint
    old_checkpoint = torch.load(old_path)

    # Extract the model weights
    model_weights = old_checkpoint['base_model']

    # Save the model weights to a new .pth file
    torch.save(model_weights, new_path)


# Use the function
old_path = './experiments/P2C/EPN3D_models/hi/ckpt-best.pth'
new_path = './experiments/P2C/EPN3D_models/hi/model_weights.pth'
extract_and_save_weights(old_path, new_path)
