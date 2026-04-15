#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Simple MLP policy network for Robot Vacuum.
清扫大作战策略网络。
"""

import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config

def _make_fc(in_dim, out_dim, gain=1.41421):
    """Linear layer with orthogonal initialization.

    使用正交初始化的全连接层。
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer

def _make_cnn(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Conv2d layer with orthogonal initialization."""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
    nn.init.zeros_(layer.bias)
    return layer

class Model(nn.Module):
    """Multi-stream CNN+MLP for Robot Vacuum.

    清扫大作战多流 CNN+MLP 策略网络。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        obs_dim = Config.DIM_OF_OBSERVATION
        act_num = Config.ACTION_NUM  # 8

        # 1. CNN Branch (Local View 7x7)
        self.cnn_net = nn.Sequential(
            _make_cnn(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            _make_cnn(16, 32, kernel_size=3, stride=1), # 7x7 -> 5x5
            nn.GELU(),
            nn.Flatten() # 32 * 5 * 5 = 800
        )
        cnn_out_dim = 32 * 5 * 5

        # 2. MLP Branch (Global & Entity & History)
        mlp_in_dim = obs_dim - 49
        self.mlp_net = nn.Sequential(
            _make_fc(mlp_in_dim, 64),
            nn.GELU()
        )
        mlp_out_dim = 64

        combined_dim = cnn_out_dim + mlp_out_dim

        # 3. Decoupled Actor and Critic with LayerNorm
        self.actor_backbone = nn.Sequential(
            _make_fc(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            _make_fc(256, 128),
            nn.GELU()
        )
        self.actor_head = _make_fc(128, act_num, gain=0.01)

        self.critic_backbone = nn.Sequential(
            _make_fc(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            _make_fc(256, 128),
            nn.GELU()
        )
        self.critic_head = _make_fc(128, 1, gain=1.0)

    def set_eval_mode(self):
        """Set to evaluation mode.

        设置为评估模式。
        """
        self.eval()

    def forward(self, s, inference=False):
        """Forward pass.

        前向传播。
        """
        x = s.to(torch.float32)
        
        # Split features
        x_grid = x[:, :49].view(-1, 1, 7, 7)
        x_rest = x[:, 49:]
        
        # Branch forward
        f_cnn = self.cnn_net(x_grid)
        f_mlp = self.mlp_net(x_rest)
        
        # Fusion
        f_combined = torch.cat([f_cnn, f_mlp], dim=1)
        
        # Heads
        h_actor = self.actor_backbone(f_combined)
        logits = self.actor_head(h_actor)
        
        h_critic = self.critic_backbone(f_combined)
        value = self.critic_head(h_critic)
        
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
