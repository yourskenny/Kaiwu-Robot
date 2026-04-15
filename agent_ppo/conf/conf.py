#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum PPO agent.
清扫大作战 PPO 配置。
"""


class Config:

    # Feature dimensions
    # 特征维度
    LOCAL_VIEW_DIM = 7 * 7
    GLOBAL_STATE_DIM = 24 # 6 base + 16 rays + 2 extra
    ENTITY_STATE_DIM = 12
    HISTORY_STATE_DIM = 10
    LAST_ACTION_DIM = 8
    FEATURES = [
        LOCAL_VIEW_DIM,
        GLOBAL_STATE_DIM,
        ENTITY_STATE_DIM,
        HISTORY_STATE_DIM,
        LAST_ACTION_DIM,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space: 8 directional moves
    # 动作空间：8个方向移动
    ACTION_NUM = 8

    # Single-head value
    # 单头价值
    VALUE_NUM = 1

    # PPO hyperparameters
    # PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 0.5

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # Compact PPO improvements
    # 保持 PPO 架构不变，仅增加轻量训练细节
    USE_ADVANTAGE_NORM = True
    ADVANTAGE_NORM_EPS = 1e-8

    # Lightweight feature/reward thresholds
    # 轻量特征与奖励塑形阈值
    LOW_BATTERY_RATIO = 0.35
    NPC_DANGER_DISTANCE = 4.0
