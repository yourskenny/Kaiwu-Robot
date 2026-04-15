#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np

from agent_ppo.conf.conf import Config


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


def _signed_norm(v, abs_max):
    """Normalize signed value to [0, 1] with 0 mapped to 0.5.

    将带符号值归一化到 [0, 1]，其中 0 映射到 0.5。
    """
    if abs_max <= 0:
        return 0.5
    clipped = float(np.clip(v, -abs_max, abs_max))
    return 0.5 * (clipped / abs_max + 1.0)


def _grid_dist(dx, dz):
    """Chebyshev distance under 8-direction movement.

    8方向移动下，使用切比雪夫距离近似最少步数。
    """
    return float(max(abs(dx), abs(dz)))


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.max_step = 1000
        self.battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)
        self.last_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.cleaned_this_step = 0
        self.total_dirt = 1

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self.nearest_charger_dist = float(self.GRID_SIZE)
        self.last_nearest_charger_dist = float(self.GRID_SIZE)
        self.nearest_charger_dx = 0.0
        self.nearest_charger_dz = 0.0

        self.nearest_npc_dist = float(self.GRID_SIZE)
        self.last_nearest_npc_dist = float(self.GRID_SIZE)
        self.nearest_npc_dx = 0.0
        self.nearest_npc_dz = 0.0

        self.charge_count = 0
        self.last_charge_count = 0
        self.charged_last_step = 0.0

        self.prev_action = -1
        self.no_move_streak = 0
        self.same_action_streak = 0
        self.steps_since_clean = 0
        self.blocked_step_count = 0

        # 访问计数只用作轻量记忆，不参与框架逻辑。
        self.visit_count_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int16)
        self.current_cell_visit_count = 0
        self.revisit_step_count = 0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        self.clean_combo = 0
        self.last_action = None
        self.current_action = None

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]
        prev_pos = self.cur_pos

        self.step_no = int(observation["step_no"])
        self.last_action = self.current_action
        self.current_action = last_action # in pb2struct last_action parameter actually means the action taken in this step
        self.max_step = max(int(env_info.get("max_step", self.max_step)), 1)
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # Battery / 电量
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Charge / 充电
        self.last_charge_count = self.charge_count
        self.charge_count = int(env_info.get("charge_count", 0))
        self.charged_last_step = 1.0 if self.charge_count > self.last_charge_count else 0.0

        # Legal actions / 合法动作
        legal_act = observation.get("legal_action") or observation.get("legal_act") or [1] * 8
        self._legal_act = [int(x) for x in legal_act]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

        # 这里集中维护轻量时序记忆，避免引入 recurrent PPO。
        self._update_history(prev_pos, last_action)
        self._update_visit_count()
        self._update_entity_cache(frame_state)

    def _update_history(self, prev_pos, last_action):
        """Update compact temporal statistics.

        更新轻量时序统计量。
        """
        self.last_pos = prev_pos

        if last_action != -1 and self.cur_pos == prev_pos:
            self.no_move_streak += 1
            self.blocked_step_count += 1
        else:
            self.no_move_streak = 0

        if last_action != -1 and last_action == self.prev_action:
            self.same_action_streak += 1
        else:
            self.same_action_streak = 0

        if self.cleaned_this_step > 0:
            self.steps_since_clean = 0
        elif last_action != -1:
            self.steps_since_clean += 1
        else:
            self.steps_since_clean = 0

        self.prev_action = last_action

    def _update_visit_count(self):
        """Update per-cell visit statistics.

        更新当前位置访问统计，用于压制原地打转与重复覆盖。
        """
        x, z = self.cur_pos
        if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            self.visit_count_map[x, z] += 1
            self.current_cell_visit_count = int(self.visit_count_map[x, z])
            if self.current_cell_visit_count > 1:
                self.revisit_step_count += 1
        else:
            self.current_cell_visit_count = 0

    def _update_entity_cache(self, frame_state):
        """Cache nearest charger and NPC states.

        缓存最近充电桩与官方机器人的相对信息。
        """
        hx, hz = self.cur_pos

        self.last_nearest_charger_dist = self.nearest_charger_dist
        self.nearest_charger_dist = float(self.GRID_SIZE)
        self.nearest_charger_dx = 0.0
        self.nearest_charger_dz = 0.0

        for organ in frame_state.get("organs") or []:
            if int(organ.get("sub_type", 1)) != 1:
                continue
            pos = organ.get("pos") or {}
            dx = int(pos.get("x", hx)) - hx
            dz = int(pos.get("z", hz)) - hz
            dist = _grid_dist(dx, dz)
            if dist < self.nearest_charger_dist:
                self.nearest_charger_dist = dist
                self.nearest_charger_dx = float(dx)
                self.nearest_charger_dz = float(dz)

        self.last_nearest_npc_dist = self.nearest_npc_dist
        self.nearest_npc_dist = float(self.GRID_SIZE)
        self.nearest_npc_dx = 0.0
        self.nearest_npc_dz = 0.0

        for npc in frame_state.get("npcs") or []:
            pos = npc.get("pos") or {}
            dx = int(pos.get("x", hx)) - hx
            dz = int(pos.get("z", hz)) - hz
            dist = _grid_dist(dx, dz)
            if dist < self.nearest_npc_dist:
                self.nearest_npc_dist = dist
                self.nearest_npc_dx = float(dx)
                self.nearest_npc_dz = float(dz)

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        """Local view feature (49D): crop center 7×7 from 21×21.

        局部视野特征（49D）：从 21×21 视野中心裁剪 7×7。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_global_state_feature(self):
        """Global state feature (24D).

        全局状态特征（24D）。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6..13] ray_dirt       8-directional ray distance to dirt / 8方向最近污渍距离
          [14..21] ray_obs       8-directional ray distance to obstacle / 8方向最近障碍物距离
          [22] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
          [23] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
        """
        step_norm = _norm(self.step_no, self.max_step)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 8-directional ray to find nearest dirt and obstacle
        # 八方向射线找最近污渍和障碍物距离
        ray_dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ] # N, NE, E, SE, S, SW, W, NW
        ray_dirt = []
        ray_obs = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found_dirt = max_ray
            found_obs = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    if found_obs == max_ray:
                        found_obs = step
                    break
                
                if self._view_map is not None:
                    # check local view if within bounds
                    lx = x - hx + self.VIEW_HALF
                    lz = z - hz + self.VIEW_HALF
                    if 0 <= lx < 21 and 0 <= lz < 21:
                        cell = int(self._view_map[lx, lz])
                        if cell == 2 and found_dirt == max_ray:
                            found_dirt = step
                        elif cell == 0 and found_obs == max_ray:
                            found_obs = step
                    else:
                        # outside local view, assume obstacle if out of grid, else ignore dirt
                        pass
                        
                if found_dirt != max_ray and found_obs != max_ray:
                    break
            
            ray_dirt.append(_norm(found_dirt, max_ray))
            ray_obs.append(_norm(found_obs, max_ray))

        # Nearest dirt Euclidean distance (estimated from 7×7 crop)
        # 最近污渍欧氏距离（视野内 7×7 粗估）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        res = [
            step_norm,
            battery_ratio,
            cleaning_progress,
            remaining_dirt,
            pos_x_norm,
            pos_z_norm,
        ]
        res.extend(ray_dirt)
        res.extend(ray_obs)
        res.extend([nearest_dirt_norm, dirt_delta])
        
        return np.array(res, dtype=np.float32)

    def _get_entity_feature(self):
        """Entity feature (12D).

        充电桩与官方机器人相关特征（12D）。
        """
        battery_ratio = self.battery / max(self.battery_max, 1)
        battery_margin = (self.battery - self.nearest_charger_dist) / max(self.battery_max, 1)
        low_battery = 1.0 if battery_ratio < Config.LOW_BATTERY_RATIO else 0.0
        charger_progress = 1.0 if self.nearest_charger_dist < self.last_nearest_charger_dist else 0.0

        npc_danger = 1.0 if self.nearest_npc_dist <= Config.NPC_DANGER_DISTANCE else 0.0
        npc_approaching = 1.0 if self.nearest_npc_dist < self.last_nearest_npc_dist else 0.0

        # Entity relative positions as polar coordinates (distance, angle)
        # 实体相对位置转换为极坐标（距离，角度）
        import math
        charger_angle = math.atan2(self.nearest_charger_dx, self.nearest_charger_dz) / math.pi # [-1, 1]
        npc_angle = math.atan2(self.nearest_npc_dx, self.nearest_npc_dz) / math.pi # [-1, 1]

        return np.array(
            [
                _norm(self.nearest_charger_dist, self.GRID_SIZE),
                charger_angle,
                _signed_norm(battery_margin, 1.0),
                low_battery,
                charger_progress,
                _norm(self.charge_count, 8),
                _norm(self.nearest_npc_dist, self.GRID_SIZE),
                npc_angle,
                npc_danger,
                npc_approaching,
                0.0, # padding to keep 12D for compatibility
                0.0,
            ],
            dtype=np.float32,
        )

    def _get_history_feature(self):
        """History feature (10D).

        轻量时序特征（10D），用于替代更重的循环结构。
        """
        revisit_ratio = np.clip(self.revisit_step_count / max(self.step_no, 1), 0.0, 1.0)
        low_battery_mode = 1.0 if self.battery / max(self.battery_max, 1) < Config.LOW_BATTERY_RATIO else 0.0
        cleaned_last_step = 1.0 if self.cleaned_this_step > 0 else 0.0
        stuck_flag = 1.0 if self.no_move_streak >= 2 or self.steps_since_clean >= 15 else 0.0
        repeated_action_flag = 1.0 if self.same_action_streak >= 2 else 0.0
        blocked_rate = np.clip(self.blocked_step_count / max(self.step_no, 1), 0.0, 1.0)

        return np.array(
            [
                _norm(self.no_move_streak, 10),
                _norm(self.same_action_streak, 10),
                _norm(self.steps_since_clean, 50),
                _norm(max(self.current_cell_visit_count - 1, 0), 7),
                revisit_ratio,
                cleaned_last_step,
                self.charged_last_step,
                low_battery_mode,
                stuck_flag,
                blocked_rate,
            ],
            dtype=np.float32,
        )

    def _get_last_action_feature(self, last_action):
        """Encode previous action as one-hot vector.

        将上一动作编码成 one-hot，帮助策略识别摆动与重复。
        """
        action_feature = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        if 0 <= last_action < Config.ACTION_NUM:
            action_feature[last_action] = 1.0
        return action_feature

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        return list(self._legal_act)

    def feature_process(self, env_obs, last_action):
        """Generate compact feature vector, legal action mask, and scalar reward.

        生成紧凑特征向量、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 49D
        global_state = self._get_global_state_feature()  # 12D
        entity_state = self._get_entity_feature()  # 12D
        history_state = self._get_history_feature()  # 10D
        last_action_feature = self._get_last_action_feature(last_action)  # 8D
        legal_action = self.get_legal_action()  # 8D
        feature = np.concatenate(
            [
                local_view,
                global_state,
                entity_state,
                history_state,
                last_action_feature,
            ]
        )

        reward = self.reward_process()

        return feature, legal_action, reward

    def reward_process(self):
        """Reward shaping aligned with score and survival.

        奖励尽量贴近真实得分，只保留轻量塑形项。
        """
        cleaning_reward = float(self.cleaned_this_step)
        step_penalty = -0.002

        stuck_penalty = -0.03 if self.no_move_streak > 0 else 0.0
        revisit_penalty = -0.005 if self.current_cell_visit_count > 2 and self.cleaned_this_step == 0 else 0.0

        charger_reward = 0.0
        battery_ratio = self.battery / max(self.battery_max, 1)
        if battery_ratio < Config.LOW_BATTERY_RATIO:
            if self.nearest_charger_dist < self.last_nearest_charger_dist:
                charger_reward += 0.05
            elif self.nearest_charger_dist > self.last_nearest_charger_dist:
                charger_reward -= 0.03
        if self.charged_last_step > 0:
            charger_reward += 0.1

        npc_penalty = 0.0
        if self.nearest_npc_dist <= 2.0:
            npc_penalty = -0.08
        elif self.nearest_npc_dist <= Config.NPC_DANGER_DISTANCE:
            npc_penalty = -0.02

        exploration_reward = 0.01 if self.current_cell_visit_count == 1 else 0.0

        # Anti-oscillation penalty (prevent back-and-forth without cleaning)
        # 防止反复横跳（当前动作与上一动作相反，且未清扫到污渍）
        oscillation_penalty = 0.0
        if self.last_action is not None and self.current_action is not None and self.cleaned_this_step == 0:
            if abs(self.current_action - self.last_action) == 4 and self.current_action < 8: # opposite directions in 8-way (0 vs 4, 1 vs 5, etc)
                oscillation_penalty = -0.01
                
        # Combo cleaning multiplier
        # 连续清扫加成
        if self.cleaned_this_step > 0:
            self.clean_combo += 1
            cleaning_reward += self.clean_combo * 0.1 # Bonus for combo
        else:
            self.clean_combo = 0

        return cleaning_reward + step_penalty + stuck_penalty + revisit_penalty + charger_reward + npc_penalty + exploration_reward + oscillation_penalty
