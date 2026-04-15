#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Robot Vacuum.
清扫大作战监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("清扫大作战")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .end_group()
        .add_group(
            group_name="对局指标",
            group_name_en="episode",
        )
        .add_panel(
            name="局均得分",
            name_en="episode_score",
            type="line",
        )
        .add_metric(
            metrics_name="episode_score",
            expr="avg(episode_score{})",
        )
        .end_panel()
        .add_panel(
            name="存活步数",
            name_en="finished_steps",
            type="line",
        )
        .add_metric(
            metrics_name="finished_steps",
            expr="avg(finished_steps{})",
        )
        .end_panel()
        .add_panel(
            name="回充次数",
            name_en="charge_count",
            type="line",
        )
        .add_metric(
            metrics_name="charge_count",
            expr="avg(charge_count{})",
        )
        .end_panel()
        .add_panel(
            name="剩余电量",
            name_en="remaining_charge",
            type="line",
        )
        .add_metric(
            metrics_name="remaining_charge",
            expr="avg(remaining_charge{})",
        )
        .end_panel()
        .add_panel(
            name="清扫比例",
            name_en="cleaning_ratio",
            type="line",
        )
        .add_metric(
            metrics_name="cleaning_ratio",
            expr="avg(cleaning_ratio{})",
        )
        .end_panel()
        .add_panel(
            name="失败率",
            name_en="fail_rate",
            type="line",
        )
        .add_metric(
            metrics_name="fail_rate",
            expr="avg(fail_rate{})",
        )
        .end_panel()
        .add_panel(
            name="重访率",
            name_en="revisit_rate",
            type="line",
        )
        .add_metric(
            metrics_name="revisit_rate",
            expr="avg(revisit_rate{})",
        )
        .end_panel()
        .add_panel(
            name="卡住率",
            name_en="blocked_rate",
            type="line",
        )
        .add_metric(
            metrics_name="blocked_rate",
            expr="avg(blocked_rate{})",
        )
        .end_panel()
        .add_panel(
            name="模型重载率",
            name_en="model_reload",
            type="line",
        )
        .add_metric(
            metrics_name="model_reload",
            expr="avg(model_reload{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
