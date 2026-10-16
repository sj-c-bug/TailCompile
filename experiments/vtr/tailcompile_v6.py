#!/usr/bin/env python3
"""TailCompile v6 routability models and spatial demand/capacity features.

The two model modes deliberately share all weights and heads.  ``early`` sees
only the packed logical graph, the regional RRG graph and channel width;
``placement`` additionally sees a candidate cluster-to-region assignment.
This makes the placement delta an information ablation instead of a change of
model capacity.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch import nn

from tailcompile_v3 import Tower
from tailcompile_v4_risk import cvar


QOR_NAMES = ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')
SPATIAL_REGION_WIDTH = 7
SPATIAL_GLOBAL_WIDTH = 9


def gini(values):
    ordered = sorted(max(0.0, float(value)) for value in values)
    total, count = sum(ordered), len(ordered)
    if not count or total == 0:
        return 0.0
    return sum((2 * index - count - 1) * value
               for index, value in enumerate(ordered, 1)) / (count * total)


def spatial_proxy(logic, device, evaluator, assignment):
    """Return assignment-conditioned regional demand/capacity features.

    RiskEvaluator routes every logical hyperedge over the same directed,
    segment-aware RRG supply pools used by v4/v5.  Pool risk is then projected
    back to its incident 4x4 regions.  This is a pre-route proxy, never a VPR
    label.
    """
    state = evaluator.state(list(map(int, assignment)))
    if state is None:
        raise ValueError('candidate assignment is infeasible in the regional RRG proxy')
    rho = [load / capacity for load, capacity in zip(state['loads'], evaluator.capacities)]
    incident = defaultdict(list)
    directional = defaultdict(lambda: [[], []])
    segmented = defaultdict(lambda: [[], []])
    for pool, value in zip(evaluator.supply['pools'], rho):
        regions = {int(region) for arc in pool['arcs'] for region in arc}
        axis = 0 if pool['kind'] == 'CHANX' else 1
        segment = str(pool['segment']).upper()
        segment_class = 0 if 'L4' in segment else 1 if 'L16' in segment else None
        for region in regions:
            incident[region].append(value)
            directional[region][axis].append(value)
            if segment_class is not None:
                segmented[region][segment_class].append(value)
    region_rows = []
    for region in range(len(device['regions'])):
        values = incident[region]
        x_values, y_values = directional[region]
        l4_values, l16_values = segmented[region]
        region_rows.append([
            sum(values) / len(values) if values else 0.0,
            max(values, default=0.0),
            cvar(values, 0.25) if values else 0.0,
            sum(x_values) / len(x_values) if x_values else 0.0,
            sum(y_values) / len(y_values) if y_values else 0.0,
            sum(l4_values) / len(l4_values) if l4_values else 0.0,
            sum(l16_values) / len(l16_values) if l16_values else 0.0,
        ])
    upper = max(1, math.ceil(0.10 * len(rho)))
    global_rows = [
        state['mean'], state['cvar'], state['max'], gini(rho),
        sum(sorted(rho, reverse=True)[:upper]) / max(sum(rho), 1e-12),
        statistics_mean([value for values in segmented.values() for value in values[0]]),
        statistics_mean([value for values in segmented.values() for value in values[1]]),
        math.log1p(state['wire'] / max(len(logic['clusters']), 1)),
        math.log1p(state['timing'] / max(len(logic['clusters']), 1)),
    ]
    return (torch.tensor(region_rows, dtype=torch.float32),
            torch.tensor(global_rows, dtype=torch.float32))


def statistics_mean(values):
    return sum(values) / len(values) if values else 0.0


def apply_tailrisk_ablation(data, use_tailrisk):
    """Zero only upper-tail fields while retaining mean demand and geometry."""
    result = dict(data)
    result['cluster_x'] = data['cluster_x'].clone()
    if not use_tailrisk:
        # v3 cluster feature 7 is local incident-net P95.
        result['cluster_x'][:, 7] = 0
        if 'spatial_region_x' in data:
            result['spatial_region_x'] = data['spatial_region_x'].clone()
            result['spatial_region_x'][:, 1:3] = 0
        if 'spatial_global_x' in data:
            result['spatial_global_x'] = data['spatial_global_x'].clone()
            result['spatial_global_x'][1:5] = 0
    return result


class RoutabilityGNN(nn.Module):
    """Shared two-tower predictor with an optional coarse-placement adapter."""

    def __init__(self, cluster_width, region_width, hidden=48):
        super().__init__()
        self.logic_tower = Tower(cluster_width, hidden)
        self.rrg_tower = Tower(region_width, hidden)
        # Both modes use this full input width. Early mode supplies zeros for
        # all placement fields, so parameter count is exactly matched.
        context_width = hidden * 4 + SPATIAL_GLOBAL_WIDTH
        self.context = nn.Sequential(nn.Linear(context_width, hidden), nn.ReLU(),
                                     nn.LayerNorm(hidden))
        self.route_head = nn.Linear(hidden, 1)
        self.qor_head = nn.Linear(hidden, len(QOR_NAMES))
        self.hotspot_head = nn.Sequential(nn.Linear(hidden + SPATIAL_REGION_WIDTH, hidden), nn.ReLU(),
                                          nn.Linear(hidden, 1))
        self.intervention_head = nn.Sequential(
            nn.Linear(hidden * 2 + SPATIAL_REGION_WIDTH, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, data, mode='early'):
        if mode not in ('early', 'placement'):
            raise ValueError('mode must be early or placement')
        logic = self.logic_tower(data['cluster_x'], data['logic_src'], data['logic_dst'],
                                 data['logic_weight'])
        region = self.rrg_tower(data['region_x'], data['region_src'], data['region_dst'],
                                data['region_weight'])
        logic_summary = torch.cat((logic.mean(0), logic.amax(0)))
        region_summary = torch.cat((region.mean(0), region.amax(0)))
        spatial_global = logic.new_zeros(SPATIAL_GLOBAL_WIDTH)
        spatial_region = logic.new_zeros((region.shape[0], SPATIAL_REGION_WIDTH))
        if mode == 'placement':
            spatial_global = data['spatial_global_x'].to(logic.device)
            spatial_region = data['spatial_region_x'].to(logic.device)
        context = self.context(torch.cat((logic_summary, region_summary, spatial_global)))
        hotspot = self.hotspot_head(torch.cat((region, spatial_region), dim=1)).squeeze(1)
        intervention = logic.new_zeros(logic.shape[0])
        if mode == 'placement':
            assignment = data['assignment'].to(logic.device)
            intervention = self.intervention_head(torch.cat(
                (logic, region[assignment], spatial_region[assignment]), dim=1)).squeeze(1)
        return {
            'route_logit': self.route_head(context).squeeze(0),
            'qor': self.qor_head(context),
            'hotspot': hotspot,
            'intervention_logit': intervention,
        }


def release_mask(intervention_logits, threshold=0.70, max_fraction=0.25):
    """Bind only confident clusters, with a hard cap on intervention scope."""
    if not 0 <= threshold <= 1 or not 0 < max_fraction <= 1:
        raise ValueError('invalid release threshold or max_fraction')
    probability = torch.sigmoid(intervention_logits)
    count = min(len(probability), max(1, math.ceil(len(probability) * max_fraction)))
    chosen = torch.topk(probability, count).indices
    mask = torch.zeros_like(probability, dtype=torch.bool)
    mask[chosen] = probability[chosen] >= threshold
    return mask


def move_graph(data, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in data.items()}
