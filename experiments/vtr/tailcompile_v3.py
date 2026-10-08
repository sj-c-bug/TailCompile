#!/usr/bin/env python3
"""TailCompile v3 minimal learnable cluster-to-region placement loop.

The model is intentionally untrained in this first milestone.  It proves that
the two graph towers, exact capacity decoder, intra-region coordinate head,
and VPR-consumable fplace lowering form one executable path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import networkx as nx
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tailcompile_v0 as v0
import tailcompile_v2 as v2
from tailrisk_core import parse_blif


RESOURCE_TYPES = ('clb', 'memory', 'dsp_top')
RESOURCE_INDEX = {'clb': 0, 'memory': 1, 'dsp_top': 2, 'io': 3}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def zscore_columns(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (tensor - mean) / std


def normalized_channel_width(channel_width: int) -> float:
    if channel_width <= 0:
        raise ValueError('channel_width must be positive')
    return math.log1p(channel_width) / math.log1p(300.0)


def graph_inputs(logic: dict, device: dict, blif: Path,
                 channel_width: int | None = None):
    if channel_width is not None and channel_width <= 0:
        raise ValueError('channel_width must be positive')
    primitives, _ = parse_blif(blif)
    atom_to_cluster = {}
    for cluster in logic['clusters']:
        for atom in cluster['atoms']:
            atom_to_cluster[atom] = int(cluster['id'])
    edges, _ = v2.contract_edges(primitives, atom_to_cluster)
    n = len(logic['clusters'])
    degree = [0.0] * n
    critical = [0.0] * n
    tail_samples = [[] for _ in range(n)]
    directed = defaultdict(float)
    for edge in edges:
        pins = list(edge.pins)
        for node in pins:
            degree[node] += edge.weight
            critical[node] += edge.weight * edge.criticality
            tail_samples[node].append(edge.weight)
        drivers = list(edge.drivers) or pins[:1]
        sinks = list(edge.sinks) or pins[1:]
        for source in drivers:
            for target in sinks:
                if source != target:
                    directed[source, target] += edge.weight
                    directed[target, source] += edge.weight
    cluster_features = []
    for cluster in logic['clusters']:
        node = int(cluster['id'])
        one_hot = [0.0] * 4
        one_hot[RESOURCE_INDEX.get(cluster['resource'], 3)] = 1.0
        samples = sorted(tail_samples[node])
        p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))] if samples else 0.0
        cluster_features.append(one_hot + [
            math.log1p(len(cluster['atoms'])), math.log1p(degree[node]),
            math.log1p(critical[node]), math.log1p(p95),
        ])
    if directed:
        logic_src, logic_dst, logic_weight = zip(*(
            (source, target, weight) for (source, target), weight in directed.items()))
    else:
        logic_src, logic_dst, logic_weight = (), (), ()

    region_features = []
    for region in device['regions']:
        region_features.append([
            math.log1p(region['tile_counts'].get('clb', 0)),
            math.log1p(region['tile_counts'].get('memory', 0)),
            math.log1p(region['tile_counts'].get('dsp_top', 0)),
            math.log1p(region['channel_internal']),
            math.log1p(region['escape_left']), math.log1p(region['escape_right']),
            math.log1p(region['escape_down']), math.log1p(region['escape_up']),
            math.log1p(region['segment_supply'].get('L4', 0.0)),
            math.log1p(region['segment_supply'].get('L16', 0.0)),
            region['center_x'] / max(device['grid_width'] - 1, 1),
            region['center_y'] / max(device['grid_height'] - 1, 1),
        ])
    region_edges = device['inter_region_rr_edge_counts_per_channel_width']
    region_directed = []
    for edge in region_edges:
        a, b, weight = int(edge['region_a']), int(edge['region_b']), float(edge['count'])
        region_directed.extend(((a, b, weight), (b, a, weight)))
    region_x = zscore_columns(torch.tensor(region_features, dtype=torch.float32))
    if channel_width is not None:
        # Width is a physical routing-capacity condition, not a random run ID.
        # A fixed reference keeps the feature meaningful across designs.  It is
        # appended after per-design z-scoring so the constant is not erased.
        width_feature = normalized_channel_width(channel_width)
        width_column = torch.full((region_x.shape[0], 1), width_feature,
                                  dtype=region_x.dtype)
        region_x = torch.cat((region_x, width_column), dim=1)
    return {
        'cluster_x': zscore_columns(torch.tensor(cluster_features, dtype=torch.float32)),
        'logic_src': torch.tensor(logic_src, dtype=torch.long),
        'logic_dst': torch.tensor(logic_dst, dtype=torch.long),
        'logic_weight': torch.tensor(logic_weight, dtype=torch.float32),
        'region_x': region_x,
        'region_src': torch.tensor([row[0] for row in region_directed], dtype=torch.long),
        'region_dst': torch.tensor([row[1] for row in region_directed], dtype=torch.long),
        'region_weight': torch.tensor([row[2] for row in region_directed], dtype=torch.float32),
        'contracted_edge_count': len(edges),
        'message_edge_count': len(directed),
    }


class GraphLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_linear = nn.Linear(width, width)
        self.neighbor_linear = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, src, dst, weight):
        aggregate = torch.zeros_like(x)
        # Allocate temporary state beside the input features.  Creating this
        # with torch.zeros(...) implicitly placed it on CPU and broke CUDA
        # training when dst/scaled lived on the GPU.
        denominator = x.new_zeros((x.shape[0], 1))
        if src.numel():
            scaled = weight / weight.mean().clamp_min(1e-6)
            aggregate.index_add_(0, dst, x[src] * scaled[:, None])
            denominator.index_add_(0, dst, scaled[:, None])
        neighbor = aggregate / denominator.clamp_min(1.0)
        return torch.relu(self.norm(self.self_linear(x) + self.neighbor_linear(neighbor)))


class Tower(nn.Module):
    def __init__(self, input_width: int, hidden: int):
        super().__init__()
        self.input = nn.Linear(input_width, hidden)
        self.layers = nn.ModuleList([GraphLayer(hidden), GraphLayer(hidden)])

    def forward(self, x, src, dst, weight):
        x = torch.relu(self.input(x))
        for layer in self.layers:
            x = layer(x, src, dst, weight)
        return x


class TwoTowerPlacementGNN(nn.Module):
    def __init__(self, cluster_width: int, region_width: int, hidden: int = 32):
        super().__init__()
        self.logic_tower = Tower(cluster_width, hidden)
        self.rrg_tower = Tower(region_width, hidden)
        self.logic_projection = nn.Linear(hidden, hidden, bias=False)
        self.region_projection = nn.Linear(hidden, hidden, bias=False)
        self.coordinate_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                             nn.Linear(hidden, 2), nn.Tanh())

    def forward(self, data):
        logic = self.logic_tower(data['cluster_x'], data['logic_src'], data['logic_dst'],
                                 data['logic_weight'])
        region = self.rrg_tower(data['region_x'], data['region_src'], data['region_dst'],
                                data['region_weight'])
        scores = self.logic_projection(logic) @ self.region_projection(region).T
        scores = scores / math.sqrt(logic.shape[1])
        return scores, self.coordinate_head(logic)


def add_partition_prior(scores: torch.Tensor, coordinates: torch.Tensor, logic: dict,
                        device: dict, weight: float) -> torch.Tensor:
    if weight == 0:
        return scores
    columns = int(device['region_grid']['columns'])
    rows = int(device['region_grid']['rows'])
    region_xy = [(int(region['column']), int(region['row'])) for region in device['regions']]
    prior = torch.zeros_like(scores)
    for cluster in logic['clusters']:
        node, group = int(cluster['id']), int(cluster['group'])
        gx, gy = group % columns, group // columns
        for region, (rx, ry) in enumerate(region_xy):
            prior[node, region] = -(abs(gx - rx) / max(columns - 1, 1)
                                    + abs(gy - ry) / max(rows - 1, 1))
    # The coordinate head is relative to a region; the prior stabilizes the
    # untrained model around the deterministic partition without fixing it.
    return scores + weight * prior


def min_cost_decode(scores: torch.Tensor, logic: dict, device: dict) -> tuple[list[int], dict]:
    assignments = [-1] * len(logic['clusters'])
    audit = {'resources': {}}
    score_rows = scores.detach().cpu().tolist()
    for resource in RESOURCE_TYPES:
        cluster_ids = [int(row['id']) for row in logic['clusters'] if row['resource'] == resource]
        capacities = [int(region['tile_counts'].get(resource, 0)) for region in device['regions']]
        targets = v2.allocate_targets(len(cluster_ids), capacities)
        graph = nx.DiGraph()
        for cluster_id in cluster_ids:
            graph.add_node(f'c{cluster_id}', demand=-1)
        for region, target in enumerate(targets):
            graph.add_node(f'r{region}', demand=target)
        for cluster_id in cluster_ids:
            for region in range(len(targets)):
                graph.add_edge(f'c{cluster_id}', f'r{region}', capacity=1,
                               weight=int(round(-score_rows[cluster_id][region] * 1_000_000)))
        cost, flow = nx.network_simplex(graph) if cluster_ids else (0, {})
        counts = [0] * len(targets)
        for cluster_id in cluster_ids:
            chosen = next(region for region in range(len(targets))
                          if flow[f'c{cluster_id}'][f'r{region}'] == 1)
            assignments[cluster_id] = chosen
            counts[chosen] += 1
        audit['resources'][resource] = {
            'demand': len(cluster_ids), 'capacities': capacities, 'targets': targets,
            'assigned': counts, 'integer_min_cost': cost,
        }
    io_counts = [0] * len(device['regions'])
    for cluster in logic['clusters']:
        cluster_id = int(cluster['id'])
        if cluster['resource'] != 'io':
            continue
        chosen = max(range(len(device['regions'])),
                     key=lambda region: (score_rows[cluster_id][region] - 0.01 * io_counts[region],
                                         -region))
        assignments[cluster_id] = chosen
        io_counts[chosen] += 1
    audit['io_assigned'] = io_counts
    if any(region < 0 for region in assignments):
        raise AssertionError('decoder left clusters unassigned')
    return assignments, audit


def lower_fplace(logic: dict, device: dict, assignments: list[int], relative_xy: torch.Tensor,
                 path: Path) -> dict:
    desired = relative_xy.detach().cpu().tolist()
    placed = {}
    buckets = defaultdict(list)
    for cluster in logic['clusters']:
        cluster_id, resource = int(cluster['id']), cluster['resource']
        if resource != 'io':
            buckets[assignments[cluster_id], resource].append(cluster_id)
    for (region_id, resource), cluster_ids in buckets.items():
        region = device['regions'][region_id]
        available = [tuple(tile) for tile in region['tile_coordinates'].get(resource, [])]
        if len(cluster_ids) > len(available):
            raise ValueError(f'capacity overflow region={region_id} resource={resource}')
        pending = sorted(cluster_ids, key=lambda node: -(abs(desired[node][0]) + abs(desired[node][1])))
        for node in pending:
            tx = region['center_x'] + desired[node][0] * (region['x_high'] - region['x_low']) / 2
            ty = region['center_y'] + desired[node][1] * (region['y_high'] - region['y_low']) / 2
            chosen = min(range(len(available)),
                         key=lambda index: ((available[index][0] - tx) ** 2
                                            + (available[index][1] - ty) ** 2, available[index]))
            placed[node] = (*available.pop(chosen), 0)
    io_tiles = [tuple(tile) for tile in device['perimeter_tiles']['io']]
    io_index = 0
    lines = ['# TailCompile v3 GNN cluster-level flat placement',
             '# <atom_name> <x> <y> <layer> <atom_sub_tile>']
    records = []
    for cluster in sorted(logic['clusters'], key=lambda row: int(row['id'])):
        node, resource = int(cluster['id']), cluster['resource']
        if resource == 'io':
            x, y = io_tiles[(io_index // 4) % len(io_tiles)]
            subtile = io_index % 4
            io_index += 1
        else:
            x, y, subtile = placed[node]
        for atom in cluster['atoms']:
            lines.append(f'{atom} {x} {y} 0 {subtile}')
        records.append({'cluster': node, 'region': assignments[node], 'resource': resource,
                        'x': x, 'y': y, 'subtile': subtile,
                        'relative_xy': desired[node], 'atom_count': len(cluster['atoms'])})
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    return {'schema': 'tailcompile-v3-lowering', 'fplace': str(path),
            'sha256': digest(path), 'cluster_count': len(records),
            'atom_count': sum(row['atom_count'] for row in records), 'clusters': records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logic-ir', type=Path, required=True)
    parser.add_argument('--device-ir', type=Path, required=True)
    parser.add_argument('--blif', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidates', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--partition-prior-weight', type=float, default=0.35)
    parser.add_argument('--channel-width', type=int,
                        help='Target route channel width for a width-conditioned checkpoint')
    parser.add_argument('--checkpoint', type=Path,
                        help='Trained structured-ranker checkpoint; omit for random candidates')
    parser.add_argument('--score-noise-std', type=float, default=0.0,
                        help='Gaussian score perturbation used to sample candidates')
    parser.add_argument('--coordinate-noise-std', type=float, default=0.0,
                        help='Gaussian relative-coordinate perturbation used to sample candidates')
    args = parser.parse_args()
    if args.candidates < 1:
        parser.error('--candidates must be positive')
    if args.channel_width is not None and args.channel_width <= 0:
        parser.error('--channel-width must be positive')
    if args.score_noise_std < 0 or args.coordinate_noise_std < 0:
        parser.error('candidate noise standard deviations must be nonnegative')
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f'missing checkpoint: {args.checkpoint}')
    if args.checkpoint is not None and args.candidates > 1 and not (
            args.score_noise_std or args.coordinate_noise_std):
        parser.error('multiple trained candidates require score or coordinate noise')
    logic, device = json.loads(args.logic_ir.read_text()), json.loads(args.device_ir.read_text())
    if int(device['region_grid']['count']) != 16 or int(logic['scale']) != 16:
        parser.error('v3 milestone currently requires matching 4x4 Logic and Device IR')
    blif = args.blif or Path(logic['mapped_blif'])
    if not blif.is_file():
        parser.error(f'missing BLIF: {blif}')
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    checkpoint_state = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        conditioned = bool(checkpoint.get('conditioned_on_channel_width', False))
        if conditioned and args.channel_width is None:
            parser.error('this checkpoint requires --channel-width')
        checkpoint_state = checkpoint.get('model_state', checkpoint)
    data = graph_inputs(logic, device, blif, args.channel_width)
    if checkpoint is not None:
        expected = checkpoint.get('region_feature_width')
        if expected is not None and int(expected) != int(data['region_x'].shape[1]):
            parser.error('checkpoint/device feature-width mismatch; check --channel-width')
    manifest = {
        'schema': 'tailcompile-v3-build', 'design': logic['design'], 'region_grid': '4x4',
        'status': ('trained_checkpoint_candidate_sampling' if args.checkpoint
                   else 'untrained_forward_pass_and_legal_decoder'),
        'logic_ir_sha256': digest(args.logic_ir), 'device_ir_sha256': digest(args.device_ir),
        'blif_sha256': digest(blif), 'hidden_width': args.hidden,
        'partition_prior_weight': args.partition_prior_weight,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'checkpoint_sha256': digest(args.checkpoint) if args.checkpoint else None,
        'score_noise_std': args.score_noise_std,
        'coordinate_noise_std': args.coordinate_noise_std,
        'channel_width': args.channel_width,
        'contracted_hyperedge_count': data['contracted_edge_count'],
        'message_edge_count': data['message_edge_count'], 'candidates': [],
    }
    for index in range(args.candidates):
        seed = args.seed + index
        random.seed(seed)
        torch.manual_seed(seed)
        model = TwoTowerPlacementGNN(data['cluster_x'].shape[1], data['region_x'].shape[1], args.hidden)
        if checkpoint_state is not None:
            model.load_state_dict(checkpoint_state)
        model.eval()
        with torch.no_grad():
            scores, relative_xy = model(data)
            scores = add_partition_prior(scores, relative_xy, logic, device,
                                         args.partition_prior_weight)
            if args.score_noise_std:
                scores = scores + torch.randn_like(scores) * args.score_noise_std
            if args.coordinate_noise_std:
                relative_xy = (relative_xy + torch.randn_like(relative_xy)
                               * args.coordinate_noise_std).clamp(-1.0, 1.0)
        assignments, decoder_audit = min_cost_decode(scores, logic, device)
        stem = f'candidate_{index:02d}_seed{seed}'
        lowering = lower_fplace(logic, device, assignments, relative_xy, args.out / f'{stem}.fplace')
        torch.save(model.state_dict(), args.out / f'{stem}.pt')
        binding = {
            'schema': 'tailcompile-v3-cluster-binding', 'seed': seed,
            'assignment': assignments, 'decoder': decoder_audit,
            'score_shape': list(scores.shape), 'score_min': float(scores.min()),
            'score_max': float(scores.max()), 'lowering_sha256': lowering['sha256'],
        }
        (args.out / f'{stem}_binding.json').write_text(json.dumps(binding, indent=2) + '\n')
        (args.out / f'{stem}_lowering.json').write_text(json.dumps(lowering, indent=2) + '\n')
        manifest['candidates'].append({
            'index': index, 'seed': seed, 'fplace': f'{stem}.fplace',
            'fplace_sha256': lowering['sha256'], 'model': f'{stem}.pt',
            'binding': f'{stem}_binding.json', 'lowering': f'{stem}_lowering.json',
            'region_assignment_sha256': hashlib.sha256(
                json.dumps(assignments, separators=(',', ':')).encode()).hexdigest(),
        })
    manifest['distinct_fplace_count'] = len({row['fplace_sha256'] for row in manifest['candidates']})
    manifest['distinct_region_assignment_count'] = len({
        row['region_assignment_sha256'] for row in manifest['candidates']})
    (args.out / 'build_summary.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
