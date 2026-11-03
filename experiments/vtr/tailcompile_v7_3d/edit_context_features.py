"""012 frozen local representations of an already validated 011 edit bank.

No actions are generated or changed here. Features describe the edited nodes,
their one-hop boundary, and the source/destination regions. Teacher, measured
route and QoR fields are never read. The caller retains 011's full legality and
provenance checks; this extractor checks shapes, values and the sparse budget
without repeating the dense logical-node x region compatibility calculation.
"""
from __future__ import annotations

import math
import torch

FEATURE_SCHEMA = 'tailcompile-v7-local-edit-features-012-1'
MAX_EDITED_NODES = 256
EMBEDDING_BLOCKS = (
    'edited_logic_mean', 'edited_logic_std',
    'boundary_neighbor_logic_mean', 'boundary_neighbor_logic_std',
    'source_region_mean', 'destination_region_mean',
    'destination_minus_source_region_mean', 'edited_logic_times_region_delta',
    'boundary_neighbor_region_mean', 'incident_region_distance_delta',
)
RESOURCE_NAMES = ('lut', 'ff', 'memory', 'dsp', 'io')
STRUCTURAL_NAMES = ('edited_fraction', 'edited_log_fraction', 'xy_edited_fraction',
    'layer_edited_fraction', 'region_edited_fraction', 'boundary_neighbor_fraction',
    'incident_edge_fraction', 'edited_isolated_fraction')
GEOMETRY_NAMES = ('base_x_mean', 'base_y_mean', 'target_x_mean', 'target_y_mean',
    'delta_x_mean', 'delta_y_mean', 'abs_delta_x_mean', 'abs_delta_y_mean',
    'delta_x_std', 'delta_y_std', 'max_abs_delta_x', 'max_abs_delta_y',
    'source_layer_mean', 'target_layer_mean', 'layer_delta_mean',
    'source_leaf_x_span', 'source_leaf_y_span', 'target_leaf_x_span', 'target_leaf_y_span')
RESOURCE_BLOCKS = ('edited_resource_fraction', 'edited_mean_scaled_demand',
    'source_leaf_utilization', 'destination_leaf_utilization',
    'destination_leaf_free_fraction', 'destination_minus_source_capacity',
    'boundary_neighbor_resource_fraction')
EDGE_NAMES = ('incident_weight_fraction', 'incident_critical_weight_fraction',
    'incident_mean_criticality', 'internal_edited_edge_weight_fraction',
    'incoming_boundary_weight_fraction', 'outgoing_boundary_weight_fraction',
    'base_incident_xy_length', 'target_incident_xy_length', 'incident_xy_length_delta',
    'base_critical_xy_length', 'target_critical_xy_length', 'critical_xy_length_delta',
    'base_cross_layer_fraction', 'target_cross_layer_fraction', 'cross_layer_fraction_delta')
SCALAR_FEATURE_NAMES = (STRUCTURAL_NAMES + GEOMETRY_NAMES +
    tuple(f'{block}_{resource}' for block in RESOURCE_BLOCKS for resource in RESOURCE_NAMES) + EDGE_NAMES)


def edit_feature_dim(hidden):
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
        raise ValueError('012 embedding width must be a positive integer')
    return len(EMBEDDING_BLOCKS)*hidden+len(SCALAR_FEATURE_NAMES)


def _float(value):
    raw = torch.as_tensor(value)
    if raw.is_complex():
        raise ValueError('012 local features require real inputs')
    value = raw.detach().to(device='cpu', dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError('012 local features require finite inputs')
    return value


def _index(value, size, name):
    raw = torch.as_tensor(value).detach().cpu()
    if raw.dtype == torch.bool or raw.is_complex() or not torch.isfinite(raw).all():
        raise ValueError('012 '+name+' must be finite integer indices')
    if raw.is_floating_point() and (raw != raw.floor()).any():
        raise ValueError('012 '+name+' must be integer indices')
    result = raw.long()
    if result.ndim != 1 or (result < 0).any() or (result >= size).any():
        raise ValueError('012 '+name+' index shape/range mismatch')
    return result


def _action(action, n, regions, region_layer):
    assignment = _index(action['assignment'], regions, 'assignment')
    layers = _index(action['layer'], int(region_layer.max())+1, 'layer')
    xy = _float(action['xy'])
    if assignment.shape != (n,) or layers.shape != (n,) or xy.shape != (n, 2):
        raise ValueError('012 action must cover every APBlock')
    if not torch.equal(layers, region_layer[assignment]):
        raise ValueError('012 action region/layer disagreement')
    if (xy < 0).any() or (xy >= 1).any():
        raise ValueError('012 action coordinates must be normalized [0,1)')
    return assignment, layers, xy


def _moments(values, hidden):
    if len(values) == 0:
        zero = torch.zeros(hidden)
        return zero, zero.clone()
    return values.mean(0), values.std(0, unbiased=False)


def _weighted(values, weight):
    denominator = weight.sum()
    if len(weight) == 0 or float(denominator) <= 0:
        return values.new_zeros(values.shape[1:] or ())
    if values.ndim == 1:
        return (values*weight).sum()/denominator
    return (values*weight[:, None]).sum(0)/denominator


@torch.no_grad()
def local_features(data, base_action, candidates, logic_embeddings, region_embeddings):
    """Return detached CPU float32 [K, 10*hidden+77], with identity all zero.

    At most three candidates and 256 edited APBlocks per candidate are accepted.
    A candidate's graph work is one vectorized incident-edge pass, never a dense
    adjacency or logical-node x region tensor. One-hop boundary nodes are pooled
    once each, independently of node or region numbering. Embeddings are frozen
    representations supplied by D, not labels or newly trainable parameters.
    """
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 3:
        raise ValueError('012 expects a finite bank of one to three candidates')
    demand, capacity = _float(data['demand']), _float(data['capacity'])
    if demand.ndim != 2 or demand.shape[1] != 5 or capacity.ndim != 2 or capacity.shape[1] != 5:
        raise ValueError('012 expects five resource types')
    n, r = len(demand), len(capacity)
    if min(n, r) == 0 or (demand < 0).any() or (capacity < 0).any():
        raise ValueError('012 requires nonempty graphs and nonnegative resources')
    logic, region = _float(logic_embeddings), _float(region_embeddings)
    if logic.ndim != 2 or region.ndim != 2 or logic.shape[0] != n or region.shape[0] != r or logic.shape[1] != region.shape[1]:
        raise ValueError('012 frozen logical/device embedding dimensions disagree')
    hidden = logic.shape[1]
    dimension = edit_feature_dim(hidden)
    logic, region = logic.tanh(), region.tanh()
    raw_layers = torch.as_tensor(data['region_layer'])
    if raw_layers.ndim != 1 or len(raw_layers) != r:
        raise ValueError('012 requires one layer per region')
    region_layer = _index(raw_layers, int(raw_layers.max())+1, 'region_layer')
    bounds = _float(data['region_bounds'])
    if bounds.shape != (r, 4) or (bounds[:, [0, 2]] > bounds[:, [1, 3]]).any():
        raise ValueError('012 invalid region bounds')
    base_leaf, base_layer, base_xy = _action(base_action, n, r, region_layer)
    src = _index(data['logic_src'], n, 'logic_src')
    dst = _index(data['logic_dst'], n, 'logic_dst')
    weight = _float(data['logic_weight'])
    critical = _float(data.get('logic_criticality', torch.zeros(len(src))))
    if src.shape != dst.shape or weight.shape != src.shape or critical.shape != src.shape:
        raise ValueError('012 logical edge dimensions disagree')
    if (weight < 0).any() or (critical < 0).any() or (critical > 1).any():
        raise ValueError('012 edge weights/criticality outside the supported domain')
    total_demand = demand.sum(0).clamp_min(1.)
    demand_scale = demand.amax(0).log1p().clamp_min(1.)
    cap_scale = capacity.amax(0).clamp_min(1.)
    layer_scale = max(int(region_layer.max()), 1)
    usage = torch.zeros_like(capacity)
    usage.index_add_(0, base_leaf, demand)
    degree = torch.zeros(n)
    degree.index_add_(0, src, weight); degree.index_add_(0, dst, weight)
    features = []
    for candidate in candidates:
        leaf, layer, xy = _action(candidate, n, r, region_layer)
        region_changed = leaf != base_leaf
        layer_changed = layer != base_layer
        xy_changed = (xy != base_xy).any(1)
        changed = region_changed | layer_changed | xy_changed
        ids = changed.nonzero().flatten()
        if len(ids) > MAX_EDITED_NODES:
            raise ValueError('012 descriptor exceeds the frozen 256-node edit budget')
        if len(ids) == 0:
            features.append(torch.zeros(dimension))
            continue
        # Boundary is a set of unedited one-hop nodes. Parallel and hypergraph
        # expansion edges may repeat nodes, but never overweight their pooling.
        incident = (changed[src] | changed[dst]) & (weight > 0)
        isrc, idst = src[incident], dst[incident]
        iw, ic = weight[incident], critical[incident]
        neighbors = torch.unique(torch.cat((isrc[~changed[isrc]], idst[~changed[idst]])))
        moved_mean, moved_std = _moments(logic[ids], hidden)
        neighbor_mean, neighbor_std = _moments(logic[neighbors], hidden)
        old_region, new_region = region[base_leaf[ids]], region[leaf[ids]]
        delta_region = new_region-old_region
        edge_region_delta = ((region[leaf[isrc]]-region[leaf[idst]]).abs()-
                             (region[base_leaf[isrc]]-region[base_leaf[idst]]).abs())
        embedding_blocks = (moved_mean, moved_std, neighbor_mean, neighbor_std,
            old_region.mean(0), new_region.mean(0), delta_region.mean(0),
            (logic[ids]*delta_region).mean(0),
            region[base_leaf[neighbors]].mean(0) if len(neighbors) else torch.zeros(hidden),
            _weighted(edge_region_delta, iw*(1+ic)))
        delta_xy = xy[ids]-base_xy[ids]
        source_spans = bounds[base_leaf[ids]][:, [1, 3]]-bounds[base_leaf[ids]][:, [0, 2]]
        target_spans = bounds[leaf[ids]][:, [1, 3]]-bounds[leaf[ids]][:, [0, 2]]
        scalars = torch.tensor([len(ids)/n, math.log1p(len(ids))/math.log1p(n),
            float(xy_changed.sum())/n, float(layer_changed.sum())/n, float(region_changed.sum())/n,
            len(neighbors)/n, int(incident.sum())/max(len(src), 1), float((degree[ids] == 0).float().mean())])
        geometry = torch.cat((base_xy[ids].mean(0), xy[ids].mean(0), delta_xy.mean(0),
            delta_xy.abs().mean(0), delta_xy.std(0, unbiased=False), delta_xy.abs().amax(0),
            torch.tensor([float(base_layer[ids].float().mean())/layer_scale,
                          float(layer[ids].float().mean())/layer_scale,
                          float((layer[ids]-base_layer[ids]).float().mean())/layer_scale]),
            source_spans.mean(0), target_spans.mean(0)))
        after_usage = usage.clone()
        if region_changed.any():
            moved = region_changed.nonzero().flatten()
            after_usage.index_add_(0, base_leaf[moved], -demand[moved])
            after_usage.index_add_(0, leaf[moved], demand[moved])
        old_cap, new_cap = capacity[base_leaf[ids]], capacity[leaf[ids]]
        resource = torch.cat((demand[ids].sum(0)/total_demand,
            demand[ids].log1p().mean(0)/demand_scale,
            (usage[base_leaf[ids]]/old_cap.clamp_min(1.)).mean(0),
            (after_usage[leaf[ids]]/new_cap.clamp_min(1.)).mean(0),
            ((new_cap-after_usage[leaf[ids]]).clamp_min(0)/new_cap.clamp_min(1.)).mean(0),
            ((new_cap-old_cap)/cap_scale).mean(0),
            demand[neighbors].sum(0)/total_demand))
        before_length = (base_xy[isrc]-base_xy[idst]).abs().sum(1)
        after_length = (xy[isrc]-xy[idst]).abs().sum(1)
        bxy, axy = _weighted(before_length, iw), _weighted(after_length, iw)
        bcxy, acxy = _weighted(before_length, iw*ic), _weighted(after_length, iw*ic)
        bcut = _weighted((base_layer[isrc] != base_layer[idst]).float(), iw)
        acut = _weighted((layer[isrc] != layer[idst]).float(), iw)
        incoming = (~changed[isrc]) & changed[idst]
        outgoing = changed[isrc] & (~changed[idst])
        edge = torch.tensor([float(iw.sum()/weight.sum().clamp_min(1e-12)),
            float((iw*ic).sum()/(weight*critical).sum().clamp_min(1e-12)),
            float(_weighted(ic, iw)), float(_weighted((changed[isrc]&changed[idst]).float(), iw)),
            float(_weighted(incoming.float(), iw)), float(_weighted(outgoing.float(), iw)),
            float(bxy), float(axy), float(axy-bxy), float(bcxy), float(acxy), float(acxy-bcxy),
            float(bcut), float(acut), float(acut-bcut)])
        row = torch.cat((*embedding_blocks, scalars, geometry, resource, edge)).tanh()
        if row.shape != (dimension,) or not torch.isfinite(row).all():
            raise ValueError('012 local feature dimension/nonfinite failure')
        features.append(row)
    return torch.stack(features).detach()
