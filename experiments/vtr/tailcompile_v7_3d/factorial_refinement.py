"""017: an isolated 2x2 objective / exchange experiment on one frozen D.

L is the 016 original-net HPWL proxy, including an actual access-compatible
CHANZ coordinate. T is a device-derived interconnect delay proxy: each known
driver net contributes risk_nets.weight * max(driver-to-sink delay). Same-layer
paths use Manhattan distance times that layer's delay/tile. Cross-layer paths
use one common portal, source/sink layer delays and directed vertical delay.
Unknown drivers have no invented timing path. This is not STA or routed CPD.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
import time

import numpy as np
import torch

try:
    from .residual_actions import validate_residual_action
    from .aligned_model import action_digest
except ImportError:
    from residual_actions import validate_residual_action
    from aligned_model import action_digest

SCHEMA = 'v8-017-factorial-action-1'
DEFAULT_CONFIG = dict(max_evaluations=8192, max_moves=64, max_passes=3,
    max_seconds=20., max_setup_seconds=60., trust_fraction=.25,
    max_work_units=20000000, max_setup_work_units=200000000,
    gain_absolute=1e-10, gain_relative=1e-9, max_candidate_nodes=256,
    max_swap_partners=8, baseline_batch_size=1024)
OBJECTIVE = 'L/L0 for wl; 0.5*L/L0 + 0.5*T/T0 for joint; frozen common D reference WL routing normalizers'
TIMING_SEMANTICS = ('sum over direction-known original nets of risk_nets.weight * '
    'maximum driver-to-sink interconnect delay in ns; per-layer Manhattan delay '
    'plus directed CHANZ delay and terminal_delay_ns; one portal per cross net; '
    'criticality is pre-placement unit-depth proxy, NOT STA')


class BudgetExceeded(RuntimeError):
    pass


class SetupBudgetExceeded(BudgetExceeded):
    """A complete reference is mandatory; callers must report setup failure."""


def _config(config):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError('017 unknown refinement options: ' + str(sorted(unknown)))
        cfg.update(config)
    ints = ('max_evaluations', 'max_moves', 'max_passes', 'max_candidate_nodes',
            'max_work_units', 'max_setup_work_units', 'max_swap_partners', 'baseline_batch_size')
    for key in ints:
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError('017 nonnegative integer required: ' + key)
    if cfg['max_swap_partners'] > 8 or cfg['baseline_batch_size'] < 1:
        raise ValueError('017 at most eight swap partners and positive batch size required')
    for key in ('max_seconds', 'max_setup_seconds', 'trust_fraction', 'gain_absolute', 'gain_relative'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('017 finite nonnegative option required: ' + key)
    if cfg['trust_fraction'] > 1 or cfg['gain_absolute'] <= 0:
        raise ValueError('017 trust_fraction <= 1 and positive gain_absolute required')
    return cfg


def _array(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _core(data, action):
    core = {k: action[k] for k in ('assignment', 'layer', 'xy')}
    audit = validate_residual_action(data, core)
    return dict(assignment=_array(core['assignment'], np.int64).copy(),
        layer=_array(core['layer'], np.int64).copy(),
        xy=_array(torch.as_tensor(core['xy']).to(dtype=torch.float32)).copy()), audit


class _Budget:
    def __init__(self, cfg, setup=False):
        self.started = time.perf_counter()
        self.seconds = cfg['max_setup_seconds' if setup else 'max_seconds']
        self.limit = cfg['max_setup_work_units' if setup else 'max_work_units']
        self.work = 0
        self.setup = setup

    def check(self, work=0):
        reason = None
        if self.work + work > self.limit:
            reason = 'work_budget'
        elif time.perf_counter() - self.started >= self.seconds:
            reason = 'time_budget'
        if reason:
            if self.setup:
                raise SetupBudgetExceeded('017 incomplete common setup: ' + reason + '; no baseline action fallback')
            raise BudgetExceeded(reason)
        self.work += work


class _PortalIndex:
    """Exact separable minimization on irregular portals, not P-by-net scans.

    For each x column only the legal y immediately below/above a convex
    function's minimum is relevant. Tables share the deduplicated coordinate
    grid. A Cartesian set permits independent coordinate minimization.
    """
    def __init__(self, points):
        self.points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
        self.x, self.y = (np.unique(self.points[:, i]) for i in (0, 1))
        self.product = len(self.points) == len(self.x) * len(self.y)
        if not self.product and len(self.points):
            xi = np.searchsorted(self.x, self.points[:, 0])
            yi = np.searchsorted(self.y, self.points[:, 1])
            occupied = np.zeros((len(self.x), len(self.y)), dtype=bool)
            occupied[xi, yi] = True
            ranks = np.arange(len(self.y))[None, :]
            below = np.maximum.accumulate(np.where(occupied, ranks, -1), axis=1)
            above = np.minimum.accumulate(np.where(occupied, ranks, len(self.y))[:, ::-1], axis=1)[:, ::-1]
            self.below = np.concatenate((np.full((len(self.x), 1), -1), below), axis=1)
            self.above = np.concatenate((above, np.full((len(self.x), 1), len(self.y))), axis=1)

    @staticmethod
    def _axis(values, pivot, function):
        rank = np.searchsorted(values, pivot, side='left')
        left = values[np.maximum(rank - 1, 0)]
        right = values[np.minimum(rank, len(values) - 1)]
        # Left wins ties, so all routing ties have a stable coordinate order.
        return np.where(function(left) <= function(right), left, right)

    def separable(self, pivot, fx, fy):
        """Batched exact portals; fx/fy accept B or B-by-columns arrays."""
        if self.product:
            return np.stack((self._axis(self.x, pivot[:, 0], fx),
                             self._axis(self.y, pivot[:, 1], fy)), axis=1)
        rank = np.searchsorted(self.y, pivot[:, 1], side='left')
        lower = self.below[:, rank].T
        upper = self.above[:, rank].T
        yl = self.y[np.clip(lower, 0, len(self.y)-1)]
        yu = self.y[np.clip(upper, 0, len(self.y)-1)]
        fl, fu = fy(yl), fy(yu)
        fl = np.where(lower >= 0, fl, np.inf)
        fu = np.where(upper < len(self.y), fu, np.inf)
        use_lower = fl <= fu
        ys = np.where(use_lower, yl, yu)
        score = fx(np.broadcast_to(self.x, ys.shape)) + np.minimum(fl, fu)
        col = np.argmin(score, axis=1)
        return np.stack((self.x[col], ys[np.arange(len(col)), col]), axis=1)

    def two(self, a, b, wa, wb):
        a, b = np.asarray(a), np.asarray(b)
        wa, wb = np.broadcast_to(wa, len(a)), np.broadcast_to(wb, len(a))
        pivot = np.where((wa > wb)[:, None], a,
                        np.where((wa < wb)[:, None], b, np.minimum(a, b)))
        def axis(k):
            def value(p):
                extra = (None,) if p.ndim == 2 else ()
                return wa[(slice(None),)+extra]*np.abs(p-a[(slice(None), k)+extra]) + wb[(slice(None),)+extra]*np.abs(p-b[(slice(None), k)+extra])
            return value
        return self.separable(pivot, axis(0), axis(1))

    def boxes(self, lo, hi):
        # HPWL expansions of both layer boxes. Median of the four breakpoints
        # gives the lower end of the convex function's minimum interval.
        pivot = np.sort(np.concatenate((lo, hi), axis=1), axis=1)[:, 1, :]
        def axis(k):
            def value(p):
                if p.ndim == 1:
                    return np.maximum(lo[:, :, k]-p[:, None], 0).sum(1) + np.maximum(p[:, None]-hi[:, :, k], 0).sum(1)
                return np.maximum(lo[:, None, :, k]-p[:, :, None], 0).sum(2) + np.maximum(p[:, :, None]-hi[:, None, :, k], 0).sum(2)
            return value
        return self.separable(pivot, axis(0), axis(1))


def _timing(model):
    if model is None:
        return None
    h = np.asarray(model['horizontal_delay_per_tile_ns'], dtype=float)
    v = np.asarray(model['vertical_delay_ns'], dtype=float)
    terminal = float(model['terminal_delay_ns'])
    if h.shape != (2,) or v.shape != (2, 2) or not np.isfinite(h).all() or not np.isfinite(v).all() or not math.isfinite(terminal) or (h < 0).any() or (v < 0).any() or terminal < 0:
        raise ValueError('017 invalid device timing model')
    return h, v, terminal


class _Problem:
    def __init__(self, data, base, summary, timing_model, cfg, budget):
        self.base = {k: v.copy() for k, v in base.items()}
        self.base_digest = action_digest(base)
        self.scale = np.asarray(data.get('grid', data.get('teacher_axis_scale', []))[:2], dtype=float)
        if self.scale.shape != (2,) or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError('017 explicit positive XY tile scale required')
        if set(_array(data['region_layer']).tolist()) != {0, 1}:
            raise ValueError('017 requires exactly two physical layers')
        if 'nets' not in data:
            raise ValueError('017 original net incidence required')
        self.timing = _timing(timing_model)
        self.timing_model = timing_model
        self.n = len(base['xy'])
        self.nets, self.drivers, self.sinks, self.weights, self.timing_known = [], [], [], [], []
        self.incidence = [[] for _ in range(self.n)]
        rows = {int(r['id']): r for r in data.get('risk_nets', [])}
        if len(rows) != len(data.get('risk_nets', [])):
            raise ValueError('017 duplicate directional net id')
        for ni, pins in enumerate(data['nets']):
            budget.check(len(pins)+1)
            raw = np.asarray(pins)
            if raw.ndim != 1 or (len(raw) and (raw.dtype.kind not in 'iu' or (raw < 0).any() or (raw >= self.n).any())):
                raise ValueError('017 incidence must contain in-range integer APBlock ids')
            ids = np.unique(raw).astype(np.int64)
            self.nets.append(ids)
            for node in ids:
                self.incidence[node].append(ni)
            row = rows.get(ni, {})
            if row and set(row['pins']) != set(ids.tolist()):
                raise ValueError('017 directional net differs from original incidence')
            drivers = row.get('drivers', [])
            driver = int(drivers[0]) if len(drivers) == 1 and int(drivers[0]) in ids else None
            sinks = np.asarray(row.get('sinks', []), dtype=np.int64)
            known = driver is not None and set(sinks.tolist()) == set(ids.tolist())-{driver}
            self.drivers.append(driver)
            self.timing_known.append(known)
            self.sinks.append(np.unique(sinks) if known else np.empty(0, dtype=np.int64))
            weight = float(row.get('weight', 1.+.5*float(row.get('criticality', 0.))))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('017 nonnegative finite risk weight required')
            self.weights.append(weight)
        self.weights = np.asarray(self.weights)
        self.profiles = []
        # Combine equivalent access profiles before any net queries. A profile
        # is region reachability, not a pin-level routing certificate.
        merged = defaultdict(list)
        self.raw_pool_count = len(summary.get('pools', [])) if summary else 0
        if summary is not None:
            grid = summary.get('grid', [])
            if len(grid) != 3 or grid[2] != 2 or not np.array_equal(grid[:2], self.scale):
                raise ValueError('017 vertical summary must match XY grid and two layers')
            if 'region_layer' in summary and not np.array_equal(summary['region_layer'], _array(data['region_layer'])):
                raise ValueError('017 vertical summary leaf ordering mismatch')
            for pool in summary.get('pools', []):
                points = []
                for member in pool.get('members', []):
                    budget.check(1)
                    capacity = float(member.get('capacity', 0))
                    p = np.asarray(member['xy'], dtype=float)
                    if not math.isfinite(capacity) or capacity < 0 or p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any() or (p >= 1).any():
                        raise ValueError('017 invalid actual CHANZ member')
                    if capacity:
                        points.append(p*self.scale)
                if not points:
                    continue
                points = np.unique(points, axis=0)
                for orientation in pool.get('orientations', []):
                    sl, tl = int(orientation['source_layer']), int(orientation['sink_layer'])
                    if (sl, tl) not in ((0, 1), (1, 0)):
                        raise ValueError('017 unsupported vertical orientation')
                    key = sl, frozenset(orientation['source_regions']), frozenset(orientation['sink_regions'])
                    merged[key].append(points)
        self.access_cache = {}
        self.coordinate_indices = {}
        for key, arrays in merged.items():
            budget.check(sum(len(a) for a in arrays))
            points = np.unique(np.concatenate(arrays), axis=0)
            self.profiles.append((key, self._index(points)))
        self.baseline = {}
        # Reference portals always minimize WL, for both objective arms.
        ref = self.costs(self.base, 'wl', None, budget, cfg)
        l0 = math.fsum(c[0] for c in ref if c is not None)
        t0 = math.fsum(c[1] for c in ref if c is not None)
        self.reference = (l0, t0)
        self.normalizers = dict(wirelength=l0 if l0 > 0 else 1., timing=t0 if t0 > 0 else 1.)
        self.baseline['wl'] = ref
        self.baseline['joint'] = self.costs(self.base, 'joint', self.normalizers, budget, cfg) if self.timing else ref
        self.unsupported = [i for i, c in enumerate(ref) if c is None]
        self.frozen = {int(node) for ni in self.unsupported for node in self.nets[ni]}
        self.priority = np.zeros(self.n)
        for ni, c in enumerate(ref):
            if c is not None:
                for node in self.nets[ni]:
                    self.priority[node] += c[0]
        self.setup_seconds = time.perf_counter()-budget.started
        self.setup_work = budget.work
        budget.check()

    def fork(self):
        # Every arm starts with the same warm reference caches. Candidate-only
        # cache additions cannot make a later arm's timed budget more generous.
        result = copy.copy(self)
        result.access_cache = dict(self.access_cache)
        result.coordinate_indices = dict(self.coordinate_indices)
        return result

    def _index(self, points):
        points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
        key = points.shape, points.tobytes()
        if key not in self.coordinate_indices:
            self.coordinate_indices[key] = _PortalIndex(points)
        return self.coordinate_indices[key]

    def entrances(self, source, source_regions, sink_regions, budget):
        key = source, tuple(sorted(source_regions)), tuple(sorted(sink_regions))
        if key not in self.access_cache:
            selected = []
            for (sl, sr, tr), index in self.profiles:
                budget.check(1)
                if sl == source and source_regions <= sr and sink_regions <= tr:
                    selected.append(index.points)
            self.access_cache[key] = self._index(np.concatenate(selected) if selected else np.empty((0, 2)))
        return self.access_cache[key]

    def portal(self, ni, state, budget):
        ids = self.nets[ni]
        z = state['layer'][ids]
        regions = [set(state['assignment'][ids[z == layer]].tolist()) for layer in (0, 1)]
        driver = self.drivers[ni]
        if driver is not None:
            source = int(state['layer'][driver])
            return self.entrances(source, regions[source], regions[1-source], budget)
        first = self.entrances(0, regions[0], regions[1], budget)
        second = self.entrances(1, regions[1], regions[0], budget)
        key = 'bidirectional', id(first), id(second)
        if key not in self.access_cache:
            other = set(map(tuple, second.points.tolist()))
            self.access_cache[key] = self._index([p for p in first.points if tuple(p) in other])
        return self.access_cache[key]

    def _at_portals(self, ni, state, points):
        ids = self.nets[ni]
        xy, z = state['xy'][ids]*self.scale, state['layer'][ids]
        length = np.zeros(len(points))
        for layer in (0, 1):
            pins = xy[z == layer]
            lo, hi = pins.min(0), pins.max(0)
            length += (hi-lo).sum()+np.maximum(lo-points, 0).sum(1)+np.maximum(points-hi, 0).sum(1)
        timing = np.zeros(len(points))
        driver = self.drivers[ni]
        if self.timing is not None and self.timing_known[ni]:
            h, v, terminal = self.timing
            source = int(state['layer'][driver])
            dxy = state['xy'][driver]*self.scale
            sinks = self.sinks[ni]
            for target in (0, 1):
                group = sinks[state['layer'][sinks] == target]
                if not len(group):
                    continue
                sxy = state['xy'][group]*self.scale
                if target == source:
                    delay = np.full(len(points), h[source]*np.abs(dxy-sxy).sum(1).max()+terminal)
                else:
                    # max_s ||p-s||_1 is the maximum of four diagonal
                    # support functions, exact in O(sinks + portals).
                    plus, minus = sxy[:, 0]+sxy[:, 1], sxy[:, 0]-sxy[:, 1]
                    px, py = points[:, 0], points[:, 1]
                    far = np.maximum.reduce((px+py-plus.min(), px-py-minus.min(),
                        -px+py+minus.max(), -px-py+plus.max()))
                    delay = h[source]*np.abs(points-dxy).sum(1)+h[target]*far+v[source, target]+terminal
                timing = np.maximum(timing, delay)
            timing *= self.weights[ni]
        return length, timing

    def cost(self, ni, state, mode, norms, budget):
        ids = self.nets[ni]
        budget.check(len(ids)+1)
        if len(ids) < 2:
            return (0., 0.)
        xy, z = state['xy'][ids]*self.scale, state['layer'][ids]
        if np.all(z == z[0]):
            length = float(np.ptp(xy, axis=0).sum())
            timing = 0.
            if self.timing is not None and self.timing_known[ni]:
                h, _, terminal = self.timing
                delta = np.abs(state['xy'][self.sinks[ni]]*self.scale-state['xy'][self.drivers[ni]]*self.scale).sum(1)
                timing = self.weights[ni]*(h[z[0]]*float(delta.max())+terminal) if len(delta) else 0.
            return length, timing
        index = self.portal(ni, state, budget)
        if not len(index.points):
            return None
        if len(ids) == 2 and self.drivers[ni] is not None:
            return self._two_batch([ni], state, mode, norms, index, budget)[0]
        if mode == 'wl' or not self.timing_known[ni] or self.timing is None:
            lo = np.array([[xy[z == l].min(0) for l in (0, 1)]])
            hi = np.array([[xy[z == l].max(0) for l in (0, 1)]])
            points = index.boxes(lo, hi)
            budget.check(len(index.x)*2)
        else:
            # General max-sink timing is not separable. Enumerate the actual
            # portals exactly; no surrogate weighted-HPWL substitution.
            points = index.points
            budget.check(len(points)+len(ids))
        length, timing = self._at_portals(ni, state, points)
        score = length if mode == 'wl' else .5*length/norms['wirelength']+.5*timing/norms['timing']
        best = int(np.argmin(score))
        return float(length[best]), float(timing[best])

    def _two_batch(self, net_ids, state, mode, norms, index, budget):
        ids = np.asarray(net_ids, dtype=np.int64)
        drivers = np.asarray([self.drivers[n] for n in ids], dtype=np.int64)
        sinks = np.asarray([next(int(pin) for pin in self.nets[n] if pin != self.drivers[n]) for n in ids], dtype=np.int64)
        a, b = state['xy'][drivers]*self.scale, state['xy'][sinks]*self.scale
        source, target = state['layer'][drivers], state['layer'][sinks]
        weight = self.weights[ids]
        wa, wb = np.ones(len(ids)), np.ones(len(ids))
        if mode == 'joint' and self.timing is not None:
            h, _, _ = self.timing
            timing_weight = weight*np.asarray([self.timing_known[n] for n in ids])
            wa = .5/norms['wirelength']+.5*timing_weight*h[source]/norms['timing']
            wb = .5/norms['wirelength']+.5*timing_weight*h[target]/norms['timing']
        budget.check(len(ids)*(2 if index.product else len(index.x)*2))
        p = index.two(a, b, wa, wb)
        da, db = np.abs(a-p).sum(1), np.abs(b-p).sum(1)
        length = da+db
        timing = np.zeros(len(ids))
        if self.timing is not None:
            h, v, terminal = self.timing
            timing = weight*(h[source]*da+h[target]*db+v[source, target]+terminal)*np.asarray([self.timing_known[n] for n in ids])
        return list(zip(length.tolist(), timing.tolist()))

    def costs(self, state, mode, norms, budget, cfg):
        values = [None]*len(self.nets)
        batches = defaultdict(list)
        # Batch common cross-layer two-pin nets. No repeated P*net work, and
        # no shared mutable geometry or cost state between experimental arms.
        for ni, ids in enumerate(self.nets):
            budget.check(1)
            if state is self.base and mode == 'joint' and 'wl' in self.baseline and len(ids) == 2 and self.timing is not None and self.timing[0][0] == self.timing[0][1]:
                # Equal horizontal coefficients make two-pin T an affine
                # function of L (directed vertical constant is unchanged).
                # This exact identity avoids a redundant complete pass.
                values[ni] = self.baseline['wl'][ni]
                continue
            if len(ids) == 2 and self.drivers[ni] is not None and state['layer'][ids[0]] != state['layer'][ids[1]]:
                index = self.portal(ni, state, budget)
                if len(index.points):
                    batches[id(index)].append((ni, index))
            else:
                values[ni] = self.cost(ni, state, mode, norms, budget)
        for rows in batches.values():
            index = rows[0][1]
            for start in range(0, len(rows), cfg['baseline_batch_size']):
                ids = [r[0] for r in rows[start:start+cfg['baseline_batch_size']]]
                costs = self._two_batch(ids, state, mode, norms, index, budget)
                for ni, value in zip(ids, costs):
                    values[ni] = value
        return values


def prepare_problem(data, base, vertical_summary=None, timing_model=None, config=None):
    """Complete immutable common reference, portal indices and fixed L0/T0."""
    cfg = _config(config)
    budget = _Budget(cfg, setup=True)
    budget.check()
    core, _ = _core(data, base)
    return _Problem(data, core, vertical_summary, timing_model, cfg, budget)


def _totals(costs, mode, norms):
    length = math.fsum(c[0] for c in costs if c is not None)
    timing = math.fsum(c[1] for c in costs if c is not None)
    f = length/norms['wirelength'] if mode == 'wl' else .5*length/norms['wirelength']+.5*timing/norms['timing']
    return length, timing, f


def objective(data, action, vertical_summary=None, timing_model=None, objective_mode='wl', config=None, problem=None):
    """Public exact recomputation; optionally keep a prepared D's normalizers."""
    cfg = _config(config)
    if objective_mode not in ('wl', 'joint'):
        raise ValueError('017 objective_mode must be wl or joint')
    core, _ = _core(data, action)
    prepared = (problem or prepare_problem(data, action, vertical_summary, timing_model, cfg)).fork()
    if objective_mode == 'joint' and (prepared.timing is None or not any(prepared.timing_known)):
        raise ValueError('017 joint objective requires a device timing model and at least one known driver/sink net')
    budget = _Budget(cfg, setup=True)
    costs = prepared.costs(core, objective_mode, prepared.normalizers, budget, cfg)
    length, timing, f = _totals(costs, objective_mode, prepared.normalizers)
    missing = [i for i, c in enumerate(costs) if c is None]
    return dict(value=f, wirelength=length, timing=timing, normalizers=dict(prepared.normalizers),
        net_costs=costs, unsupported_net_ids=missing, full_net_coverage=not missing,
        represented_net_count=len(costs)-len(missing), total_net_count=len(costs),
        objective_semantics=OBJECTIVE, timing_semantics=TIMING_SEMANTICS,
        timing_known_net_count=sum(prepared.timing_known))


def refine(data, base, vertical_summary=None, timing_model=None, objective_mode='wl', allow_swaps=False, config=None, problem=None):
    """Current-state strict deployed-coordinate descent; max_moves counts nodes."""
    cfg = _config(config)
    if objective_mode not in ('wl', 'joint'):
        raise ValueError('017 objective_mode must be wl or joint')
    prepared = (problem or prepare_problem(data, base, vertical_summary, timing_model, cfg)).fork()
    if action_digest(base) != prepared.base_digest:
        raise ValueError('017 prepared reference differs from frozen D')
    if objective_mode == 'joint' and (prepared.timing is None or not any(prepared.timing_known)):
        raise ValueError('017 joint objective requires a device timing model and at least one known driver/sink net')
    budget = _Budget(cfg)
    state = {k: v.copy() for k, v in prepared.base.items()}
    costs = list(prepared.baseline[objective_mode])
    length, timing, current = _totals(costs, objective_mode, prepared.normalizers)
    known = sum(prepared.timing_known)
    diag = dict(config=cfg, objective_mode=objective_mode, pair_swaps=bool(allow_swaps), setup_complete=True,
        setup_seconds=prepared.setup_seconds, setup_work_units=prepared.setup_work,
        initial_wirelength=length, final_wirelength=length, initial_timing=timing, final_timing=timing,
        initial_objective=current, final_objective=current, normalizers=dict(prepared.normalizers),
        reference_wirelength=prepared.reference[0], reference_timing=prepared.reference[1],
        history=[current], wirelength_history=[length], timing_history=[timing], accepted_moves=[],
        counters=dict(evaluations=0, accepted_moves=0, accepted_swaps=0, node_touches=0, passes=0, work_units=0),
        unsupported_net_ids=list(prepared.unsupported), frozen_unsupported_nodes=sorted(prepared.frozen),
        full_net_coverage=not prepared.unsupported, total_net_count=len(costs),
        represented_net_count=len(costs)-len(prepared.unsupported), timing_known_net_count=known,
        timing_known_fraction=known/len(costs) if costs else 1., direction_unknown_net_count=sum(d is None for d in prepared.drivers),
        timing_represented_net_count=sum(k and c is not None for k, c in zip(prepared.timing_known, costs)),
        timing_full_net_coverage=known == len(costs) and not prepared.unsupported and prepared.timing is not None,
        objective_semantics=OBJECTIVE, timing_semantics=TIMING_SEMANTICS,
        criticality_semantics=data.get('criticality_semantics', 'pre-placement unit-depth criticality proxy, NOT STA'),
        portal_semantics='one actual directed-access-compatible portal per net; region reachability upper bound, not pin-route certificate',
        raw_pool_count=prepared.raw_pool_count, merged_access_profile_count=len(prepared.profiles),
        unique_portal_coordinate_sets=len(prepared.coordinate_indices),
        candidate_rule='common frozen D incident-WL priority and original-D trust box; single 3x3 grid plus up to eight opposite-layer partners',
        move_budget_semantics='max_moves is accepted node touches; single=1, atomic exchange=2',
        physical_legality_claimed=False, timing_improvement_claimed=False, global_optimum_claimed=False,
        stationarity_certified=False, stop_reason=None)
    demand, capacity = _array(data['demand'], float), _array(data['capacity'], float)
    bounds, region_layer = _array(data['region_bounds'], float), _array(data['region_layer'], np.int64)
    movable = _array(data.get('movable', np.ones(prepared.n, dtype=bool)), bool)
    fixed_xy = _array(data.get('fixed_xy', np.full((prepared.n, 2), -1.)), float)
    fixed_layer = _array(data.get('fixed_layer', np.full(prepared.n, -1)), np.int64)
    fixed_region = _array(data.get('fixed_region', np.full(prepared.n, -1)), np.int64)
    legal = _array(data.get('legal_layers', np.ones((prepared.n, 2), dtype=bool)), bool)
    usage = np.zeros_like(capacity); np.add.at(usage, state['assignment'], demand)
    layer_capacity = np.zeros((2, 5)); np.add.at(layer_capacity, region_layer, capacity)
    layer_usage = np.zeros_like(layer_capacity); np.add.at(layer_usage, state['layer'], demand)
    spans = bounds[prepared.base['assignment']]
    radius = (spans[:, [1, 3]]-spans[:, [0, 2]])*cfg['trust_fraction']
    low = np.maximum(0, prepared.base['xy']-radius)
    high = np.minimum(np.nextafter(np.array(1, dtype=state['xy'].dtype), np.array(0, dtype=state['xy'].dtype)), prepared.base['xy']+radius)
    eligible = [i for i in range(prepared.n) if movable[i] and i not in prepared.frozen and prepared.incidence[i]]
    eligible_set = set(eligible)
    order = sorted(eligible, key=lambda i: (-prepared.priority[i], i))[:cfg['max_candidate_nodes']]
    rejects = Counter()

    def compatible(node, target):
        layer = region_layer[target]
        return (legal[node, layer] and (fixed_layer[node] < 0 or fixed_layer[node] == layer) and
            (fixed_region[node] < 0 or fixed_region[node] == target) and (demand[node] <= capacity[target]+1e-6).all())

    def box(node, target):
        lo = np.maximum(low[node], bounds[target, [0, 2]])
        hi = np.minimum(high[node], np.nextafter(bounds[target, [1, 3]].astype(state['xy'].dtype), np.array(-np.inf, dtype=state['xy'].dtype)))
        rounded = lo.astype(state['xy'].dtype)
        lo = np.where(rounded < lo, np.nextafter(rounded, np.array(np.inf, dtype=state['xy'].dtype)), rounded)
        rounded = hi.astype(state['xy'].dtype)
        hi = np.where(rounded > hi, np.nextafter(rounded, np.array(-np.inf, dtype=state['xy'].dtype)), rounded)
        for axis in (0, 1):
            if fixed_xy[node, axis] >= 0:
                lo[axis] = max(lo[axis], fixed_xy[node, axis])
                hi[axis] = min(hi[axis], fixed_xy[node, axis])
        return None if (lo > hi).any() else (lo, hi)

    def snapshot(nodes):
        return [dict(node=int(n), assignment=int(state['assignment'][n]), layer=int(state['layer'][n]), xy=state['xy'][n].tolist()) for n in nodes]

    def evaluate(nodes, targets, points):
        budget.check()
        if diag['counters']['evaluations'] >= cfg['max_evaluations']:
            raise BudgetExceeded('evaluation_budget')
        if diag['counters']['node_touches']+len(nodes) > cfg['max_moves']:
            rejects['node_touch_budget'] += 1
            return None
        if all(t == state['assignment'][n] and np.array_equal(p, state['xy'][n]) for n, t, p in zip(nodes, targets, points)):
            rejects['identity'] += 1
            return None
        leaf_delta, layer_delta = defaultdict(lambda: np.zeros(5)), defaultdict(lambda: np.zeros(5))
        for n, t in zip(nodes, targets):
            leaf_delta[int(state['assignment'][n])] -= demand[n]; leaf_delta[t] += demand[n]
            layer_delta[int(state['layer'][n])] -= demand[n]; layer_delta[int(region_layer[t])] += demand[n]
        if any((usage[t]+v > capacity[t]+1e-6).any() for t, v in leaf_delta.items()) or any((layer_usage[t]+v > layer_capacity[t]+1e-6).any() for t, v in layer_delta.items()):
            rejects['capacity'] += 1
            return None
        affected = sorted({ni for n in nodes for ni in prepared.incidence[n]})
        old = snapshot(nodes)
        diag['counters']['evaluations'] += 1
        try:
            for n, t, p in zip(nodes, targets, points):
                state['assignment'][n], state['layer'][n], state['xy'][n] = t, region_layer[t], p
            values = [prepared.cost(ni, state, objective_mode, prepared.normalizers, budget) for ni in affected]
        finally:
            for row in old:
                n = row['node']
                state['assignment'][n], state['layer'][n], state['xy'][n] = row['assignment'], row['layer'], row['xy']
        if any(v is None for v in values):
            rejects['unsupported_vertical_access'] += 1
            return None
        dl = math.fsum(v[0]-costs[ni][0] for ni, v in zip(affected, values))
        dt = math.fsum(v[1]-costs[ni][1] for ni, v in zip(affected, values))
        new_l, new_t = length+dl, timing+dt
        new_f = new_l/prepared.normalizers['wirelength'] if objective_mode == 'wl' else .5*new_l/prepared.normalizers['wirelength']+.5*new_t/prepared.normalizers['timing']
        threshold = max(cfg['gain_absolute'], cfg['gain_relative']*max(1., abs(current)))
        if current-new_f <= threshold:
            rejects['no_strict_gain'] += 1
            return None
        return dict(f=new_f, length=new_l, timing=new_t, nodes=nodes, targets=targets,
            points=[p.copy() for p in points], affected=affected, values=values, old=old,
            leaf_delta=dict(leaf_delta), layer_delta=dict(layer_delta))

    def partners(node):
        # Connections and nearby opposite-layer nodes use the same frozen
        # common ordering in all objective arms; exact-vector matches first.
        z = state['layer'][node]
        neighbours = {int(n) for ni in prepared.incidence[node] for n in prepared.nets[ni] if n != node}
        candidates = {n for n in neighbours if n in eligible_set and state['layer'][n] != z}
        opposite = np.flatnonzero((state['layer'] != z) & movable)
        if len(opposite):
            distance = np.abs(state['xy'][opposite]-state['xy'][node]).sum(1)
            take = min(len(opposite), max(32, cfg['max_swap_partners']*4))
            near = opposite[np.argpartition(distance, take-1)[:take]]
            candidates.update(int(n) for n in near if n in eligible_set)
        return sorted(candidates, key=lambda n: (not np.array_equal(demand[n], demand[node]),
            n not in neighbours, float(np.abs(state['xy'][n]-state['xy'][node]).sum()), n))[:cfg['max_swap_partners']]

    try:
        for iteration in range(cfg['max_passes']):
            budget.check()
            diag['counters']['passes'] = iteration+1
            accepted = 0
            for node in order:
                budget.check()
                if diag['counters']['node_touches'] >= cfg['max_moves']:
                    raise BudgetExceeded('node_touch_budget')
                best = None
                target_rows = np.flatnonzero((bounds[:, 0] <= high[node, 0]) & (bounds[:, 1] >= low[node, 0]) & (bounds[:, 2] <= high[node, 1]) & (bounds[:, 3] >= low[node, 1]))
                for target in target_rows:
                    target = int(target)
                    if not compatible(node, target):
                        rejects['fixed_or_resource_incompatible'] += 1
                        continue
                    limits = box(node, target)
                    if limits is None:
                        continue
                    lo, hi = limits
                    axes = [sorted(set((float(lo[k]), float(np.clip(state['xy'][node, k], lo[k], hi[k])), float(hi[k])))) for k in (0, 1)]
                    for x in axes[0]:
                        for y in axes[1]:
                            candidate = evaluate([node], [target], [np.asarray([x, y], dtype=state['xy'].dtype)])
                            if candidate is not None and (best is None or candidate['f'] < best['f']):
                                best = candidate
                if allow_swaps and diag['counters']['node_touches']+2 <= cfg['max_moves']:
                    for partner in partners(node):
                        nodes = [node, partner]
                        targets = [int(state['assignment'][partner]), int(state['assignment'][node])]
                        if not all(compatible(n, t) for n, t in zip(nodes, targets)):
                            rejects['swap_incompatible'] += 1
                            continue
                        limits = [box(n, t) for n, t in zip(nodes, targets)]
                        if any(b is None for b in limits):
                            rejects['swap_trust_box'] += 1
                            continue
                        points = [np.clip(state['xy'][n], *b).astype(state['xy'].dtype) for n, b in zip(nodes, limits)]
                        candidate = evaluate(nodes, targets, points)
                        if candidate is not None and (best is None or candidate['f'] < best['f']):
                            best = candidate
                if best is None:
                    continue
                # Full fsum once per accepted proposal protects the strict
                # global descent invariant while evaluations stay incident-only.
                updates = dict(zip(best['affected'], best['values']))
                new_costs = [updates.get(ni, c) for ni, c in enumerate(costs)]
                next_l, next_t, next_f = _totals(new_costs, objective_mode, prepared.normalizers)
                threshold = max(cfg['gain_absolute'], cfg['gain_relative']*max(1., abs(current)))
                if current-next_f <= threshold:
                    rejects['global_roundoff_no_gain'] += 1
                    continue
                old_f, old_l, old_t = current, length, timing
                for n, t, p in zip(best['nodes'], best['targets'], best['points']):
                    state['assignment'][n], state['layer'][n], state['xy'][n] = t, region_layer[t], p
                for t, v in best['leaf_delta'].items(): usage[t] += v
                for t, v in best['layer_delta'].items(): layer_usage[t] += v
                costs = new_costs
                length, timing, current = next_l, next_t, next_f
                counters = diag['counters']
                counters['accepted_moves'] += 1; counters['node_touches'] += len(best['nodes'])
                counters['accepted_swaps'] += int(len(best['nodes']) == 2)
                accepted += 1
                diag['history'].append(current); diag['wirelength_history'].append(length); diag['timing_history'].append(timing)
                diag['accepted_moves'].append(dict(nodes=[int(n) for n in best['nodes']],
                    kind='swap' if len(best['nodes']) == 2 else 'single', node_touches=len(best['nodes']),
                    old=best['old'], new=snapshot(best['nodes']), before=old_f, after=current, gain=old_f-current,
                    before_wirelength=old_l, after_wirelength=length, before_timing=old_t, after_timing=timing,
                    affected_nets=best['affected']))
            if not accepted:
                diag['stop_reason'] = 'no_improving_declared_candidate' if len(order) == len(eligible) else 'candidate_node_budget'
                break
        else:
            diag['stop_reason'] = 'pass_budget'
    except BudgetExceeded as exc:
        diag['stop_reason'] = str(exc)
    diag.update(final_wirelength=length, final_timing=timing, final_objective=current,
        rejections=dict(rejects), elapsed_search_seconds=time.perf_counter()-budget.started)
    diag['counters']['work_units'] = budget.work
    result = {k: torch.as_tensor(v, device=data['demand'].device,
        dtype=torch.float32 if k == 'xy' else torch.long).clone() for k, v in state.items()}
    result.update(schema=SCHEMA, initializer='S', decoder_kind='finite-monotone-factorial-refinement-017-1',
        base_action_sha256=prepared.base_digest, refinement_diagnostics=diag)
    result['deployed_action_sha256'] = action_digest(result)
    result['capacity_diagnostics'] = validate(data, result, base=base)
    return result


def validate(data, action, base=None):
    """Own 017 audit: hard constraints, base trust, touches and recorded descent."""
    audit = validate_residual_action(data, action)
    diag = action.get('refinement_diagnostics', {})
    if action.get('schema') == SCHEMA:
        if action.get('deployed_action_sha256') != action_digest(action):
            raise ValueError('017 deployed action digest mismatch')
        hist = diag.get('history', [])
        if not hist or any(not math.isfinite(x) for x in hist) or any(b >= a for a, b in zip(hist, hist[1:])):
            raise ValueError('017 missing or nondecreasing objective history')
        touches = sum(len(m['nodes']) for m in diag.get('accepted_moves', []))
        if touches != diag['counters']['node_touches'] or touches > diag['config']['max_moves']:
            raise ValueError('017 accepted node-touch budget mismatch')
        if len(hist) != len(diag.get('accepted_moves', []))+1 or hist[-1] != diag['final_objective']:
            raise ValueError('017 recorded descent history mismatch')
        moves = diag.get('accepted_moves', [])
        if diag['counters']['accepted_moves'] != len(moves) or diag['counters']['accepted_swaps'] != sum(len(m['nodes']) == 2 for m in moves):
            raise ValueError('017 recorded accepted move counters mismatch')
        for i, move in enumerate(moves):
            if len(move['nodes']) not in (1, 2) or len(set(move['nodes'])) != len(move['nodes']) or move['before'] != hist[i] or move['after'] != hist[i+1] or not math.isclose(move['gain'], hist[i]-hist[i+1], rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError('017 accepted move record mismatch')
    if base is not None:
        before, _ = _core(data, base)
        after, _ = _core(data, action)
        changed = (after['assignment'] != before['assignment']) | (after['layer'] != before['layer']) | np.any(after['xy'] != before['xy'], axis=1)
        movable = _array(data.get('movable', np.ones(len(changed), dtype=bool)), bool)
        if changed[~movable].any():
            raise ValueError('017 changed an immovable APBlock')
        spans = _array(data['region_bounds'], float)[before['assignment']]
        radius = (spans[:, [1, 3]]-spans[:, [0, 2]])*diag.get('config', DEFAULT_CONFIG)['trust_fraction']
        if (np.abs(after['xy']-before['xy']) > radius+1e-7).any():
            raise ValueError('017 exceeded original D trust box')
        if action.get('schema') == SCHEMA and action.get('base_action_sha256') != action_digest(base):
            raise ValueError('017 frozen D digest mismatch')
        if action.get('schema') == SCHEMA and int(changed.sum()) > diag['counters']['node_touches']:
            raise ValueError('017 changes exceed accepted node touches')
        audit.update(changed_nodes=int(changed.sum()), changed_layers=int((after['layer'] != before['layer']).sum()),
            changed_xy=int(np.any(after['xy'] != before['xy'], axis=1).sum()), immovable_preserved=True)
    return audit
