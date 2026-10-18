"""Post-GP actions on intact AP blocks (molecules + physical RAM groups).

RUDY is a demand heuristic, NOT actual RRG cut capacity or a routing certificate.
Swaps preserve the complete model multiset and coordinate occupancy. VPR remains
responsible for legalization. No routing outcome is used to construct actions.
"""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def snapshot(prefix):
    prefix = str(prefix)
    blocks = {}
    atoms = set()
    for line in Path(prefix + '.blocks.tsv').read_text().splitlines():
        i, atom, model, fixed, x, y, z, sub = line.split('\t')
        i = int(i)
        loc = [float(x), float(y), float(z), int(sub)]
        if not all(math.isfinite(t) for t in loc) or atom in atoms:
            raise ValueError('invalid/duplicate GP atom')
        atoms.add(atom)
        block = blocks.setdefault(i, dict(atoms=[], models=[], fixed=bool(int(fixed)), loc=loc))
        if block['loc'] != loc:
            raise ValueError('incoherent AP block')
        block['atoms'].append(atom)
        block['models'].append(model)
    ids = sorted(blocks)
    remap = {n: i for i, n in enumerate(ids)}
    nets = []
    for line in Path(prefix + '.nets.tsv').read_text().splitlines():
        pins = [remap[int(x)] for x in line.split('\t')[1:]]
        if len(pins) > 1:
            nets.append(pins)
    size = list(map(int, Path(prefix + '.meta.tsv').read_text().split()))
    if not ids or len(size) != 2:
        raise ValueError('empty/incomplete GP snapshot')
    return dict(blocks=[blocks[i] for i in ids], nets=nets, size=size)

def positions(graph):
    return np.asarray([b['loc'][:2] for b in graph['blocks']], dtype=np.float64)

def flat(path, graph, xy):
    # 17 significant digits make an unchanged GP replay lossless for C++ doubles.
    lines = ['# TailCompile post-GP AP-block coherent placement; not a legal placement\n']
    for i, b in enumerate(graph['blocks']):
        for atom in b['atoms']:
            lines.append(f"{atom} {xy[i,0]:.17g} {xy[i,1]:.17g} {b['loc'][2]:.17g} {b['loc'][3]}\n")
    Path(path).write_text(''.join(lines))

def pressure(graph, xy, tile=8):
    nx, ny = [max(1, math.ceil(n / tile)) for n in graph['size']]
    grid = np.zeros((nx, ny), dtype=np.float64)
    wire = 0.0
    for pins in graph['nets']:
        pts = xy[pins]
        lo, hi = pts.min(0), pts.max(0)
        span = hi - lo
        wire += float(span.sum())
        low = np.clip((lo / tile).astype(int), 0, [nx-1, ny-1])
        high = np.clip((hi / tile).astype(int), 0, [nx-1, ny-1])
        area = max(1., float((high[0]-low[0]+1)*(high[1]-low[1]+1)))
        grid[low[0]:high[0]+1, low[1]:high[1]+1] += float(span.sum()) / area
    cell = np.clip((xy / tile).astype(int), [0, 0], [nx-1, ny-1])
    local = grid[cell[:, 0], cell[:, 1]]
    tail = np.sort(grid.reshape(-1))[-max(1, math.ceil(grid.size*.05)):].mean()
    return local, dict(hpwl=wire, mean=float(grid.mean()), tail=float(tail), maximum=float(grid.max()))

def candidates(graph, width, replicas=2):
    xy = positions(graph)
    local, base = pressure(graph, xy)
    groups = defaultdict(list)
    incident = [[] for _ in graph['blocks']]
    for j, pins in enumerate(graph['nets']):
        for i in pins:
            incident[i].append(j)
    for i, b in enumerate(graph['blocks']):
        # Do not swap perimeter blocks, including unconstrained IO.
        if b['fixed'] or min(xy[i]) < 1 or any(xy[i] >= np.asarray(graph['size'])-1):
            continue
        key = tuple(sorted(Counter(b['models']).items()))
        groups[key].append(i)
    partners = {i: ids for ids in groups.values() for i in ids}
    movable = sorted(partners)
    if not movable:
        return [('replay', xy.copy(), dict(base, moved=0, strategy='replay'))]
    output = [('replay', xy.copy(), dict(base, moved=0, strategy='replay'))]
    # Frozen proposal budget, not tuned on test routing labels.
    for fraction in (.01, .02, .05):
        for strategy in ('hotspot', 'wire'):
            for rep in range(replicas):
                rng = np.random.default_rng(6600 + rep + int(fraction*1000))
                work = xy.copy()
                used = set()
                limit = max(2, int(len(movable)*fraction))
                order = sorted(movable, key=lambda i: (-local[i], i)) if strategy == 'hotspot' else list(rng.permutation(movable))
                for i in order:
                    if len(used)+2 > limit:
                        break
                    if i in used:
                        continue
                    eligible = [j for j in partners[i] if j != i and j not in used
                                and 0 < np.abs(work[j]-work[i]).sum() <= 16]
                    if not eligible:
                        continue
                    sampled = rng.choice(eligible, min(16, len(eligible)), replace=False)
                    best = None
                    for j in sampled:
                        nets = sorted(set(incident[i]+incident[j]))
                        old = sum(float(np.ptp(work[graph['nets'][n]], axis=0).sum()) for n in nets)
                        work[[i,j]] = work[[j,i]]
                        new = sum(float(np.ptp(work[graph['nets'][n]], axis=0).sum()) for n in nets)
                        work[[i,j]] = work[[j,i]]
                        # hotspot terms are load proxies, not ground-truth congestion.
                        delta = (new-old) / max(1., old)
                        if strategy == 'hotspot':
                            delta += .15 * (local[j]-local[i]) / max(1., local.max())
                        item = (delta, int(j))
                        if best is None or item < best:
                            best = item
                    if best and best[0] < 0:
                        j = best[1]
                        work[[i,j]] = work[[j,i]]
                        used.update((i,j))
                _, features = pressure(graph, work)
                # Retain rejected/no-op candidates explicitly; never claim diversity from names.
                if features['hpwl'] > 1.03 * max(base['hpwl'], 1):
                    work = xy.copy()
                    features = base.copy()
                moved = int(np.any(work != xy, axis=1).sum())
                output.append((f'{strategy}_{int(fraction*100):02d}_r{rep}', work,
                               dict(features, moved=moved, strategy=strategy, budget=fraction)))
    return output

def features(graph, xy, width):
    original = positions(graph)
    scale = np.maximum(graph['size'], 1)
    local, summary = pressure(graph, xy)
    degree = np.zeros(len(xy))
    src, dst = [], []
    # Hypernet star, linear in pin count; no quadratic clique materialization.
    for pins in graph['nets']:
        for i in pins:
            degree[i] += 1
        for i in pins[1:]:
            src.extend((pins[0], i)); dst.extend((i, pins[0]))
    nodes = np.column_stack((xy/scale, (xy-original)/scale,
        np.log1p(degree), [b['fixed'] for b in graph['blocks']],
        np.log1p([len(b['atoms']) for b in graph['blocks']]),
        np.log1p(local/max(width, 1)), np.any(xy != original, axis=1))).astype(np.float32)
    # Fixed architecture model IDs are categorical, never ordinal magnitudes.
    models=np.zeros((len(xy),32),dtype=np.float32)
    for i,b in enumerate(graph['blocks']):
        for model in b['models']:
            index=int(model)
            if index<0 or index>=32:
                raise ValueError('architecture has >32 model categories; revise schema explicitly')
            models[i,index]+=1
    nodes=np.column_stack((nodes,np.log1p(models))).astype(np.float32)
    global_x = np.asarray([math.log1p(width), math.log1p(len(xy)),
        math.log1p(len(graph['nets'])), summary['tail']/max(summary['mean'],1e-6),
        summary['hpwl']/max(1,len(graph['nets'])*sum(graph['size']))], dtype=np.float32)
    return dict(x=nodes, src=np.asarray(src,dtype=np.int64), dst=np.asarray(dst,dtype=np.int64), g=global_x)

def retention(before, after, graph, xy):
    def parse(path):
        result = {}
        if not Path(path).is_file():
            return result
        for line in Path(path).read_text().splitlines():
            fields = line.split()
            if len(fields) >= 5 and not line.startswith('#'):
                try:
                    result[fields[0]] = (float(fields[1]), float(fields[2]))
                except ValueError:
                    pass
        return result
    reference, actual = parse(before), parse(after)
    deltas = []
    original = positions(graph)
    for i,b in enumerate(graph['blocks']):
        desired = xy[i]-original[i]
        if not np.any(desired):
            continue
        for atom in b['atoms']:
            if atom in reference and atom in actual:
                displacement = np.asarray(actual[atom])-reference[atom]
                deltas.append((float(displacement @ desired / max(desired @ desired,1e-12)),
                               float(np.linalg.norm(displacement-desired))))
    expected = sum(len(b['atoms']) for i,b in enumerate(graph['blocks']) if np.any(xy[i] != original[i]))
    return dict(expected_moved_atoms=expected, matched_moved_atoms=len(deltas),
                directional_retention=float(np.mean([d[0] for d in deltas])) if deltas else None,
                displacement_error=float(np.mean([d[1] for d in deltas])) if deltas else None)
