"""Parse VPR work diagnostics without confusing nested timers with wall time.

The authoritative end-to-end measurement is the subprocess wall_seconds from
common.execute. VPR's printed timers are retained as *VPR-reported seconds*;
they are not promoted to CPU or external wall measurements. Missing completion
lines (especially after timeout) stay None. Timers for AP, its GP/legalizer/
detailed subphases, and the entire VPR flow must never be added together.
"""
from __future__ import annotations

import re


SCHEMA = 'tailcompile-v69-phase-metrics-1'
NUMBER = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
TIMER = re.compile(r'^\s*(?P<depth>#+)?\s*(?P<label>.+?) took '
                   rf'(?P<value>{NUMBER}) seconds(?:\s|$)')
LABELS = {
    'AP Global Placer': 'global_placement_seconds',
    'AP Full Legalizer': 'full_legalization_seconds',
    'AP Detailed Placer': 'detailed_placement_seconds',
    'Routing': 'routing_seconds',
    'Analytical Placement': 'analytical_placement_seconds',
    'The entire flow of VPR': 'vpr_reported_seconds',
}
COUNTERS = {
    'gp_cg_iterations': r'Total number of CG iterations:\s*(\d+)',
    'legalizer_partitioned_windows': r'Total number of windows partitioned:\s*(\d+)',
    'legalizer_partitioned_blocks': r'Total number of blocks partitioned:\s*(\d+)',
    'placement_temperatures': r'Placement number of temperatures:\s*(\d+)',
    'placement_swap_attempts': r'Placement total # of swap attempts:\s*(\d+)',
    'routing_iterations_reported_success': r'Successfully routed after (\d+) routing iterations',
}


def parse_phase_metrics(text: str) -> dict:
    """Return JSON-safe metrics and auditable line-numbered evidence.

    Iteration-table counts are explicitly observations, not inferred completed
    iteration totals. GP row 0 is initialization and is not an optimizer step.
    A timeout can therefore retain useful partial work counts without inventing
    a completed phase runtime. Repeated completed phase occurrences are summed
    only within that exact label; enclosing and nested labels remain separate.
    """
    times = {key: None for key in LABELS.values()}
    occurrences = {key: 0 for key in times}
    counts = {key: None for key in COUNTERS}
    evidence = []
    table = None
    table_rows = {'gp': [], 'routing': [], 'annealing': []}
    table_headers = {}
    counter_patterns = {k: re.compile(v) for k, v in COUNTERS.items()}

    for number, line in enumerate(text.splitlines(), 1):
        match = TIMER.match(line)
        if match and match['label'] in LABELS:
            key = LABELS[match['label']]
            value = float(match['value'])
            times[key] = (times[key] or 0.0) + value
            occurrences[key] += 1
            evidence.append(dict(kind='timer', key=key, label=match['label'],
                                 seconds=value, line=number, raw=line.strip(),
                                 nesting_depth=len(match['depth'] or '')))
        for key, pattern in counter_patterns.items():
            match = pattern.search(line)
            if match:
                value = int(match[1])
                counts[key] = (counts[key] or 0) + value
                evidence.append(dict(kind='counter', key=key, value=value,
                                     line=number, raw=line.strip()))

        if re.match(r'^\s*Iter\s+Lower Bound HPWL\b', line):
            table = 'gp'
        elif re.match(r'^\s*Iter\s+Time\s+pres\b', line):
            table = 'routing'
        elif re.match(r'^\s*Tnum\s+Time\s+T\s+AvCost\b', line):
            table = 'annealing'
        else:
            if table is None:
                continue
            # Headers' units and separator lines are not iteration rows.
            if not line.strip() or re.match(r'^\s*[- ]+$', line) or '(sec)' in line:
                continue
            row = re.match(rf'^\s*(\d+)\s+({NUMBER})\s+', line)
            if row:
                table_rows[table].append((int(row[1]), number, line.strip()))
                continue
            table = None
            continue
        table_headers[table] = dict(line=number, raw=line.strip())

    for name, rows in table_rows.items():
        counts[f'{name}_iteration_rows_observed'] = len(rows) if rows else None
        counts[f'{name}_last_iteration_index'] = rows[-1][0] if rows else None
        if rows:
            index, number, raw = rows[-1]
            evidence.append(dict(kind='iteration_table_last_row', table=name,
                                 last_index=index, line=number, raw=raw,
                                 header=table_headers[name]))
    counts['gp_optimizer_steps_observed'] = (
        sum(index > 0 for index, _, _ in table_rows['gp']) if table_rows['gp'] else None)

    completed = {key: occurrences[key] > 0 for key in occurrences}
    # Four disjoint high-level portions of the AP+route flow are useful for
    # diagnosis, but their sum omits setup, lookahead and analysis. Do not expose
    # that subtotal as either the VPR total or a separately measured runtime.
    disjoint_keys = ('global_placement_seconds', 'full_legalization_seconds',
                     'detailed_placement_seconds', 'routing_seconds')
    return dict(schema=SCHEMA, seconds=times, counters=counts,
        completed_timer_occurrences=occurrences, completed_phase_timers=completed,
        disjoint_phase_keys=list(disjoint_keys), evidence=evidence,
        timing_semantics={
            'printed_timers': 'vpr_reported_seconds; clock kind is not asserted from log text',
            'external_wall': 'Use trial.wall_seconds from the process stopwatch; not reconstructed here',
            'cpu_seconds': None,
            'nesting': 'Analytical Placement contains GP, full legalization and detailed placement; entire VPR contains all phases. Do not add enclosing and nested timers.',
            'partial': 'An absent completion timer is unknown, not zero; observed table rows may be partial.'})
