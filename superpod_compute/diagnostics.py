"""Optional mapping diagnostics; never changes the autonomous prediction path.

Observed tiles constrain this diagnostic search only. A macro tile does not
describe vector accesses, compiler register allocation, or a complete kernel.
Service demands are work totals, not additive elapsed-time contributions.
"""
from dataclasses import dataclass, replace

from .api import evaluate_mapping, predict
from .contracts import Hardware, Policy, integer
from .operators import GemmModel


@dataclass(frozen=True)
class ObservedTile:
    bm: int
    bn: int
    bk: int

    def __post_init__(self):
        for name in ('bm', 'bn', 'bk'):
            integer(getattr(self, name), name)


def service_ledger(request, result, hardware, policy):
    """Reconstruct the existing model's stage demand without changing its cost.

    H/R are shared whole-device services for a wave. S/C denote the longest
    CU's service in that wave. Consequently, stage totals must not be treated
    as measured device busy counters or summed to claim the elapsed duration.
    """
    model = GemmModel()
    p = result['mapping']
    totals = dict(H=0.0, R=0.0, S=0.0, C=0.0, output=0.0)
    rows = []
    for wave in result['wave_classes']:
        demand = dict(H=0.0, R=0.0, S=0.0, C=0.0)
        per_cu_m = min(p['bm'], wave['valid_m'])
        per_cu_n = min(p['bn'], wave['valid_n'])
        for run in wave['panel_byte_runs']:
            number, k = run['panels'], run['k_elements']
            service = wave['full_panel_compute'] if k == p['bk'] else wave['tail_panel_compute']
            if service is None:
                raise ValueError('Missing panel compute description')
            dh = run['HBM_bytes_per_panel']
            dr = run['L2_to_core_bytes_per_panel']
            h = max(dh/model.input_hbm_bandwidth(hardware), dh/hardware.l2_write_bandwidth_Bps)
            if dh and policy.charge_hbm_latency:
                h += hardware.hbm_latency_s
            r = run.get('R_service_s')
            if r is None:
                r = max(dr/hardware.l2_read_bandwidth_Bps,
                        model.input_shared_write_time(per_cu_m, per_cu_n, k, hardware, policy))
                r += policy.l2_transfer_latency_s
            read = policy.effective_shared_read_bytes_per_cycle_per_core or hardware.shared_read_bytes_per_cycle_per_core
            s = run.get('Shared_read_service_s_per_core',run['Shared_to_register_bytes_per_core']/(read*hardware.clock_hz) if read else 0.0)
            c = service['duration_s'] + model.panel_control_time(k, hardware, policy)
            for stage, value in zip(('H', 'R', 'S', 'C'), (h, r, s, c)):
                demand[stage] += number*value
        count = wave['count']
        for stage in demand:
            totals[stage] += count*demand[stage]
        output = wave['output_stream']['duration_s']
        totals['output'] += count*output
        rows.append(dict(count=count, active_cores=wave['active_cores'],
                         service_demand_s=demand,
                         modeled_input_pipeline_s=wave['pipeline']['duration_s'],
                         modeled_output_s=output,
                         modeled_wave_s=wave['duration_s']))
    fixed = hardware.kernel_launch_s + policy.device_service_s
    reconstructed = fixed + sum(w['count']*w['modeled_wave_s'] for w in rows)
    return dict(service_demand_s=totals, fixed_s=fixed,
                elapsed_s=reconstructed,
                all_input_stages_serial_s=fixed+sum(totals.values()),
                waves=rows,
                meanings=dict(H='HBM/L2-fill endpoint service',
                              R='L2 read / Shared input-write endpoint service',
                              S='Shared operand reads for the longest CU',
                              C='MMAC plus explicit K-loop control for the longest CU'),
                warning='Serial total is a sensitivity experiment, not an alternative validated predictor')


def diagnose_tile(request, observed_tile, hardware=None, *, policy=None,
                  backend='auto', trace_limit=8):
    """Search model mappings with the observed tile, retaining all other choices.

    No measured GEMM duration is accepted. The unconstrained mapper, its cache
    keys, and its policy are untouched. No kernel selection or exact program
    equivalence is inferred from matching tile dimensions.
    """
    hardware = hardware or Hardware.load()
    policy = policy or Policy()
    if not isinstance(observed_tile, ObservedTile):
        raise TypeError('observed_tile must be ObservedTile')
    restricted = replace(policy, bm_values=(observed_tile.bm,),
                         bn_values=(observed_tile.bn,), bk_values=(observed_tile.bk,),
                         add_shape_candidate=False)
    result = predict(request, hardware, policy=restricted, backend=backend)
    best = result['best']
    if best is None:
        return dict(purpose='diagnostic', alignment='tile-only', prediction=result)
    # Add a representative trace only after the constrained search is finished.
    from .contracts import Mapping
    detailed = evaluate_mapping(request, Mapping(**best['mapping']), hardware,
                                policy=restricted, backend=backend, trace_limit=trace_limit)
    return dict(purpose='diagnostic', alignment='tile-only',
                observed_tile=[observed_tile.bm, observed_tile.bn, observed_tile.bk],
                prediction=result, ledger=service_ledger(request, detailed, hardware, restricted),
                representative_waves=detailed['wave_classes'],
                unresolved_software=['global access vector width and addresses',
                                     'actual Shared layout and instruction traffic',
                                     'compiler allocation and resident workgroup limits',
                                     'prefetch/barrier instruction ordering'])
