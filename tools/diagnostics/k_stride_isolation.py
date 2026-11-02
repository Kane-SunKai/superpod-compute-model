"""Keep M/N aligned; isolate K-tail work from A row-start misalignment.

Reuses exactly the existing Graph measurement/correctness protocol. This is a
new diagnostic plan, not the reserved held-out validation set. Output files
are separate; the original plan and measurements are never overwritten.
"""
import boundary_experiment


def plan():
    rows = []
    for m, n in ((512, 768), (1536, 1152)):
        for k in (1023, 1024, 1025):
            for layout in ('contiguous', 'pad_a16'):
                rows.append({
                    'id': f'k_stride_isolation_{m}x{n}x{k}_{layout}',
                    'group': 'k_stride_isolation', 'shape': [m, n, k],
                    'layout': layout, 'role': 'diagnosis',
                })
    return rows


if __name__ == '__main__':
    # Inject only the case list; all timing, precision and checking stay shared.
    boundary_experiment.plan = plan
    boundary_experiment.main()
