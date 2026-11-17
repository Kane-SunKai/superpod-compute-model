"""Vector-path operators: row softmax, elementwise activations, row normalisation.

The point of this file is that the hardware is a parameter, not a constant. Every
term below is written so that changing a hardware field changes the prediction:
swap in a chip with twice the vector width, twice the cores, twice the bandwidth
or a bigger L1D and the answer moves accordingly. What deliberately does *not*
move with the hardware is one dimensionless number per operator and dtype - the
efficiency with which the vendor library uses the vector unit. That is the only
calibrated quantity, and it is calibrated on this machine.

Physical layer
    A row reduction has the structure the matrix model already describes: the row
    is streamed through the on-chip buffer, compute overlaps the load, the result
    is written back, and the cost is the larger roofline divided by the wave
    occupancy.

        t_hbm     (scans + 1) * elements * element_bytes / stream bandwidth
        t_l2_read  elements * element_bytes / l2 read bytes per cycle / clock
        t_l2_write elements * element_bytes / l2 write bytes per cycle / clock
        t_mem     max(t_hbm, t_l2_read, t_l2_write)
        t_vector  total_ops / (cores * vector ops per cycle * clock * efficiency)
        spill     rows * max(0, length * element_bytes - l1d_bytes)
                  / (l2 read bytes per cycle * clock)
        t         max(t_mem, t_vector) / occupancy + spill + launch + latency
                  + startup

    `total_ops` is the operator's own work: `elements * ops_per_element`, plus a
    small per-row term for the cross-lane reduction. `ops_per_element` is a
    property of the operation, taken from what the operation is defined to do,
    not from how fast this library happens to run it - a gelu is a gelu on any
    chip. The overlap of the load and the arithmetic is the `max()`.

    `scans` follows the row: the warp kernel keeps a row in registers and reads
    it once; the streamed kernel reads it twice and relies on L1D for the second
    read; anything past L1D is `spill`, charged at the L2 byte rate.

Library layer
    Four things the vendor library decided, measured rather than assumed:

      `row_capacity_elements`  where it stops using the warp kernel
                               (`softmax_warp_forward`, template parameter
                               log2(R)) and switches to the streamed one
                               (`cunn_SoftMaxForwardGmem`).
      `efficiency`             fraction of the vector unit the kernel actually
                               reaches. 0.05-0.10 here, which is why the vector
                               ALU is never the binding constraint for these
                               operators and why the model cannot be built from
                               hardware alone.
      `row_cycles`             per-row block setup and cross-lane reduction, in
                               cycles rather than seconds: it is latency, not
                               arithmetic, so it scales with the clock instead
                               of with the vector width.
      `row_latency_factor`     how much slower one row runs when it cannot
                               overlap with other rows. Fitted on softmax; 1.0
                               where the effect was measured to be absent.
      `achievement`            whatever is left, allowed only when the residual
                               is flat.

    `startup` is the one latency term that is not fitted: `startup_round_trips`
    is a kernel property, and the seconds come from `hardware.hbm_latency_s`,
    so a machine with slower HBM is predicted to start slower by itself.

    When the hardware changes, the fitted cycle counts and efficiencies have to
    be re-checked; the start-up latency follows the hardware latency field.

Nothing here is registered in the operator registry: these operators have no
tiling to search, so they do not need Mapping, candidates() or capacity().
"""
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path

PROFILE = Path(__file__).resolve().parent / 'data/bw1100_vector_profile.json'

ELEMENT_BYTES = {'fp16': 2, 'bf16': 2, 'fp32': 4}
REDUCTIONS = ('softmax', 'layernorm')


@dataclass(frozen=True)
class VectorRequest:
    kind: str
    rows: int
    length: int
    dtype: str = 'fp16'

    def __post_init__(self):
        if self.kind not in ('softmax', 'gelu', 'gelu-tanh', 'layernorm'):
            raise ValueError('Unknown vector operator: %r' % self.kind)
        if self.dtype not in ELEMENT_BYTES:
            raise ValueError('Unknown dtype: %r' % self.dtype)
        if self.rows <= 0 or self.length <= 0:
            raise ValueError('rows and length must be positive')

    @property
    def elements(self):
        return self.rows * self.length

    @property
    def element_bytes(self):
        return ELEMENT_BYTES[self.dtype]


def load_profile(path=PROFILE):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def regime(request, profile):
    """Which of the library's row kernels this shape lands in.

    The base warp / gmem split uses `row_capacity_elements` (2048 here). A
    further subdivision of the gmem regime is needed for layernorm: the
    library falls into a lower-throughput mode in two measured situations
    (换机必须重测):
      * a large hidden size with only a few rows (few workgroups), where the
        streamed kernel suffers extra L2 spill and bandwidth contention;
      * a footprint of 200 MB or more, where the library drops to a slower
        streaming mode irrespective of the row count.
    Both land in `gmem_large`, which carries its own calibrated efficiency.
    """
    if request.kind not in REDUCTIONS:
        return 'elem'
    if request.length <= profile['row_capacity_elements']:
        return 'warp'
    # gmem regime. Sub-classify the library's slow modes.
    if request.kind == 'layernorm':
        large_n_threshold = profile.get('layernorm_large_n_threshold', 8192)
        small_rows_threshold = profile.get('layernorm_small_rows_threshold', 64)
        footprint_threshold = profile.get('layernorm_slow_footprint_bytes', 0)
        footprint = request.elements * request.element_bytes
        if ((request.length > large_n_threshold and request.rows < small_rows_threshold)
                or (footprint_threshold and footprint >= footprint_threshold)):
            return 'gmem_large'
    return 'gmem'


def occupancy(workgroups, hardware):
    waves = ceil(workgroups / hardware.cores)
    return workgroups / (waves * hardware.cores)


def workgroups_of(request, profile):
    if request.kind in REDUCTIONS:
        return request.rows
    return ceil(request.elements / profile.get('elements_per_block', 1024))


def row_latency_factor(profile, kind, dtype, key):
    """How much slower one row runs when it cannot overlap with other rows.

    A per-row critical path, not a memory latency: softmax keeps a row in
    registers and its un-overlapped row costs several times the saturated one,
    so the factor is fitted. `row_latency_factor` overrides the shared
    `latency_factor` per operator where the effect was measured to be absent.
    """
    override = profile.get('row_latency_factor', {}).get(kind, {}).get(dtype, {})
    if key in override:
        return override[key]
    return profile.get('latency_factor', {}).get(key, 1.0)


def startup_latency_s(profile, kind, dtype, key, hardware):
    """Memory latency exposed once per kernel, in seconds.

    No block can begin its row before its first bytes arrive, and at the start
    of a kernel there is nothing in flight to hide it. The round-trip count is
    a property of the kernel's structure; the latency itself is a hardware
    field, so a machine with slower HBM starts slower without a new constant.
    """
    trips = (profile.get('startup_round_trips', {}).get(kind, {})
             .get(dtype, {}).get(key, 0.0))
    return trips * hardware.hbm_latency_s



def predict(request, hardware, profile=None):
    """Prediction for one vector operator, in seconds."""
    profile = profile or load_profile()
    b = request.element_bytes
    n = request.elements
    key = regime(request, profile)

    # Only the streamed reduction kernel reads a row twice; the warp kernel keeps
    # it, and an elementwise operator never reads anything twice.
    scans = 2 if key in ('gmem', 'gmem_large') else 1
    # The achievable streaming rate on this card is a fixed ratio above the
    # profile's nominal HBM figure (1.80 against 1.72 TB/s), so it moves with the
    # hardware rather than being pinned in the vector profile.
    bandwidth = hardware.hbm_bandwidth_Bps * profile.get('stream_bandwidth_ratio', 1.0)
    t_hbm = (scans + 1) * n * b / bandwidth

    # L2 sits between the CUs and HBM and its two directions are separate ports,
    # so every byte read or written passes one of them and each is its own floor.
    # The write port matters once HBM stops being the slower of the two.
    rate = hardware.clock_hz
    t_l2_read = (n * b / (hardware.l2_read_bytes_per_cycle * rate)
                 if hardware.l2_read_bytes_per_cycle else 0.0)
    t_l2_write = (n * b / (hardware.l2_write_bytes_per_cycle * rate)
                  if hardware.l2_write_bytes_per_cycle else 0.0)
    t_mem = max(t_hbm, t_l2_read, t_l2_write)

    # Per element: the operator's own work divided by the hardware's vector rate.
    # This is the term that moves when the hardware moves - a wider vector unit
    # or a faster clock makes it smaller, and the efficiency is the only thing
    # that has to be re-measured on a new machine.
    ops = profile['ops_per_element'][request.kind][request.dtype]
    efficiency = profile['efficiency'][request.kind][request.dtype][key]
    elem_cycles = ops / (hardware.vector_flops_per_cycle * efficiency)

    # Per row: block setup and the cross-lane reduction. Kept in cycles rather
    # than converted to ops because it is latency, not arithmetic - a wider
    # vector unit does not make a barrier cheaper.
    row_cycles = profile['row_cycles'].get(request.kind, {}).get(request.dtype, {}).get(key, 0.0)
    per_row = row_cycles + request.length * elem_cycles
    t_vector = request.rows * per_row / (hardware.cores * hardware.clock_hz)
    ideal = request.rows * request.length * ops / (
        hardware.cores * hardware.vector_flops_per_cycle * hardware.clock_hz)

    # The streamed kernel reads each row twice. The second read is served from
    # L1D while the row fits in 32 KiB; whatever exceeds that misses and is
    # fetched from L2, which the whole chip shares at a fixed byte rate.
    spill = 0.0
    if key in ('gmem', 'gmem_large') and hardware.l1d_bytes and hardware.l2_read_bytes_per_cycle:
        excess = request.rows * max(0, request.length * request.element_bytes
                                    - hardware.l1d_bytes)
        spill = excess / (hardware.l2_read_bytes_per_cycle * hardware.clock_hz)

    workgroups = workgroups_of(request, profile)
    util = occupancy(workgroups, hardware)

    t_physical = max(t_mem, t_vector) / util
    achievement = profile['achievement'].get(request.kind, {}).get(request.dtype, 1.0)

    # With fewer rows than the machine can keep busy, a row cannot overlap with
    # other rows, so it costs `row_latency_factor` times the saturated per-row
    # cost. This is a property of the row's critical path (softmax keeps a row
    # in registers), not of the memory latency, and it is 1.0 for layernorm,
    # which measured no such effect.
    alpha = row_latency_factor(profile, request.kind, request.dtype, key)
    latency = (alpha - 1.0) * per_row / hardware.clock_hz

    # Separate from that: no block can start its row until its first bytes
    # arrive, and at the start of a kernel there is nothing in flight to hide
    # it. The number of exposed round trips belongs to the kernel; the latency
    # is a hardware field, so a machine with slower HBM starts slower without
    # a new fitted constant.
    startup = startup_latency_s(profile, request.kind, request.dtype, key, hardware)

    duration = (t_physical / achievement + spill + profile.get('launch_s', 0.0)
                + latency + startup)

    return dict(kind=request.kind, dtype=request.dtype, rows=request.rows,
                length=request.length, elements=n, regime=key, scans=scans,
                workgroups=workgroups, occupancy=round(util, 4),
                total_ops=n * ops, ops_per_element=ops, efficiency=efficiency,
                elem_cycles=elem_cycles, row_cycles=row_cycles,
                t_mem_s=t_mem, t_hbm_s=t_hbm, t_l2_read_s=t_l2_read,
                t_l2_write_s=t_l2_write, t_vector_s=t_vector, t_ideal_s=ideal,
                bound='vector' if t_vector >= t_mem else 'mem',
                t_physical_s=t_physical, achievement=achievement,
                per_row_cycles=per_row, spill_s=spill, latency_s=latency,
                startup_s=startup,
                duration_s=duration, duration_us=round(duration * 1e6, 1))
