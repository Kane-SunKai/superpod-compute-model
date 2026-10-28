"""Immutable, validated contracts; all internal time values are seconds."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

MODEL_VERSION = 'resident-k-streamed-c-2.0'
SEARCH_VERSION = 'dense-candidates-reuse-2.0'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')

def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')

@dataclass(frozen=True)
class GemmRequest:
    M: int
    N: int
    K: int
    dtype: str = 'fp16'
    accumulator_dtype: str = 'fp32'
    layout: str = 'row-major-contiguous'
    operator: str = 'gemm'

    def __post_init__(self):
        for name in ('M', 'N', 'K'):
            integer(getattr(self, name), name, 0)
        if self.operator != 'gemm' or self.dtype not in ('fp16', 'bf16') or self.accumulator_dtype != 'fp32' or self.layout != 'row-major-contiguous':
            raise ValueError('V1 GEMM supports contiguous fp16/bf16 input/output, fp32 accumulation, C=AB only')

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class Hardware:
    name: str = 'A100-LLMCompass-with-project-budgets'
    cores: int = 108
    clock_hz: float = 1.41e9
    arrays_per_core: int = 4
    array_rows: int = 16
    array_cols: int = 16
    mac_per_cell_per_cycle: float = 1.0
    vector_fp16_flops_per_cycle_per_core: float = 512
    hbm_bandwidth_Bps: float = 2039e9
    hbm_latency_s: float = 1e-6
    hbm_capacity_bytes: int = 80_000_000_000
    l2_bytes: int = 40 * 1024**2
    l2_bytes_per_cycle: float = 5120
    source_sram_bytes_per_core: int = 192 * 1024
    shared_budget_bytes_per_core: int = 128 * 1024
    register_budget_bytes_per_core: int = 128 * 1024
    register_reserve_bytes_per_core: int = 32 * 1024
    kernel_launch_s: float = 21e-6
    matrix_k_quantum: int = 16

    def __post_init__(self):
        for name in ('cores','arrays_per_core','array_rows','array_cols','hbm_capacity_bytes','l2_bytes','source_sram_bytes_per_core','shared_budget_bytes_per_core','register_budget_bytes_per_core','matrix_k_quantum'):
            integer(getattr(self,name),name)
        integer(self.register_reserve_bytes_per_core,'register_reserve_bytes_per_core',0)
        for name in ('clock_hz','mac_per_cell_per_cycle','vector_fp16_flops_per_cycle_per_core','hbm_bandwidth_Bps','l2_bytes_per_cycle'):
            positive(getattr(self,name),name)
        for name in ('hbm_latency_s','kernel_launch_s'):
            positive(getattr(self,name),name,True)
        if self.shared_budget_bytes_per_core > self.source_sram_bytes_per_core:
            raise ValueError('shared budget exceeds the configured on-chip SRAM capacity')

    @property
    def core_peak_flops(self):
        return self.arrays_per_core*self.array_rows*self.array_cols*self.mac_per_cell_per_cycle*2*self.clock_hz

    @property
    def l2_bandwidth_Bps(self):
        return self.l2_bytes_per_cycle*self.clock_hz

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, filename=None):
        if filename is None:
            filename=Path(__file__).parent/'data/a100.json'
        return cls(**json.loads(Path(filename).read_text(encoding='utf-8-sig')))

@dataclass(frozen=True)
class Policy:
    bm_values: tuple = (16,32,64,128,192,256,384)
    bn_values: tuple = (16,32,64,128,192,256,384)
    bk_values: tuple = (16,32,64,128,256,512)
    buffer_counts: tuple = (1,2)
    add_shape_candidate: bool = True
    matrix_efficiency: float = 1.0
    charge_hbm_latency: bool = False
    l2_transfer_latency_s: float = 0.0
    epilogue_s_per_wave: float = 0.0
    transfer_resource_mode: str = 'independent'
    top_k: int = 5
    input_reuse: bool = True
    initial_cache_state: str = 'cold'
    device_service_s: float = 0.0
    output_chunk_values: tuple = (4096,16384)
    output_buffer_counts: tuple = (1,2)
    calibration_hardware_sha256: str = ''

    def __post_init__(self):
        for name in ('add_shape_candidate','charge_hbm_latency','input_reuse'):
            if not isinstance(getattr(self,name),bool):raise ValueError(f'{name} must be boolean')
        for name in ('bm_values','bn_values','bk_values','buffer_counts','output_chunk_values','output_buffer_counts'):
            values=tuple(getattr(self,name));object.__setattr__(self,name,values)
            if not values or len(values)!=len(set(values)):
                raise ValueError(f'{name} must contain unique positive integers')
            for v in values:
                integer(v,name)
            if name in ('buffer_counts','output_buffer_counts') and not set(values)<= {1,2}:
                raise ValueError('Only one or two buffers are supported by the fast pipeline')
        positive(self.matrix_efficiency,'matrix_efficiency')
        if self.matrix_efficiency>1:raise ValueError('matrix_efficiency must be <= 1')
        for name in ('l2_transfer_latency_s','epilogue_s_per_wave','device_service_s'):
            positive(getattr(self,name),name,True)
        if self.transfer_resource_mode not in ('independent','shared-port'):
            raise ValueError('Unknown transfer resource mode')
        integer(self.top_k,'top_k')
        if self.initial_cache_state not in ('cold','warm-if-fit'):raise ValueError('Invalid initial cache state')
        fingerprint=self.calibration_hardware_sha256
        if not isinstance(fingerprint,str) or (fingerprint and (len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint))):
            raise ValueError('calibration_hardware_sha256 must be an empty string or lowercase SHA256')

    def to_dict(self):return asdict(self)

@dataclass(frozen=True)
class Mapping:
    bm: int
    bn: int
    bk: int
    gm: int
    gn: int
    l2_buffers: int
    shared_buffers: int
    output_chunk_bytes: int = 16384
    output_buffers: int = 2

    def __post_init__(self):
        for name,value in asdict(self).items():integer(value,name)
        if self.l2_buffers not in (1,2) or self.shared_buffers not in (1,2) or self.output_buffers not in (1,2):raise ValueError('buffer count must be 1 or 2')

    def to_dict(self):return asdict(self)
