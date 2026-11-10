"""Optional per-tile correction for the autonomous mapper (off by default).

A correction table maps each compiled library tile (bm, bn, bk) to
c = median over calibration shapes of model(eta=1) / measured, where measured is
the fastest library kernel with that tile. With Policy.tile_correction set, the
mapper only proposes tiles in the table and reports duration / c, so candidates
are ranked by what the hardware achieves with that tile under the shipped library.

This is an empirical lookup, not a physical model: tile features explained only
18% of the variance in held-out-tile tests. It is valid for the hardware build,
library and model version it was fitted on, and for compute-bound prefill
shapes like the calibration set. It does not extrapolate to unseen tiles or to
other hardware. Build a table with tools/build_tile_correction.py.
"""
import hashlib, json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / 'data'


@lru_cache(maxsize=8)
def load(name):
    path = DATA / f'{name}.json'
    if not path.is_file():
        raise ValueError(f'Unknown tile correction table: {name!r} (expected {path})')
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8'))
    factors = {tuple(e['tile']): float(e['factor']) for e in data['tiles']}
    if not factors or any(f <= 0 for f in factors.values()):
        raise ValueError(f'Tile correction table {name!r} has no valid factors')
    return dict(factors=factors, sha256=hashlib.sha256(raw).hexdigest(),
                version=data.get('version', ''), scope=data.get('scope', ''))


def signature(name):
    """Identity that enters the mapping cache key; empty when correction is off."""
    if not name:
        return None
    t = load(name)
    return dict(name=name, sha256=t['sha256'], version=t['version'])
