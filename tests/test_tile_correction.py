import unittest
from dataclasses import replace

from superpod_compute import GemmRequest, Hardware, Policy, Mapping, predict, evaluate_mapping
from superpod_compute.tile_correction import load, signature

TABLE = 'bw1100_tile_correction'
HW = 'superpod_compute/data/bw1100_mapper_mmac_hardware.json'
PO = 'superpod_compute/data/bw1100_mapper_mmac_policy.json'
import json
from pathlib import Path


def policy():
    return Policy(**json.loads(Path(PO).read_text(encoding='utf-8-sig')))


class TileCorrectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = Hardware.load(HW)

    def test_off_by_default(self):
        self.assertEqual(policy().tile_correction, '')
        self.assertIsNone(signature(''))

    def test_table_loads_and_is_signed(self):
        t = load(TABLE)
        self.assertGreaterEqual(len(t['factors']), 100)
        self.assertTrue(all(f > 0 for f in t['factors'].values()))
        self.assertEqual(len(signature(TABLE)['sha256']), 64)

    def test_unknown_table_is_rejected(self):
        with self.assertRaises(ValueError):
            Policy(tile_correction='no_such_table')
        with self.assertRaises(ValueError):
            load('no_such_table')

    def test_corrected_search_only_proposes_table_tiles(self):
        p = replace(policy(), tile_correction=TABLE)
        r = predict(GemmRequest(2304, 2048, 8192, 'fp16', layout='row-major-contiguous'), self.h,
                    policy=p, backend='mmac', cache_path=None)
        table = load(TABLE)['factors']
        for cand in r['top_candidates']:
            m = cand['mapping']
            self.assertIn((m['bm'], m['bn'], m['bk']), table)
        self.assertIsNotNone(r['best']['tile_correction_factor'])

    def test_correction_scales_the_reported_duration(self):
        req = GemmRequest(2304, 2048, 8192, 'fp16', layout='row-major-contiguous')
        m = Mapping(256, 256, 16, 12, 2, 2, 2, 4096, 2, 2, 'compact')
        plain = evaluate_mapping(req, m, self.h, policy=policy(), backend='mmac')
        corrected = evaluate_mapping(req, m, self.h, policy=replace(policy(), tile_correction=TABLE), backend='mmac')
        factor = load(TABLE)['factors'][(256, 256, 16)]
        self.assertAlmostEqual(plain['duration_us'], corrected['uncorrected_duration_us'], places=9)
        self.assertAlmostEqual(corrected['duration_us'], plain['duration_us'] / factor, places=6)

    def test_mapping_cache_separates_corrected_and_plain(self):
        a = predict(GemmRequest(2304, 2048, 8192, 'fp16', layout='row-major-contiguous'), self.h,
                    policy=policy(), backend='mmac', cache_path=None)
        b = predict(GemmRequest(2304, 2048, 8192, 'fp16', layout='row-major-contiguous'), self.h,
                    policy=replace(policy(), tile_correction=TABLE), backend='mmac', cache_path=None)
        self.assertNotEqual(a['signature_sha256'], b['signature_sha256'])


if __name__ == '__main__':
    unittest.main()
