import json, math, unittest
from pathlib import Path
from superpod_compute.decode_replay import class_key, load, predict, select, service

ROOT = Path(__file__).resolve().parents[1]


def toy():
    return dict(version='t', cus=72, fixed_s=0.0, bandwidth_cap_Bps=1e12,
                domain=dict(M=[1, 32], N=[1024, 131072], K=[1024, 16384]),
                classes={'a': dict(tile=[16, 128, 32], prefetch_global=2, workgroup=[32, 8, 1],
                                   r_block_Bps=40e9),
                         'b': dict(tile=[64, 128, 16], prefetch_global=1, workgroup=[32, 8, 1],
                                   r_block_Bps=12e9, slow_over_fast=1.9)},
                table=[dict(shape=[2, 8192, 4096], **{'class': 'a'}),
                       dict(shape=[2, 38912, 4096], **{'class': 'b'})])


class DecodeReplayTest(unittest.TestCase):
    def test_wave_rounds_quantize_on_cu_count(self):
        p = toy()
        one = service([2, 128 * 72, 4096], 'b', p)
        two = service([2, 128 * 73, 4096], 'b', p)
        self.assertEqual((one['rounds'], two['rounds']), (1, 2))
        self.assertAlmostEqual(two['wave_s'], 2 * one['wave_s'])

    def test_bandwidth_cap_binds_when_waves_are_cheap(self):
        p = toy(); p['classes']['a']['r_block_Bps'] = 1e15
        s = service([2, 8192, 4096], 'a', p)
        self.assertEqual(s['bound'], 'bandwidth')
        self.assertAlmostEqual(s['duration_s'], (2 * 4096 + 4096 * 8192) * 2 / 1e12)

    def test_nearest_neighbour_in_log_space(self):
        p = toy()
        self.assertEqual(select([2, 36864, 4096], p)[0], 'b')
        self.assertEqual(select([2, 12288, 4096], p)[0], 'a')

    def test_address_sensitive_class_reports_interval(self):
        r = predict([2, 38912, 4096], toy())
        self.assertTrue(r['address_sensitive'])
        self.assertAlmostEqual(r['interval_us'][1] / r['interval_us'][0], 1.9)

    def test_domain_and_input_checks(self):
        with self.assertRaises(ValueError): predict([64, 8192, 4096], toy())
        with self.assertRaises(ValueError): predict([2, 8192], toy())
        self.assertTrue(predict([2, 8192, 512], toy())['extrapolated'])

    def test_class_key_is_stable(self):
        self.assertEqual(class_key([16, 128, 32], 2, [32, 8, 1]), '16x128x32_pg2_wg32x8x1')

    def test_shipped_profile_excludes_holdout(self):
        prof = load()
        table = {tuple(r['shape']) for r in prof['table']}
        hold = {tuple(s) for s in json.loads((ROOT / 'superpod_compute/data/bw1100_decode_holdout.json').read_text())}
        self.assertFalse(table & hold)
        for r in prof['table']:
            self.assertIn(r['class'], prof['classes'])


if __name__ == '__main__':
    unittest.main()
