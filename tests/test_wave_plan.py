from dataclasses import replace
from pathlib import Path
import unittest

from superpod_compute import Hardware,Mapping
from superpod_compute.backends import MmacBackend
from superpod_compute.wave_plan import WavePlan,WaveMmacBackend,WaveGemmModel


class WavePlanTests(unittest.TestCase):
    def setUp(self):
        self.h=Hardware.load(Path(__file__).resolve().parents[1]/'superpod_compute/data/bw1100.json')

    def test_four_wave_operands_match_rectangle_ownership(self):
        backend=WaveMmacBackend(self.h,WavePlan(192,128,96,64))
        value=backend.estimate(192,128,16)
        self.assertEqual(value.instruction_count,96)
        self.assertEqual(value.cycles,192)
        self.assertEqual(value.shared_read_bytes,4*(96+64)*16*2)

    def test_eight_waves_reuse_operands_locally_not_per_engine(self):
        backend=WaveMmacBackend(self.h,WavePlan(384,256,96,128))
        value=backend.estimate(384,256,16)
        baseline=MmacBackend(self.h).estimate(384,256,16)
        self.assertEqual(value.cycles,baseline.cycles)
        self.assertEqual(value.cycles,768)
        self.assertEqual(value.shared_read_bytes,57344)
        self.assertEqual(baseline.shared_read_bytes,40960)

    def test_matrix_work_is_conserved_and_engines_are_not_duplicated(self):
        plan=WavePlan(384,256,96,128)
        backend=WaveMmacBackend(self.h,plan)
        value=backend.estimate(384,256,32)
        self.assertEqual(value.instruction_count*16*16*16,384*256*32)
        fewer=WaveMmacBackend(replace(self.h,arrays_per_core=2),plan).estimate(384,256,32)
        self.assertEqual(fewer.cycles,2*value.cycles)
        self.assertEqual(fewer.shared_read_bytes,value.shared_read_bytes)

    def test_capacity_counts_the_same_operand_registers(self):
        backend=WaveMmacBackend(self.h,WavePlan(384,256,96,128))
        model=WaveGemmModel(backend)
        p=Mapping(384,256,16,8,9,2,3)
        cap=model.capacity(p,self.h)
        self.assertEqual(cap['operand_register_bytes'],2*57344)
        self.assertEqual(cap['register_bytes'],384*256*4+2*57344+32768)
        self.assertTrue(cap['valid'])
        limited=replace(self.h,register_budget_bytes_per_core=512*1024)
        self.assertIn('registers',model.capacity(p,limited)['rejections'])

    def test_explicit_plan_rejects_unsupported_blocks(self):
        with self.assertRaises(ValueError):WavePlan(129,128,64,64)
        b=WaveMmacBackend(self.h,WavePlan(192,128,96,64))
        with self.assertRaises(ValueError):b.estimate(128,192,16)


if __name__=='__main__':unittest.main()
