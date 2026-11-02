import math
import random
import unittest

from superpod_compute import Policy,Mapping
from superpod_compute.scheduler import operand_pipeline,pipeline_segments,output_pipeline,Event,EventScheduler
from test_read_latency import reference


class MultibufferTests(unittest.TestCase):
    def test_policy_accepts_observed_slot_counts_with_bounded_range(self):
        p=Policy(buffer_counts=(1,2,3,6))
        self.assertEqual(p.buffer_counts,(1,2,3,6))
        self.assertEqual(Mapping(192,128,16,8,9,3,6).shared_buffers,6)
        for value in (0,9):
            with self.assertRaises(ValueError):Policy(buffer_counts=(value,))

    def test_three_and_six_slots_against_explicit_dependency_graph(self):
        rng=random.Random(20261103)
        for buffers in ((2,3,1),(3,6,2),(6,6,2),(3,2,3),(8,3,2)):
            for shared in (False,True):
                segments=[(31,*(rng.random()*1e-6 for _ in range(4))),
                          (7,0,.03e-6,.04e-6,.11e-6)]
                actual=operand_pipeline(segments,*buffers,read_latency_s=.24e-6,shared_port=shared,trace_limit=50)
                expected=reference(segments,*buffers,.24e-6,shared)
                self.assertTrue(math.isclose(actual['duration_s'],expected[-1]['end_s'],rel_tol=1e-12))

    def test_acceleration_handles_lcm_and_segment_boundaries(self):
        segments=[(10003,.04e-6,.08e-6,.11e-6,.13e-6),(17,0,.07e-6,.09e-6,.05e-6)]
        for buffers in ((2,3,1),(3,6,2),(3,2,3),(8,3,2)):
            a=operand_pipeline(segments,*buffers,read_latency_s=.24e-6)
            b=operand_pipeline(segments,*buffers,read_latency_s=.24e-6,accelerate=False)
            self.assertTrue(math.isclose(a['duration_s'],b['duration_s'],rel_tol=1e-10))
            self.assertEqual(a['panels'],10020)
            self.assertGreater(a['skipped_panels'],9000)

    def test_hrc_and_output_recurrences_keep_slot_phase(self):
        stages=[(10003,.08e-6,.11e-6,.13e-6),(7,0,.09e-6,.05e-6)]
        for buffers in ((3,6),(8,3),(3,2)):
            for function in (pipeline_segments,output_pipeline):
                a=function(stages,*buffers)
                b=function(stages,*buffers,accelerate=False)
                self.assertTrue(math.isclose(a['duration_s'],b['duration_s'],rel_tol=1e-10))


if __name__=='__main__':unittest.main()
