import itertools
import math
import random
import unittest

from superpod_compute import Policy
from superpod_compute.scheduler import Event,EventScheduler,operand_pipeline,pipeline_segments


def reference(segments,lb,sb,rb,latency,shared):
    events=[];i=0
    for count,h,r,s,c in segments:
        for _ in range(count):
            events.extend([
                Event(f'H{i}',h,('port' if shared else 'HBM',),(f'R{i-lb}',) if i>=lb else ()),
                Event(f'Q{i}',latency,(),(f'H{i}',)+((f'S{i-sb}',) if i>=sb else ())),
                Event(f'R{i}',r,('port' if shared else 'L2',),(f'Q{i}',)),
                Event(f'S{i}',s,('LDS-read',),(f'R{i}',)+((f'C{i-rb}',) if i>=rb else ())),
                Event(f'C{i}',c,('matrix',),(f'S{i}',))])
            i+=1
    return EventScheduler().run(events)


class ReadLatencyTests(unittest.TestCase):
    def test_two_requests_overlap_latency_but_not_payload(self):
        result=operand_pipeline([(2,0,1,0,1)],2,2,2,read_latency_s=10,trace_limit=2)
        self.assertEqual(result['duration_s'],13)
        a,b=result['trace']
        self.assertEqual((a['R_request_s'],b['R_request_s']),(0,0))
        self.assertEqual((a['R_start_s'],a['R_end_s']),(10,11))
        self.assertEqual((b['R_start_s'],b['R_end_s']),(11,12))

    def test_single_shared_slot_holds_the_inflight_destination(self):
        result=operand_pipeline([(2,0,1,0,1)],2,1,2,read_latency_s=10,trace_limit=2)
        self.assertEqual(result['duration_s'],23)
        self.assertEqual(result['trace'][1]['R_request_s'],11)

    def test_fast_recurrence_matches_independent_event_graph(self):
        rng=random.Random(20261102)
        for lb,sb,rb,shared in itertools.product((1,2),(1,2),(1,2),(False,True)):
            for latency in (0,0.24e-6,2e-6):
                segments=[(rng.randint(1,17),*(rng.random()*1e-6 for _ in range(4))) for _ in range(3)]
                actual=operand_pipeline(segments,lb,sb,rb,read_latency_s=latency,shared_port=shared,trace_limit=100)
                expected=reference(segments,lb,sb,rb,latency,shared)
                self.assertTrue(math.isclose(actual['duration_s'],expected[-1]['end_s'],rel_tol=1e-12,abs_tol=1e-15))
                events={x['name']:x for x in expected}
                for row in actual['trace']:
                    i=row['panel']
                    for stage in ('H','R','S','C'):
                        self.assertAlmostEqual(row[stage+'_start_s'],events[f'{stage}{i}']['start_s'],places=15)
                    self.assertAlmostEqual(row['R_request_s'],events[f'Q{i}']['start_s'],places=15)

    def test_acceleration_across_long_runs_and_tail(self):
        segments=[(10001,.1e-6,.05e-6,.08e-6,.2e-6),(3,0,.07e-6,.04e-6,.03e-6)]
        for buffers in itertools.product((1,2),repeat=3):
            slow=operand_pipeline(segments,*buffers,read_latency_s=.24e-6,accelerate=False)
            fast=operand_pipeline(segments,*buffers,read_latency_s=.24e-6)
            self.assertTrue(math.isclose(slow['duration_s'],fast['duration_s'],rel_tol=1e-10,abs_tol=1e-14))
            self.assertEqual(fast['panels'],10004)

    def test_three_stage_pipeline_has_the_same_request_semantics(self):
        result=pipeline_segments([(2,0,1,1)],2,2,read_latency_s=10,trace_limit=2)
        self.assertEqual(result['duration_s'],13)
        self.assertEqual(result['trace'][1]['R_request_s'],0)

    def test_invalid_latency_rejected_and_default_stays_zero(self):
        self.assertEqual(Policy().input_read_latency_s,0)
        for bad in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):Policy(input_read_latency_s=bad)
            with self.assertRaises(ValueError):operand_pipeline([(1,0,1,0,1)],1,1,1,read_latency_s=bad)


if __name__=='__main__':unittest.main()
