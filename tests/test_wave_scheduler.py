import math
import random
import unittest

from superpod_compute.wave_plan import WavePlan
from superpod_compute.wave_scheduler import wave_operand_pipeline
from superpod_compute.scheduler import Event,EventScheduler,operand_pipeline


def reference(segments,lb,sb,rb,plan,engines,latency):
    events=[];i=0;waves=plan.waves
    for count,h,r,s,c in segments:
        for _ in range(count):
            events.append(Event(f'H{i}',h,('H',),(f'R{i-lb}',) if i>=lb else ()))
            released=tuple(f'S{i-sb}:{w}' for w in range(waves)) if i>=sb else ()
            events.append(Event(f'Q{i}',latency,(),(f'H{i}',)+released))
            events.append(Event(f'R{i}',r,('R',),(f'Q{i}',)))
            for w in range(waves):
                previous=tuple(f'C{i-rb}:{v}' for v in range(waves)) if plan.read_waits_for_group_release else (f'C{i-rb}:{w}',)
                events.append(Event(f'S{i}:{w}',s/waves,('S',),(f'R{i}',)+(previous if i>=rb else ())))
            barrier=tuple(f'C{i-1}:{w}' for w in range(waves)) if i else ()
            if plan.barrier_after_reads:barrier+=tuple(f'S{i}:{w}' for w in range(waves))
            events.append(Event(f'B{i}',0,(),barrier))
            for w in range(waves):
                events.append(Event(f'C{i}:{w}',c/math.ceil(waves/engines),(f'engine{w%engines}',),(f'B{i}',f'S{i}:{w}')))
            i+=1
    return EventScheduler().run(events)


class WaveSchedulerTests(unittest.TestCase):
    def test_single_wave_matches_existing_register_pipeline(self):
        plan=WavePlan(16,16,16,16)
        segments=[(31,.04,.03,.1,.3),(3,0,.02,.05,.1)]
        for rb in (1,2):
            a=wave_operand_pipeline(segments,2,3,rb,plan,4,read_latency_s=.2)
            b=operand_pipeline(segments,2,3,rb,read_latency_s=.2)
            self.assertTrue(math.isclose(a['duration_s'],b['duration_s'],rel_tol=1e-12))

    def test_single_buffer_per_wave_does_not_serialize_whole_cu(self):
        plan=WavePlan(64,32,16,16)
        a=wave_operand_pipeline([(2,0,0,8,4)],2,3,1,plan,4,trace_limit=2)
        self.assertEqual(a['duration_s'],22)
        first,second=a['trace']
        self.assertLess(second['wave_reads'][0][0],first['C_end_s'])
        for w in range(8):
            self.assertGreaterEqual(second['wave_reads'][w][0],first['wave_compute'][w][1])

    def test_explicit_event_graph_equivalence(self):
        rng=random.Random(1043)
        for waves in (1,4,8):
            for barrier in (False,True):
                for group_release in (False,True):
                    for rb in (1,2):
                        plan=WavePlan(16*waves,16,16,16,barrier_after_reads=barrier,read_waits_for_group_release=group_release)
                        stages=[(19,*(rng.random()*1e-6 for _ in range(4))),(3,0,.01e-6,.03e-6,.09e-6)]
                        a=wave_operand_pipeline(stages,2,6,rb,plan,4,read_latency_s=.2e-6)
                        b=reference(stages,2,6,rb,plan,4,.2e-6)
                        self.assertTrue(math.isclose(a['duration_s'],max(e['end_s'] for e in b),rel_tol=1e-12))


if __name__=='__main__':unittest.main()
