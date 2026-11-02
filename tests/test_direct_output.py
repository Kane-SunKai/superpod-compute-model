import math
import random
import unittest
from dataclasses import replace

from superpod_compute import Hardware,Policy,Mapping,GemmRequest,evaluate_mapping,predict
from superpod_compute.operators import GemmModel
from superpod_compute.scheduler import direct_output_pipeline,Event,EventScheduler


class DirectOutputTests(unittest.TestCase):
    def test_direct_pipeline_matches_two_stage_event_graph(self):
        rng=random.Random(204)
        for slots in (1,2,3):
            for shared_port in (False,True):
                segments=[(rng.randint(1,8),rng.random()*1e-6,rng.random()*1e-6) for _ in range(4)]
                events=[];index=0
                for count,load,drain in segments:
                    for _ in range(count):
                        events.append(Event(f'L{index}',load,('port' if shared_port else 'L2',),
                                            (f'D{index-slots}',) if index>=slots else ()))
                        events.append(Event(f'D{index}',drain,('port' if shared_port else 'HBM',),(f'L{index}',)))
                        index+=1
                expected=EventScheduler().run(events)[-1]['end_s']
                actual=direct_output_pipeline(segments,slots,shared_port=shared_port,trace_limit=100)
                self.assertTrue(math.isclose(actual['duration_s'],expected,rel_tol=1e-12))
                self.assertFalse(any('shared_slot' in r or 'W_start_s' in r for r in actual['trace']))

    def test_direct_pipeline_acceleration_keeps_tail(self):
        segments=[(2000,2e-6,3e-6),(1,.3e-6,.4e-6)]
        fast=direct_output_pipeline(segments,3)
        slow=direct_output_pipeline(segments,3,accelerate=False)
        self.assertTrue(math.isclose(fast['duration_s'],slow['duration_s'],rel_tol=1e-12))
        self.assertGreater(fast['skipped_panels'],1000)

    def test_direct_output_is_independent_of_shared_write_bandwidth(self):
        h=Hardware.load('superpod_compute/data/bw1100.json')
        mapping=Mapping(256,384,16,12,6,1,2,4096,2,2,output_route='direct-register')
        request=GemmRequest(3072,2304,1024)
        a=evaluate_mapping(request,mapping,h)
        b=evaluate_mapping(request,mapping,replace(h,shared_write_bytes_per_cycle_per_core=.01))
        self.assertEqual(a['duration_s'],b['duration_s'])
        self.assertEqual(a['capacity']['shared_output_bytes'],0)
        for wave in a['wave_classes']:
            output=wave['output_stream']
            self.assertEqual(output['register_to_shared_service_s'],0)
            self.assertEqual(output['shared_output_limit_per_core'],0)
            self.assertEqual(output['pipeline']['stages'],['register-to-L2','L2-read/HBM-drain'])
        via=evaluate_mapping(request,replace(mapping,output_route='via-shared'),h)
        for key in ('hbm_read_bytes','l2_to_core_bytes','output_bytes','executed_tile_flops'):
            self.assertEqual(a[key],via[key])
        self.assertLessEqual(a['duration_s'],via['duration_s'])

    def test_direct_route_removes_only_shared_output_capacity(self):
        h=replace(Hardware(),source_sram_bytes_per_core=8*1024,shared_budget_bytes_per_core=8*1024)
        mapping=Mapping(128,128,16,1,1,1,1,16384,2)
        via=GemmModel.capacity(mapping,h)
        direct=GemmModel.capacity(replace(mapping,output_route='direct-register'),h)
        self.assertIn('shared',via['rejections']);self.assertNotIn('shared',direct['rejections'])
        self.assertEqual(via['l2_bytes'],direct['l2_bytes'])
        self.assertEqual(via['register_bytes'],direct['register_bytes'])

    def test_policy_search_and_signature_preserve_route(self):
        p=Policy(bm_values=(16,),bn_values=(16,),bk_values=(16,),buffer_counts=(1,),
                 output_buffer_counts=(1,),output_chunk_values=(512,),add_shape_candidate=False,
                 output_route='direct-register')
        a=predict(GemmRequest(16,16,16),policy=p,backend='peak')
        b=predict(GemmRequest(16,16,16),policy=replace(p,output_route='via-shared'),backend='peak')
        self.assertEqual(a['best']['mapping']['output_route'],'direct-register')
        self.assertNotEqual(a['signature_sha256'],b['signature_sha256'])
        with self.assertRaises(ValueError):replace(p,output_route='unknown')
        with self.assertRaises(ValueError):Mapping(16,16,16,1,1,1,1,output_route='unknown')


if __name__=='__main__':unittest.main()
