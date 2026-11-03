import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from superpod_compute import Hardware,Policy,GemmRequest,Mapping,predict,evaluate_mapping
from superpod_compute.contracts import digest
from tools.diagnostics.experimental_stage_services import ServiceProfile,IDENTITY

class ServiceProfileTests(unittest.TestCase):
    def setUp(self):
        self.hw=replace(Hardware.load(Path(__file__).resolve().parents[1]/'superpod_compute/data/bw1100.json'),
                        kernel_launch_s=0)
        self.request=GemmRequest(32,32,64,rhs_storage='nk-contiguous-transposed')
        self.policy=Policy(bm_values=(16,),bn_values=(16,),bk_values=(16,),buffer_counts=(1,),
                           output_chunk_values=(256,),output_buffer_counts=(1,),register_buffer_counts=(1,))
    def profile(self,scale=1):
        return ServiceProfile(dict(schema='software-stage-services-1',hardware_sha256=digest(self.hw.to_dict()),
            dtype='fp16',rhs_storage='nk-contiguous-transposed',runtime='test-fixture, not a measurement',
            points=[dict(id='synthetic-calibration',shape=[32,32,64],role='calibration',family='test',
                         factors={**IDENTITY,'matrix_scale':scale})]))

    def test_identity_profile_preserves_existing_pipeline(self):
        mapping=Mapping(16,16,16,2,2,1,1,256,1,1)
        plain=evaluate_mapping(self.request,mapping,self.hw,policy=self.policy,trace_limit=0)
        prof=self.profile().evaluate(self.request,mapping,self.hw,self.policy,trace_limit=0)
        self.assertEqual(plain['duration_s'],prof['duration_s'])
        self.assertEqual(plain['hbm_read_bytes'],prof['hbm_read_bytes'])
        self.assertEqual(plain['capacity'],prof['capacity'])

    def test_software_profile_never_mutates_hardware(self):
        old=self.hw.to_dict()
        self.profile(2).evaluate(self.request,Mapping(16,16,16,2,2,1,1,256,1,1),self.hw,self.policy)
        self.assertEqual(old,self.hw.to_dict())

    def test_mismatched_hardware_and_layout_are_rejected(self):
        p=self.profile()
        with self.assertRaises(ValueError):p.validate(self.request,replace(self.hw,clock_hz=self.hw.clock_hz/2))
        with self.assertRaises(ValueError):p.validate(GemmRequest(32,32,64),self.hw)
        with self.assertRaises(ValueError):p.validate(replace(self.request,dtype='bf16'),self.hw)

    def test_profiles_do_not_contaminate_mapping_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            kw=dict(policy=self.policy,cache_path=Path(tmp)/'map.sqlite3')
            normal=predict(self.request,self.hw,**kw)
            a=self.profile(2).apply(normal,self.request,self.hw,self.policy)
            b=self.profile(3).apply(normal,self.request,self.hw,self.policy)
            again=predict(self.request,self.hw,**kw)
            self.assertNotEqual(a['signature_sha256'],b['signature_sha256'])
            self.assertEqual(normal['best']['duration_s'],again['best']['duration_s'])
            self.assertEqual(normal['best']['duration_s'],a['hardware_plan_best']['duration_s'])

    def test_unphysical_matrix_speedup_rejected(self):
        with self.assertRaises(ValueError):self.profile(.5)

if __name__=='__main__':unittest.main()
