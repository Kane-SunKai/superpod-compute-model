import dataclasses,json,pathlib,unittest
from superpod_compute import GemmRequest,Hardware,Policy,Mapping
from superpod_compute.operators import GemmModel
from superpod_compute.backends import make_backend
from superpod_compute.batched import predict_batched_gemm

class PanelServiceTests(unittest.TestCase):
    def hardware(self):
        path=pathlib.Path(__file__).resolve().parents[1]/'superpod_compute/data/bw1100_decode_profile.json'
        return Hardware(**json.loads(path.read_text())['hardware'])
    def test_zero_service_options_preserve_frozen_baseline(self):
        path=pathlib.Path(__file__).resolve().parents[1]/'superpod_compute/data/bw1100_decode_profile.json'
        f=json.loads(path.read_text());h=Hardware(**f['hardware']);p=Policy(**f['policy'])
        # Frozen public regression fixtures require no private experiment files.
        fixtures=[((1,4608,3584),(16,64,16,1,72,2,1,2048,1,1),22.424330150892114),
                  ((1280,4096,4096),(256,384,16,5,11,1,2,4096,2,2),161.3836812792585),
                  ((1024,12288,4096),(256,384,16,4,18,1,2,4096,2,2),322.23488159480064)]
        for shape,mapping,expected_us in fixtures:
            value=GemmModel().evaluate(GemmRequest(*shape),h,p,make_backend('mmac',h),Mapping(*mapping))
            self.assertAlmostEqual(value['duration_us'],expected_us,places=9)
    def test_k_control_service_has_cycles_and_no_extra_data(self):
        h=self.hardware();model=GemmModel();p=Policy(k_loop_control_cycles_per_step=40)
        self.assertAlmostEqual(model.panel_control_time(32,h,p),80/h.clock_hz)
        req=GemmRequest(256,256,512);mp=Mapping(256,256,16,1,1,2,2)
        a=model.evaluate(req,h,Policy(),make_backend('mmac',h),mp)
        b=model.evaluate(req,h,p,make_backend('mmac',h),mp)
        self.assertEqual(a['hbm_read_bytes'],b['hbm_read_bytes']);self.assertEqual(a['output_bytes'],b['output_bytes'])
        self.assertGreater(b['duration_s'],a['duration_s'])
    def test_input_store_can_limit_input_without_changing_output(self):
        h=dataclasses.replace(self.hardware(),shared_write_bytes_per_cycle_per_core=1)
        req=GemmRequest(128,128,128);mp=Mapping(128,128,16,1,1,2,2);model=GemmModel()
        a=model.evaluate(req,h,Policy(),make_backend('mmac',h),mp)
        b=model.evaluate(req,h,Policy(charge_input_shared_write=True),make_backend('mmac',h),mp)
        self.assertGreater(b['duration_s'],a['duration_s'])
        self.assertEqual(a['wave_classes'][0]['output_stream'],b['wave_classes'][0]['output_stream'])
    def test_batch_members_never_share_an_operand(self):
        h=self.hardware();p=Policy(bm_values=(128,),bn_values=(128,),bk_values=(16,),add_shape_candidate=False)
        a=predict_batched_gemm(1,128,128,16,h,p)['best'];b=predict_batched_gemm(2,128,128,16,h,p)['best']
        self.assertEqual(b['hbm_read_bytes'],2*a['hbm_read_bytes'])
        self.assertEqual(b['output_bytes'],2*a['output_bytes'])
        self.assertEqual(b['output_tasks'],2*a['output_tasks'])
    def test_new_policy_parameters_reject_invalid_values(self):
        for kw in [dict(charge_input_shared_write=1),dict(k_loop_control_cycles_per_step=-1),dict(effective_shared_read_bytes_per_cycle_per_core=float('nan'))]:
            with self.assertRaises(ValueError):Policy(**kw)

if __name__=='__main__':unittest.main()
