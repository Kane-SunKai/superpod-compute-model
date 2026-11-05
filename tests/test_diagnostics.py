import dataclasses
import unittest
from superpod_compute import GemmRequest, Hardware, Policy, Mapping, evaluate_mapping, predict
from superpod_compute.diagnostics import ObservedTile, diagnose_tile, service_ledger


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.h=dataclasses.replace(Hardware(), matrix_backend='mmac',
                                  shared_read_bytes_per_cycle_per_core=64,
                                  shared_write_bytes_per_cycle_per_core=64)
        self.p=Policy(bm_values=(16,32),bn_values=(16,32),bk_values=(16,32),
                      charge_input_shared_write=True,k_loop_control_cycles_per_step=40,
                      effective_shared_read_bytes_per_cycle_per_core=60)

    def test_diagnosis_does_not_change_policy_or_normal_prediction(self):
        req=GemmRequest(32,64,48)
        before=predict(req,self.h,policy=self.p)
        result=diagnose_tile(req,ObservedTile(16,16,16),self.h,policy=self.p,trace_limit=2)
        after=predict(req,self.h,policy=self.p)
        self.assertEqual(before,after)
        self.assertEqual(result['alignment'],'tile-only')
        self.assertEqual(result['purpose'],'diagnostic')
        self.assertEqual(self.p.bm_values,(16,32))
        best=result['prediction']['best']
        self.assertEqual([best['mapping'][k] for k in ('bm','bn','bk')],[16,16,16])

    def test_ledger_reconstructs_time_with_tail_and_reuse(self):
        req=GemmRequest(17,140,47)
        mapping=Mapping(16,32,16,1,2,2,2)
        result=evaluate_mapping(req,mapping,self.h,policy=self.p)
        ledger=service_ledger(req,result,self.h,self.p)
        self.assertAlmostEqual(ledger['elapsed_s'],result['duration_s'],places=15)
        self.assertGreaterEqual(ledger['all_input_stages_serial_s']+1e-15,result['duration_s'])
        # Exact MMAC/control demand across the uneven output waves and tail K.
        expected=sum(w['count']*((w['full_panel_compute']['duration_s']+w['full_panel_control_s'])*w['full_k_panels']+w['tail_panel_compute']['duration_s']+w['tail_panel_control_s']) for w in result['wave_classes'])
        self.assertAlmostEqual(ledger['service_demand_s']['C'],expected,places=15)

    def test_scalar_service_units(self):
        req=GemmRequest(16,16,16)
        result=evaluate_mapping(req,Mapping(16,16,16,1,1,1,1),self.h,policy=self.p)
        ledger=service_ledger(req,result,self.h,self.p)
        self.assertAlmostEqual(ledger['service_demand_s']['S'],result['wave_classes'][0]['full_panel_compute']['shared_read_bytes']/(60*self.h.clock_hz))
        self.assertAlmostEqual(ledger['service_demand_s']['C'],48/self.h.clock_hz)

    def test_plan_rejects_invalid_dimensions(self):
        for args in ((0,16,16),(16,-1,16),(16,16,True)):
            with self.assertRaises(ValueError):ObservedTile(*args)


if __name__=='__main__':unittest.main()
