import dataclasses,unittest
from superpod_compute import Hardware,Policy,Mapping,GemmRequest,evaluate_mapping,predict
from superpod_compute.operand_layout import routed_operands
from superpod_compute.shared_memory import cdna_b128_read_banks

class OperandLayoutTests(unittest.TestCase):
    def hw(self):
        b=cdna_b128_read_banks()
        return dataclasses.replace(Hardware(),matrix_backend='mmac',cores=72,
            shared_budget_bytes_per_core=65536,register_budget_bytes_per_core=768*1024,
            shared_read_bytes_per_cycle_per_core=128,shared_write_bytes_per_cycle_per_core=60,
            shared_bank_count=32,shared_read_phase_lanes=b.phase_lanes)
    def test_copies_and_padding_are_counted_in_storage_and_reads(self):
        h=self.hw();x=routed_operands(128,128,16,h)
        self.assertEqual(x['shared_bytes'],16384)
        self.assertEqual(x['shared_read_bytes'],16384)
        self.assertEqual(x['shared_read_cycles'],128)
        self.assertEqual(sum(x['register_bytes_by_engine']),x['shared_bytes'])
        # An odd number of 8-byte lane operands requires padded 16-byte packets.
        y=routed_operands(16,384,16,h)
        self.assertEqual(y['shared_bytes'],16384)
    def test_shared_storage_rejects_unbacked_double_buffer(self):
        h=self.hw()
        self.assertTrue(evaluate_mapping(GemmRequest(256,384,32),Mapping(256,384,16,1,1,1,2),h)['capacity']['valid'])
        from superpod_compute.operators import GemmModel
        self.assertFalse(GemmModel.capacity(Mapping(256,384,16,1,1,1,2,operand_layout='engine-routed'),h)['valid'])
        self.assertTrue(GemmModel.capacity(Mapping(256,384,16,1,1,1,1,operand_layout='engine-routed'),h)['valid'])
    def test_routed_layout_uses_bank_model_not_old_fitted_rate(self):
        h=self.hw();r=GemmRequest(128,128,64);m=Mapping(128,128,16,1,1,2,2,operand_layout='engine-routed')
        a=evaluate_mapping(r,m,h,policy=Policy(effective_shared_read_bytes_per_cycle_per_core=1))
        b=evaluate_mapping(r,m,h,policy=Policy(effective_shared_read_bytes_per_cycle_per_core=120))
        self.assertEqual(a['duration_s'],b['duration_s'])
        self.assertEqual(a['wave_classes'][0]['panel_byte_runs'][0]['Shared_read_service_s_per_core'],128/h.clock_hz)
    def test_l2_traffic_does_not_duplicate_with_shared_replication(self):
        h=self.hw();r=GemmRequest(128,128,64);p=Policy(charge_input_shared_write=True)
        a=evaluate_mapping(r,Mapping(128,128,16,1,1,1,1),h,policy=p)
        b=evaluate_mapping(r,Mapping(128,128,16,1,1,1,1,operand_layout='engine-routed'),h,policy=p)
        self.assertEqual(a['hbm_read_bytes'],b['hbm_read_bytes'])
        self.assertEqual(a['l2_to_core_bytes'],b['l2_to_core_bytes'])
        self.assertEqual(a['output_bytes'],b['output_bytes'])
        self.assertGreater(b['wave_classes'][0]['panel_byte_runs'][0]['Shared_input_write_service_s_per_core'],a['wave_classes'][0]['panel_byte_runs'][0]['Shared_input_write_service_s_per_core'])
    def test_new_request_layout_is_part_of_contract(self):
        a=GemmRequest(16,32,64);b=GemmRequest(16,32,64,layout='right-transposed-contiguous')
        self.assertNotEqual(a.to_dict(),b.to_dict())
    def test_k_tail_storage_matches_instruction_padding(self):
        h=self.hw()
        self.assertEqual(routed_operands(128,128,1,h),routed_operands(128,128,16,h))
        self.assertEqual(routed_operands(128,128,17,h),routed_operands(128,128,32,h))
    def test_invalid_layout_is_rejected(self):
        with self.assertRaises(ValueError):Mapping(16,16,16,1,1,1,1,operand_layout='unknown')
        with self.assertRaises(ValueError):Policy(operand_layouts=('engine-routed','engine-routed'))

if __name__=='__main__':unittest.main()
