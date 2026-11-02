"""Synthetic control-flow tests; no vendor text or device data."""
import unittest
from tools.diagnostics.trace_kernel_control import ControlMachine


def machine(*lines):
    return ControlMachine([{'address':i*4,'size':4,'text':s} for i,s in enumerate(lines)],b'',{})


class KernelControlTests(unittest.TestCase):
    def test_matrix_loop_and_barrier_count(self):
        m=machine('s_mov_b32 s8, 3',
                  'v_mmac_f32_16x16x16_f16 v[4:7], v[8:9], v[10:11], v[4:7]',
                  's_barrier','s_sub_u32 s8, s8, 1','s_cmpk_gt_u32 s8, 0',
                  's_cbranch_scc1 @0x4','v_cvt_f16_f32 v12, v4')
        result=m.run_control()
        self.assertEqual(result['matrix_instructions'],3)
        self.assertEqual(result['opcode_counts']['s_barrier'],3)
        self.assertEqual([r['matrix_instructions_before'] for r in result['events']],[1,2,3])

    def test_tensor_data_cannot_become_branch_input(self):
        m=machine('v_mmac_f32_16x16x16_f16 v[4:7], v[8:9], v[10:11], v[4:7]',
                  'v_cmp_eq_u32 vcc, v4, 0','v_cvt_f16_f32 v12, v4')
        with self.assertRaisesRegex(ValueError,'Uninitialized v4'):m.run_control()

    def test_known_half_scalar_conversion(self):
        m=machine('v_mov_b32 v8, 15360','v_cvt_f32_f16 v9, v8',
                  'v_mmac_f32_16x16x16_f16 v[4:7], v[8:9], v[10:11], v[4:7]',
                  'v_cvt_f16_f32 v12, v4')
        m.run_control();self.assertEqual(m.get('v9'),[0x3f800000]*64)

    def test_logical_scc_controls_branch(self):
        m=machine('s_mov_b32 s8, 7','s_and_b32 s9, s8, 0',
                  's_cbranch_scc0 @0x10','unsupported_skipped',
                  'v_mmac_f32_16x16x16_f16 v[4:7], v[8:9], v[10:11], v[4:7]',
                  'v_cvt_f16_f32 v12, v4')
        self.assertEqual(m.run_control()['matrix_instructions'],1)


if __name__=='__main__':unittest.main()
