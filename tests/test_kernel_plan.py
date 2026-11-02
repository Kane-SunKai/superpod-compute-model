import unittest
from superpod_compute import GemmRequest,Hardware,Policy
from superpod_compute.kernel_plan import parse_tensile_kernel,evaluate_macrotiling_ablation


def kernel(mt='128x32x16',gsu=1,wg='16_8_1'):
    return f'Cijk_Ailk_Bljk_HBH_MT{mt}_DTL0_GRVW4_GSU{gsu}_PGR2_PLR1_WG{wg}_WGM1'


class KernelPlanTests(unittest.TestCase):
    def test_row_major_blas_axes_and_hip_global_grid(self):
        plan=parse_tensile_kernel(kernel(),orientation='row-major-transposed-blas')
        request=GemmRequest(257,4093,769)
        self.assertEqual((plan.bm,plan.bn),(32,128))
        self.assertEqual(plan.coverage(request,72)['main_kernel_workgroups'],288)
        self.assertTrue(plan.validate_hip_grid(request,[4096,9,1],[128,1,1])['matches'])
        wrong=parse_tensile_kernel(kernel(),orientation='logical-mn')
        self.assertFalse(wrong.validate_hip_grid(request,[4096,9,1],[128,1,1])['matches'])

    def test_crossing_a_single_cu_round(self):
        plan=parse_tensile_kernel(kernel('128x192x16',wg='16_16_1'),orientation='row-major-transposed-blas')
        before=plan.coverage(GemmRequest(1536,1152,1024),72)
        after=plan.coverage(GemmRequest(1537,1152,1024),72)
        self.assertEqual(before['main_kernel_workgroups'],72)
        self.assertEqual(after['main_kernel_workgroups'],81)
        self.assertEqual(after['boundary_output_workgroups'],9)
        self.assertEqual(before['one_resident_block_per_cu_rounds'],1)
        self.assertEqual(after['one_resident_block_per_cu_rounds'],2)

    def test_split_k_is_not_silently_treated_as_resident_c(self):
        plan=parse_tensile_kernel(kernel('384x128x32',8,'32_16_1'),orientation='row-major-transposed-blas')
        req=GemmRequest(512,768,16384)
        self.assertEqual(plan.coverage(req,72)['main_kernel_workgroups'],64)
        self.assertTrue(plan.validate_hip_grid(req,[1024,32,1],[512,1,1])['matches'])
        with self.assertRaises(NotImplementedError):
            evaluate_macrotiling_ablation(req,Hardware(),Policy(),plan)

    def test_orientation_and_incomplete_names_fail_closed(self):
        with self.assertRaises(TypeError):parse_tensile_kernel(kernel())
        with self.assertRaises(ValueError):parse_tensile_kernel('unknown',orientation='logical-mn')
        with self.assertRaises(ValueError):parse_tensile_kernel(kernel(),orientation='guess')

    def test_axis_tail_and_empty_output(self):
        plan=parse_tensile_kernel(kernel(),orientation='row-major-transposed-blas')
        coverage=plan.coverage(GemmRequest(33,129,17),72)
        self.assertEqual(coverage['full_output_workgroups'],1)
        self.assertEqual(coverage['boundary_output_workgroups'],3)
        self.assertEqual(coverage['k_remainder_depth_u'],1)
        self.assertEqual(plan.coverage(GemmRequest(0,129,17),72)['main_kernel_workgroups'],0)


if __name__=='__main__':unittest.main()
