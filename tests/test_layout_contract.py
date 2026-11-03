"""Matrix storage semantics and prevention of cache reuse across layouts."""
import tempfile
from pathlib import Path
import unittest
from superpod_compute import GemmRequest,Policy,predict

class LayoutContractTests(unittest.TestCase):
    def test_physical_linear_storage_maps_to_logical_b(self):
        desc=GemmRequest(7,13,11,rhs_storage='nk-contiguous-transposed').storage_description()
        # W[n,k] and B[k,n] must identify the same physical element.
        for k,n in [(0,0),(10,12),(3,8)]:
            logical=k*desc['logical_B_strides_elements'][0]+n*desc['logical_B_strides_elements'][1]
            physical=n*11+k
            self.assertEqual(logical,physical)
        self.assertFalse(desc['transpose_requires_copy'])
        self.assertEqual(desc['output_shape'],[7,13])

    def test_default_storage_matches_contiguous_mm(self):
        desc=GemmRequest(7,13,11).storage_description()
        self.assertEqual(desc['logical_B_strides_elements'],[13,1])
        self.assertEqual(desc['physical_rhs_shape'],[11,13])

    def test_layout_separates_cache_signatures_without_silent_reuse(self):
        policy=Policy(bm_values=(16,),bn_values=(16,),bk_values=(16,),buffer_counts=(1,),
                      output_chunk_values=(256,),output_buffer_counts=(1,),register_buffer_counts=(1,))
        with tempfile.TemporaryDirectory() as temp:
            kw=dict(policy=policy,cache_path=Path(temp)/'cache.sqlite3')
            a=predict(GemmRequest(16,16,16),**kw)
            b=predict(GemmRequest(16,16,16,rhs_storage='nk-contiguous-transposed'),**kw)
            self.assertNotEqual(a['signature_sha256'],b['signature_sha256'])
            self.assertFalse(b['cache_hit'])
            c=predict(GemmRequest(16,16,16,rhs_storage='nk-contiguous-transposed'),**kw)
            self.assertTrue(c['cache_hit'])

    def test_unknown_storage_rejected(self):
        with self.assertRaises(ValueError):GemmRequest(1,2,3,rhs_storage='unknown')

if __name__=='__main__':unittest.main()
