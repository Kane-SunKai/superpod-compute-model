import unittest
from superpod_compute.operators import group_geometries

class GeometryFrontierTests(unittest.TestCase):
    def test_full_seventy_tile_grid_fits_seventy_two_cus(self):
        self.assertIn((10,7),group_geometries(10,7,72))

    def test_non_divisor_group_reduces_square_batch_count(self):
        self.assertIn((5,14),group_geometries(14,14,72))
        self.assertIn((14,5),group_geometries(14,14,72))

    def test_every_group_respects_physical_limits_and_tile_grid(self):
        for cores in [1,4,7,72,108]:
            for tm,tn in [(1,100),(10,7),(14,14),(100,1),(11,13)]:
                for gm,gn in group_geometries(tm,tn,cores):
                    self.assertLessEqual(gm,tm);self.assertLessEqual(gn,tn)
                    self.assertLessEqual(gm*gn,cores)
                    self.assertGreaterEqual(min(gm,gn),1)

    def test_frontier_covers_every_legal_rectangle(self):
        # Every legal grouping is dominated by a returned maximal grouping;
        # this checks coverage independently of the frontier implementation.
        for tm,tn,cores in [(10,7,72),(14,14,72),(11,13,17)]:
            groups=group_geometries(tm,tn,cores)
            for m in range(1,tm+1):
                for n in range(1,tn+1):
                    if m*n<=cores:self.assertTrue(any(gm>=m and gn>=n for gm,gn in groups))

if __name__=='__main__':unittest.main()
