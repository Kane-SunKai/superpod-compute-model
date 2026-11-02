import unittest

from superpod_compute.memory_access import RowMajorStorage,describe_read


class MemoryAccessTests(unittest.TestCase):
    def test_odd_fp16_k_alternates_word_alignment(self):
        a=RowMajorStorage(1537,1023,1023)
        self.assertEqual(a.row_start_residues(4),{0:769,2:768})
        padded=RowMajorStorage(1537,1023,1024)
        self.assertEqual(padded.logical_bytes,a.logical_bytes)
        self.assertEqual(padded.row_start_residues(4),{0:1537})
        self.assertEqual(padded.address_span_bytes-a.address_span_bytes,1536*2)

    def test_offset_changes_alignment_without_changing_shape_or_stride(self):
        a=RowMajorStorage(1536,1024,1024,base_mod_256_bytes=2)
        self.assertEqual(a.row_start_residues(4),{2:1536})
        self.assertEqual(a.address(2,16),2+2*2048+32)

    def test_periodic_histogram_matches_direct_addresses(self):
        for stride in (7,8,9,1023,1024,1025):
            for rows in (1,7,129):
                a=RowMajorStorage(rows,7,stride,base_mod_256_bytes=6)
                for alignment in (4,16,64,256):
                    expected={}
                    for r in range(rows):
                        key=a.address(r,0)%alignment;expected[key]=expected.get(key,0)+1
                    self.assertEqual(a.row_start_residues(alignment),expected)
        self.assertEqual(RowMajorStorage(10**12,8,8).row_start_residues(16),{0:10**12})

    def test_masks_and_duplicate_addresses_preserve_geometry_not_transactions(self):
        d=describe_read([0,0,60,None],8,sector_bytes=64)
        self.assertEqual(d['requested_bytes'],24)
        self.assertEqual(d['unique_address_bytes'],16)
        self.assertEqual(d['unique_address_sectors'],2)
        self.assertEqual(d['sum_per_lane_address_sectors'],4)
        self.assertEqual(d['sector_crossing_lanes'],1)
        self.assertEqual(d['word_unaligned_lanes'],0)
        self.assertEqual(d['vector_unaligned_lanes'],1)

    def test_geometry_agrees_with_explicit_byte_enumeration(self):
        for width in (2,4,8,16):
            for stride in (2,4,16,62,64,2046):
                addresses=[2+i*stride if i%7 else None for i in range(64)]
                touched={b for a in addresses if a is not None for b in range(a,a+width)}
                value=describe_read(addresses,width)
                self.assertEqual(value['unique_address_bytes'],len(touched))
                self.assertEqual(value['unique_address_sectors'],len({b//64 for b in touched}))

    def test_rejects_unknown_alignment_and_bad_views(self):
        with self.assertRaises(ValueError):RowMajorStorage(2,8,7)
        with self.assertRaises(ValueError):RowMajorStorage(2,8,8,base_mod_256_bytes=1)
        with self.assertRaises(ValueError):RowMajorStorage(2,8,8).row_start_residues(512)
        with self.assertRaises(ValueError):describe_read([0]*65,16)
        self.assertEqual(describe_read([None]*64,16)['requested_bytes'],0)
        self.assertEqual(RowMajorStorage(0,8,8).address_span_bytes,0)

if __name__=='__main__':unittest.main()
