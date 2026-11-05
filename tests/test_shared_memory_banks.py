import unittest
from superpod_compute.shared_memory import SharedReadBanks,cdna_b128_read_banks


class SharedReadBankTests(unittest.TestCase):
    def test_contiguous_request_has_eight_service_phases(self):
        b=cdna_b128_read_banks();e=b.estimate(tuple(i*16 for i in range(64)))
        self.assertEqual(e.cycles,8)
        self.assertEqual(e.phase_cycles,(1,)*8)
        self.assertEqual(e.requested_bytes,1024)
        self.assertEqual(e.serviced_bytes,1024)

    def test_changed_layout_changes_time_without_changing_requested_bytes(self):
        b=cdna_b128_read_banks()
        plain=b.estimate(tuple(i*16 for i in range(64)))
        strided=b.estimate(tuple(((i%16)*32+i//16)%512*16 for i in range(64)))
        self.assertEqual(strided.cycles,32)
        self.assertEqual(strided.requested_bytes,plain.requested_bytes)
        self.assertEqual(strided.serviced_bytes,plain.serviced_bytes)

    def test_same_word_broadcast_is_limited_to_its_phase(self):
        e=cdna_b128_read_banks().estimate((0,)*64)
        self.assertEqual(e.cycles,8)
        self.assertEqual(e.serviced_bytes,128)
        self.assertEqual(e.requested_bytes,1024)

    def test_bank_base_rotation_preserves_conflict_structure(self):
        b=cdna_b128_read_banks();a=tuple(i*16 for i in range(64))
        self.assertEqual(b.estimate(a).cycles,b.estimate(tuple(x+16 for x in a)).cycles)

    def test_rejects_malformed_hardware_and_requests(self):
        with self.assertRaises(ValueError):SharedReadBanks(32,4,4,16,((0,1),(1,3)))
        with self.assertRaises(ValueError):cdna_b128_read_banks().estimate((0,)*63)
        with self.assertRaises(ValueError):cdna_b128_read_banks().estimate(tuple(range(64)))
        with self.assertRaises(TypeError):cdna_b128_read_banks().estimate([0]*64)


if __name__=='__main__':unittest.main()
