"""Synthetic address-interpreter tests; no vendor instruction text or data."""
import unittest

from tools.diagnostics.trace_address_prologue import PrologueMachine, MASK64


def machine(*lines, abi=b'', arguments=None):
    code=[{'address':i*4,'size':4,'text':s} for i,s in enumerate(lines)]
    return PrologueMachine(code, abi, arguments or {})


class AddressPrologueTests(unittest.TestCase):
    def test_inline_float_is_bit_pattern(self):
        m=machine('s_or_b32 s8, 3, 2.0','v_mmac_stop')
        m.run()
        self.assertEqual(m.get('s8'),0x40000003)

    def test_scalar_argument_load_and_branch(self):
        m=machine('s_load_dword s8, s[0:1], 0',
                  's_cmp_eq_u32 s8, 7','s_cbranch_scc1 @0x10',
                  'unsupported_skipped','v_mmac_stop',abi=(7).to_bytes(4,'little'))
        self.assertEqual(m.run()['steps'],3)

    def test_unknown_branch_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'Unsupported branch'):
            machine('s_cbranch_unimplemented @0x4','v_mmac_stop').run()

    def test_unknown_data_register_cannot_form_address(self):
        m=machine('ds_read_b128 v[4:7], v0','v_add_u32 v8, v4, 1','v_mmac_stop')
        m.put('v[4:7]',0)
        with self.assertRaisesRegex(ValueError,'Uninitialized v4'):m.run()

    def test_vector_carry_masks(self):
        m=machine('v_add_co_u32 v3, vcc, v1, v2','v_mmac_stop')
        m.put('v1',[0xffffffff if i%2 else 1 for i in range(64)])
        m.put('v2',1);m.run()
        self.assertEqual(m.get('vcc'),sum(1<<i for i in range(1,64,2)))
        self.assertEqual(m.get('v3')[:4],[2,0,2,0])

    def test_indexed_read_addresses_and_active_mask(self):
        base=0x300000000
        m=machine('buffer_load_dwordx4 v[4:5], s[8:11], 0 idxen offen lds',
                  'v_mmac_stop',arguments={'A':base,'B':base+0x100000,'sizeA':8192,'sizeB':8192})
        m.put('s[8:11]',base+(64<<48)+(256<<64))
        m.put('v[4:5]',[i+(8<<32) for i in range(64)])
        m.put('exec',MASK64^1);m.put('m0',0x40400)
        read=m.run()['reads'][0]
        self.assertIsNone(read['addresses_bytes'][0])
        self.assertEqual(read['addresses_bytes'][1],base+64+8)
        self.assertEqual(read['lds_base_bytes'],0x40400)
        self.assertEqual(read['geometry']['requested_bytes'],63*16)

    def test_argument_bounds(self):
        with self.assertRaisesRegex(ValueError,'outside supplied ABI'):
            machine('s_load_dwordx2 s[8:9], s[0:1], 0','v_mmac_stop',abi=b'1234').run()


if __name__=='__main__':unittest.main()
