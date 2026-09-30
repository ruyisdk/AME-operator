"""ELF mutation and expected-failure mailbox checks."""
import struct
import ame

def file_offset(blob, pc):
    ame.require(blob[:4] == b'\x7fELF' and blob[5:7] == b'\x01\x01', 'need LE ELF v1')
    x = {1:32, 2:64}.get(blob[4]); ame.require(x is not None, 'ELF class')
    h = struct.unpack_from('<HHIIIIIHHHHHH' if x == 32 else '<HHIQQQIHHHHHH', blob, 16)
    ame.require(h[:2] == (2,243) and 0 < h[9] < 64, 'executable RISC-V ELF required')
    fmt = '<IIIIIIII' if x == 32 else '<IIQQQQQQ'
    ame.require(h[8] == struct.calcsize(fmt) and pc % 4 == 0, 'program header/alignment')
    offsets = []
    for i in range(h[9]):
        p = struct.unpack_from(fmt,blob,h[4]+i*h[8])
        typ,off,virt,phys,size,flags = (p[0],p[1],p[2],p[3],p[4],p[6]) if x == 32 else (p[0],p[2],p[3],p[4],p[5],p[1])
        if typ == 1 and flags & 1 and virt <= pc and pc+4 <= virt+size:
            ame.require(virt == phys and off+size <= len(blob), 'unexpected executable mapping')
            offsets.append(off+pc-virt)
    ame.require(len(offsets) == 1, 'instruction must be in one file-backed executable segment')
    return offsets[0]

def patch_word(blob, pc, old, new):
    off = file_offset(blob,pc)
    ame.require(old != new and old & 3 == 3 and new & 3 == 3, '32-bit changed instruction required')
    ame.require(struct.unpack_from('<I',blob,off)[0] == old, 'opcode/ELF mismatch')
    data = bytearray(blob); struct.pack_into('<I',data,off,new)
    return bytes(data), off

def validate_response(report, spec, request, response, n):
    # Identity is checked even for a deliberately failing guest. Never accept
    # a crash, budget stop, timeout, stale response or unrelated mismatch.
    row,c=report['row'],report['config']
    exp=spec['expected'][str(n)]
    ame.require(response['schema']==1 and response['nonce']==request['nonce'],'stale response')
    ame.require(response['elf_sha256']==report['elf_sha256']==request['elf_sha256'],'wrong ELF')
    ame.require(response['observed']==request['expected'],'wrong platform')
    ame.require(response['backend']==request['backend'] and response['revision']==request['revision'],'wrong model')
    ame.require(response['stop_pc']==report['symbols']['ame_test_done'],'did not complete')
    words=struct.unpack('<32I',bytes.fromhex(response['mailbox_hex']))
    identity={0:0x414d4554,1:1,3:row['case']['id'],4:row['xlen'],5:n,
              6:row['profile']['uds'],7:row['profile']['mregs'],8:row['profile']['accregs'],9:c['seed']}
    ame.require(all(words[i]==v for i,v in identity.items()),'wrong mailbox identity')
    ame.require(words[15]&1 and not any(words[16:]),'trap or missing acquisition')
    actual=dict(status=words[2],checks=words[10],error=words[11],index=words[12],expected=words[13],actual=words[14])
    ame.require(actual==exp,'not the predicted first mismatch: '+str(actual))
    return actual
