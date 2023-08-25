import zlib
import lzma

def compress_lzma_len(s):
    """Compress a byte string using LZMA, maximum compression, no header or checksum.
    Return the length of the compressed string."""
    filters = [
        {"id": lzma.FILTER_DELTA, "dist": 5},
        {"id": lzma.FILTER_LZMA2, "preset": 7 | lzma.PRESET_EXTREME},
    ]
    return len(lzma.compress(s, format=lzma.FORMAT_RAW, filters=filters))

def compress_zlib_len(s):
    """Compress a byte string s using zlib, maximum compression, no header or checksum. 
    Return the length of the compressed string."""
    return len(zlib.compress(s, wbits=-15, level=9))

def ncd(a, b):
    """Compute the normalised compression distance between a and b
    https://en.wikipedia.org/wiki/Normalized_compression_distance
    """
    z_ab = compress_len(a+b)
    z_a = compress_len(a)
    z_b = compress_len(b)
    return (z_ab - min(z_a, z_b)) / max(z_a, z_b)

def integer_sequence_to_bytes(ls):
    """Encode a sequence of integers as a byte string. 
    If all integers are in the range 0-255, each is encoded as a byte;
    if in range 0-65535 as byte pairs, in the platform endianness."""
    
    if len(ls)==0 or max(ls)<255:
        s = [int(l).to_bytes(1) for l in ls]
    else:
        s = [int(l).to_bytes(2) for l in ls]
    return b''.join(s)

def compress_len(ls, mode="zlib"):
    """Return the compressed length of the
       integer sequence ls. 
       Mode can be either "zlib" or "lzma"
    """
    s = integer_sequence_to_bytes(ls)
    if mode=="zlib":
        return compress_zlib_len(s)
    elif mode=="lzma":
        return compress_lzma_len(s)

def normalized_compress_len(ls, mode="zlib"):
    """Return a compression ratio from 0.0 to 1.0, 
    from a sequence of integers ls, accounting for
    the compression of an empty sequence (e.g. header data)
    mode can be anything compress_len takes ("zlib" or "lzma")
    """    
    return len(ls) / ((compress_len(ls, mode) ) - (compress_len([], mode))) 