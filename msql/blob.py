"""msgpack (de)serialisation of the IR blob columns (ir_rich / ir_grid). The one encoder every blob goes
through, so producer and consumer never disagree on the encoding. Format-only: it packs/unpacks any
msgpack-serialisable object; the blob SHAPE is the caller's (dio) concern."""
import msgpack


def pack_ir(obj) -> bytes:
    """Serialise an ir_rich / ir_grid dict to a msgpack blob"""
    return msgpack.packb(obj)


def unpack_ir(blob: bytes):
    """Deserialise a msgpack IR blob back to its dict (str keys, not bytes)"""
    return msgpack.unpackb(blob, raw=False)
