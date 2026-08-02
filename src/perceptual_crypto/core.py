"""Minimal protected-recovery core for the Article18 reviewer package."""

from __future__ import annotations

import base64
import hashlib
import hmac

import numpy as np

KEYED_FE_SCHEME = "keyed-bounded-rep-window-v1"
KEYED_FE_CTX = b"protected-perceptual-matching-fe-v1"


def bits_to_bytes(bs: np.ndarray) -> bytes:
    bs = np.asarray(bs, dtype=np.uint8) & 1
    pad = (-len(bs)) % 8
    if pad:
        bs = np.concatenate([bs, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bs, bitorder="big").tobytes()


def bytes_to_bits(data: bytes, n_bits: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="big")[:n_bits].astype(np.uint8)


def pack_bits_b64(bs: np.ndarray) -> str:
    return base64.b64encode(bits_to_bytes(bs)).decode("ascii")


def unpack_bits_b64(s: str, n_bits: int) -> np.ndarray:
    return bytes_to_bits(base64.b64decode(s), n_bits)


def hkdf_extract_expand(key_material: bytes, info: bytes = KEYED_FE_CTX, out_len: int = 32) -> bytes:
    prk = hmac.new(b"\x00" * 32, key_material, hashlib.sha256).digest()
    t = b""
    okm = b""
    counter = 1
    while len(okm) < out_len:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:out_len]


def hmac_sha256_hex(key: bytes, msg: bytes) -> str:
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def fe_block_indices(n: int, blocks: int, block_width: int, overlap: int = 0, index_seed: int = 12345) -> np.ndarray:
    if block_width % 2 == 0:
        raise ValueError("block_width must be odd for majority decoding")
    if not (0 <= overlap < block_width):
        raise ValueError("overlap must satisfy 0 <= overlap < block_width")

    rng = np.random.default_rng(index_seed)
    perm = rng.permutation(n).astype(np.int64)

    if overlap == 0:
        need = blocks * block_width
        if need > n:
            raise ValueError(f"Need {need} code positions, but n={n}. Increase n_bits or enable overlap.")
        return perm[:need].reshape(blocks, block_width)

    stride = block_width - overlap
    idx = np.empty((blocks, block_width), dtype=np.int64)
    for block_id in range(blocks):
        start = (block_id * stride) % n
        idx[block_id] = perm[(start + np.arange(block_width)) % n]
    return idx


def _as_bytes(value, *, name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"{name} must be bytes or str")


def derive_keyed_secret_bits(master_secret, salt, blocks: int) -> np.ndarray:
    master = _as_bytes(master_secret, name="master_secret")
    salt_b = _as_bytes(salt, name="salt")
    if len(master) < 16:
        raise ValueError("master_secret is too short for the keyed helper experiment")

    out = b""
    counter = 1
    while len(out) * 8 < blocks:
        msg = b"secret-bits|" + KEYED_FE_SCHEME.encode("ascii") + b"|" + salt_b + b"|" + counter.to_bytes(4, "big")
        out += hmac.new(master, msg, hashlib.sha256).digest()
        counter += 1
    return bytes_to_bits(out, blocks)


def derive_keyed_seed(master_secret, salt, purpose: str) -> int:
    master = _as_bytes(master_secret, name="master_secret")
    salt_b = _as_bytes(salt, name="salt")
    purpose_b = _as_bytes(purpose, name="purpose")
    if len(master) < 16:
        raise ValueError("master_secret is too short for the keyed helper experiment")

    msg = b"seed|" + KEYED_FE_SCHEME.encode("ascii") + b"|" + purpose_b + b"|" + salt_b
    digest = hmac.new(master, msg, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def keyed_fe_secret_to_key(secret_bits: np.ndarray, master_secret, salt):
    master = _as_bytes(master_secret, name="master_secret")
    salt_b = _as_bytes(salt, name="salt")
    secret_bytes = bits_to_bytes(secret_bits)
    r = hmac.new(master, b"R|" + KEYED_FE_SCHEME.encode("ascii") + b"|" + salt_b + b"|" + secret_bytes, hashlib.sha256).digest()
    k = hkdf_extract_expand(r, info=KEYED_FE_CTX, out_len=32)
    return r.hex(), k.hex()


def keyed_tag_from_secret_bits(secret_bits: np.ndarray, master_secret, salt, meta: str) -> str:
    _r_hex, k_hex = keyed_fe_secret_to_key(secret_bits, master_secret, salt)
    return hmac_sha256_hex(bytes.fromhex(k_hex), meta.encode("utf-8"))


def keyed_fe_gen(
    c_bits: np.ndarray,
    *,
    master_secret,
    salt,
    meta: str,
    blocks: int = 64,
    block_width: int = 31,
    overlap: int = 0,
    index_seed: int = 12345,
    max_errors_per_block: int = 8,
    max_total_errors: int = 96,
    secret_geometry: bool = False,
):
    c = np.asarray(c_bits, dtype=np.uint8) & 1
    effective_index_seed = derive_keyed_seed(master_secret, salt, "index-seed") if secret_geometry else index_seed
    idx = fe_block_indices(len(c), blocks, block_width, overlap=overlap, index_seed=effective_index_seed)

    if max_errors_per_block >= (block_width + 1) // 2:
        raise ValueError("max_errors_per_block must be stricter than majority radius")

    secret_bits = derive_keyed_secret_bits(master_secret, salt, blocks)
    code_matrix = np.repeat(secret_bits[:, None], block_width, axis=1).astype(np.uint8)
    c_matrix = c[idx]
    p_bits = (c_matrix ^ code_matrix).reshape(-1).astype(np.uint8)

    helper = {
        "scheme": KEYED_FE_SCHEME,
        "n": int(len(c)),
        "blocks": int(blocks),
        "block_width": int(block_width),
        "overlap": int(overlap),
        "secret_geometry": bool(secret_geometry),
        "max_errors_per_block": int(max_errors_per_block),
        "max_total_errors": int(max_total_errors),
        "salt": base64.b64encode(_as_bytes(salt, name="salt")).decode("ascii"),
        "helper_P": pack_bits_b64(p_bits),
    }
    if not secret_geometry:
        helper["index_seed"] = int(index_seed)

    r_hex, k_hex = keyed_fe_secret_to_key(secret_bits, master_secret, salt)
    tag = keyed_tag_from_secret_bits(secret_bits, master_secret, salt, meta)
    debug_private = {
        "secret_bits_b64": pack_bits_b64(secret_bits),
        "R_hex": r_hex,
        "K_hex": k_hex,
        "raw_secret_bits": int(blocks),
        "helper_bits": int(p_bits.size),
    }
    return {"helper": helper, "tag": tag, "debug_private": debug_private}


def keyed_fe_rep(c_bits: np.ndarray, helper: dict, *, master_secret):
    if helper.get("scheme") != KEYED_FE_SCHEME:
        raise ValueError(f"Unsupported keyed FE scheme: {helper.get('scheme')}")

    c = np.asarray(c_bits, dtype=np.uint8) & 1
    n = int(helper["n"])
    blocks = int(helper["blocks"])
    block_width = int(helper["block_width"])
    overlap = int(helper["overlap"])
    max_errors_per_block = int(helper["max_errors_per_block"])
    max_total_errors = int(helper["max_total_errors"])
    salt = base64.b64decode(helper["salt"])
    secret_geometry = bool(helper.get("secret_geometry", False))
    if secret_geometry:
        index_seed = derive_keyed_seed(master_secret, salt, "index-seed")
    else:
        index_seed = int(helper["index_seed"])
    if len(c) != n:
        raise ValueError(f"Code length mismatch: helper n={n}, query n={len(c)}")

    idx = fe_block_indices(n, blocks, block_width, overlap=overlap, index_seed=index_seed)
    p_bits = unpack_bits_b64(helper["helper_P"], blocks * block_width).reshape(blocks, block_width)

    observed = (c[idx] ^ p_bits).astype(np.uint8)
    ones = observed.sum(axis=1)
    secret_hat = (ones >= ((block_width + 1) // 2)).astype(np.uint8)
    expected_secret = derive_keyed_secret_bits(master_secret, salt, blocks)

    corrected_per_block = np.minimum(ones, block_width - ones).astype(np.int64)
    corrected_total = int(corrected_per_block.sum())
    max_corrected = int(corrected_per_block.max())
    overfull_blocks = int(np.sum(corrected_per_block > max_errors_per_block))
    decode_ok = (overfull_blocks == 0) and (corrected_total <= max_total_errors)
    secret_ok = bool(np.array_equal(secret_hat, expected_secret))

    r_hex, k_hex = keyed_fe_secret_to_key(secret_hat, master_secret, salt)
    return {
        "R_hex": r_hex,
        "K_hex": k_hex,
        "decode_ok": bool(decode_ok),
        "secret_ok": secret_ok,
        "corrected_observations": corrected_total,
        "max_corrected_in_block": max_corrected,
        "overfull_blocks": overfull_blocks,
        "max_errors_per_block": max_errors_per_block,
        "max_total_errors": max_total_errors,
    }


def keyed_verify_with_helper(c_bits: np.ndarray, helper: dict, stored_tag: str, meta: str, *, master_secret):
    rep = keyed_fe_rep(c_bits, helper, master_secret=master_secret)
    if not rep["decode_ok"]:
        return {**rep, "tag_q": None, "tag_ok": False, "verified": False}

    salt = base64.b64decode(helper["salt"])
    tag_q = hmac_sha256_hex(bytes.fromhex(rep["K_hex"]), meta.encode("utf-8"))
    expected_tag = keyed_tag_from_secret_bits(
        derive_keyed_secret_bits(master_secret, salt, int(helper["blocks"])),
        master_secret,
        salt,
        meta,
    )
    tag_ok = hmac.compare_digest(tag_q, stored_tag) and hmac.compare_digest(stored_tag, expected_tag)
    return {**rep, "tag_q": tag_q, "tag_ok": tag_ok, "verified": bool(tag_ok)}
