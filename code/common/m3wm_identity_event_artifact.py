from __future__ import annotations
import hashlib
import struct
from typing import Any, Dict
import torch

IDENTITY_EVENT_SCHEMA = "m3wm_context_identity_event"
IDENTITY_EVENT_VERSION = 2


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return
    if isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(b"I" + str(len(encoded)).encode("ascii") + b":" + encoded)
        return
    if isinstance(value, float):
        digest.update(b"F" + struct.pack(">d", value))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + str(len(encoded)).encode("ascii") + b":" + encoded)
        return
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        _update_digest(digest, str(tensor.dtype))
        _update_digest(digest, tuple((int(item) for item in tensor.shape)))
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(str(len(raw)).encode("ascii") + b":" + raw)
        return
    if isinstance(value, tuple):
        digest.update(b"U")
        _update_digest(digest, len(value))
        for child in value:
            _update_digest(digest, child)
        return
    if isinstance(value, list):
        digest.update(b"L")
        _update_digest(digest, len(value))
        for child in value:
            _update_digest(digest, child)
        return
    if isinstance(value, dict):
        digest.update(b"D")
        encoded = []
        for key, child in value.items():
            key_digest = hashlib.sha256()
            _update_digest(key_digest, key)
            encoded.append((key_digest.digest(), key, child))
        encoded.sort(key=lambda item: item[0])
        _update_digest(digest, len(encoded))
        for _, key, child in encoded:
            _update_digest(digest, key)
            _update_digest(digest, child)
        return
    raise TypeError(f"Unsupported identity-event value: {type(value).__name__}")


def identity_event_payload_sha256(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise TypeError("Identity-event payload must be a dictionary")
    digest = hashlib.sha256()
    _update_digest(digest, payload)
    return digest.hexdigest()
