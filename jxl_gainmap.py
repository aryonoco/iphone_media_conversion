"""
JXL Gain Map Helper Module

This module provides functions for creating ISO 21496-1 gain maps for JXL files,
enabling HDR display on supported devices while maintaining SDR fallback.

Key functions:
- compute_gain_map(): Compute gain map from linear SDR and HDR images
- tone_map_hdr_to_sdr_linear(): Apply Reinhard-style tone mapping
- apply_srgb_gamma(): Apply sRGB transfer function for display encoding
- serialize_iso21496_metadata(): Create ISO 21496-1 binary metadata
- serialize_jhgm_bundle(): Create jhgm box payload
- insert_jhgm_box(): Container surgery to add jhgm box to JXL
"""

import struct
import numpy as np
from typing import Tuple

# JXL container signature (12 bytes, NOT a standard ISOBMFF box)
JXL_SIGNATURE = bytes([0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20,
                       0x0D, 0x0A, 0x87, 0x0A])


def tone_map_hdr_to_sdr_linear(hdr: np.ndarray) -> np.ndarray:
    """Apply simple Reinhard-style tone mapping for SDR (LINEAR output).

    For iPhone ProRAW photos, this compresses highlights while preserving midtones.

    Input: Linear Rec.2020 HDR from dcraw_emu (normalized 0-1 after dividing by 65535)
    Output: LINEAR Rec.2020 SDR (0-1 range, NO gamma)

    Formula: x / (1 + x/peak) maps: 0->0, peak->peak/2, infinity->peak
    """
    # Find the actual peak value in the image (scene-referred)
    # Use 99.9th percentile to avoid outliers
    peak = np.percentile(hdr, 99.9)
    if peak < 0.01:
        peak = 1.0  # Avoid division issues for very dark images

    # Reinhard tone mapping with peak normalization
    sdr_linear = hdr / (1 + hdr / peak)

    return np.clip(sdr_linear, 0, 1)


def apply_srgb_gamma(linear: np.ndarray) -> np.ndarray:
    """Apply sRGB transfer function (gamma) for display encoding.

    Call this AFTER computing gain map, for SDR image encoding only.
    """
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0), 1/2.4) - 0.055
    )


def compute_gain_map(sdr_linear: np.ndarray, hdr_linear: np.ndarray,
                     target_headroom: float = 3.32) -> Tuple[np.ndarray, dict]:
    """Compute gain map from LINEAR SDR and HDR images.

    CRITICAL: Both inputs must be in LINEAR space (no gamma encoding).
    The gain map encodes the ratio needed to reconstruct HDR from SDR.

    Args:
        sdr_linear: LINEAR SDR image (tone-mapped but NOT gamma-encoded, 0-1 range)
        hdr_linear: LINEAR HDR image (same color space as SDR)
        target_headroom: log2(max_nits/203). Default 3.32 ~ log2(1000/203) for 1000 nits

    Returns:
        gain_map: 8-bit encoded gain map (grayscale)
        metadata: Dict with ISO 21496-1 parameters (keys match serialize_iso21496_metadata params)
    """
    OFFSET = 1/64  # 0.015625 - avoids division by zero

    # Convert to luminance using Rec.2020 primaries - both MUST be linear
    sdr_lum = 0.2627 * sdr_linear[:,:,0] + 0.6780 * sdr_linear[:,:,1] + 0.0593 * sdr_linear[:,:,2]
    hdr_lum = 0.2627 * hdr_linear[:,:,0] + 0.6780 * hdr_linear[:,:,1] + 0.0593 * hdr_linear[:,:,2]

    # Compute per-pixel gain ratio (both in linear space)
    pixel_gain = (hdr_lum + OFFSET) / (sdr_lum + OFFSET)

    # Content-based min/max (log2 space)
    gain_min_log2 = np.log2(np.maximum(np.min(pixel_gain), 1e-10))
    gain_max_log2 = np.log2(np.maximum(np.max(pixel_gain), 1e-10))

    # Normalize to 0-1 range
    range_log2 = gain_max_log2 - gain_min_log2
    if range_log2 < 1e-6:
        range_log2 = 1.0  # Avoid division by zero for flat images

    log_gain = np.log2(np.maximum(pixel_gain, 1e-10))
    normalized = (log_gain - gain_min_log2) / range_log2

    # Gamma encoding for gain map quantization:
    # - gamma=1.0: Linear encoding (simple, may lose shadow detail)
    # - gamma>1.0: Better quantization in shadows/midtones
    # Using 1.0 for simplicity; increase if shadow banding is visible
    GAIN_MAP_GAMMA = 1.0

    # Apply gamma encoding before quantization (if gamma != 1.0)
    if GAIN_MAP_GAMMA != 1.0:
        normalized = np.power(np.clip(normalized, 0, 1), 1.0 / GAIN_MAP_GAMMA)

    # Encode to 8-bit
    encoded = np.clip(np.round(normalized * 255), 0, 255).astype(np.uint8)

    metadata = {
        "gain_map_min": gain_min_log2,
        "gain_map_max": gain_max_log2,
        "gamma": GAIN_MAP_GAMMA,
        "offset": OFFSET,
        "hdr_headroom": target_headroom,
    }

    return encoded, metadata


def float_to_fraction(value: float, denominator: int = 10000) -> Tuple[int, int]:
    """Convert float to fraction with fixed denominator.

    Returns (numerator, denominator) where numerator can be negative.
    Validates that result fits in i32/u32 range for ISO 21496-1 format.
    """
    numerator = int(round(value * denominator))
    # Validate ranges (i32 for signed numerator, u32 for denominator)
    if not (-2**31 <= numerator < 2**31):
        raise ValueError(f"Numerator {numerator} out of i32 range")
    if not (0 < denominator < 2**32):
        raise ValueError(f"Denominator {denominator} out of u32 range")
    return (numerator, denominator)


def serialize_iso21496_metadata(
    gain_map_min: float, gain_map_max: float,
    gamma: float = 1.0,
    offset: float = 1/64,
    hdr_headroom: float = 3.32
) -> bytes:
    """Serialize ISO 21496-1 binary metadata (single-channel, grayscale gain map).

    Binary format (all multi-byte values big-endian):
    - minimum_version: 2 bytes (u16)
    - writer_version: 2 bytes (u16)
    - flags: 1 byte (is_multichannel=bit7, use_base_colour_space=bit6)
    - base_hdr_headroom: 8 bytes (u32/u32 fraction)
    - alternate_hdr_headroom: 8 bytes (u32/u32 fraction)
    - Per-channel data (1x for grayscale, 40 bytes = 10 x u32):
      - gain_map_min: i32/u32 fraction
      - gain_map_max: i32/u32 fraction
      - gain_map_gamma: u32/u32 fraction
      - base_offset: i32/u32 fraction
      - alternate_offset: i32/u32 fraction

    Total: 21 + 40 = 61 bytes for single-channel.

    Parameter names match the keys returned by compute_gain_map() for easy unpacking:
        serialize_iso21496_metadata(**metadata)
    """
    data = bytearray()

    # Version info
    data.extend(struct.pack('>H', 0))  # minimum_version
    data.extend(struct.pack('>H', 0))  # writer_version

    # Flags: is_multichannel=0 (bit 7), use_base_colour_space=0 (bit 6)
    data.append(0x00)

    # Headroom values (fractions as u32/u32)
    data.extend(struct.pack('>II', 0, 1))  # base_hdr_headroom = 0/1 (SDR)
    num, denom = float_to_fraction(hdr_headroom)
    data.extend(struct.pack('>II', num, denom))  # alternate_hdr_headroom

    # Single channel data (40 bytes = 10 x u32)
    # gain_map_min (i32/u32)
    num, denom = float_to_fraction(gain_map_min)
    data.extend(struct.pack('>iI', num, denom))
    # gain_map_max (i32/u32)
    num, denom = float_to_fraction(gain_map_max)
    data.extend(struct.pack('>iI', num, denom))
    # gain_map_gamma (u32/u32)
    num, denom = float_to_fraction(gamma)
    data.extend(struct.pack('>II', num, denom))
    # base_offset (i32/u32)
    num, denom = float_to_fraction(offset)
    data.extend(struct.pack('>iI', num, denom))
    # alternate_offset (i32/u32)
    num, denom = float_to_fraction(offset)
    data.extend(struct.pack('>iI', num, denom))

    assert len(data) == 61, f"Expected 61 bytes, got {len(data)}"
    return bytes(data)


def serialize_jhgm_bundle(metadata: bytes, gain_map_codestream: bytes,
                          color_encoding: bytes = b'') -> bytes:
    """Serialize jhgm bundle following JxlGainMapBundle format.

    Format (from lib/extras/gain_map.cc):
    - jhgm_version: 1 byte (0x00)
    - gain_map_metadata_size: 2 bytes (big-endian)
    - gain_map_metadata: N bytes (ISO 21496-1 binary)
    - color_encoding_size: 1 byte (0 if none)
    - color_encoding: N bytes (if present)
    - alt_icc_size: 4 bytes (big-endian, 0 if none)
    - alt_icc: N bytes (if present)
    - gain_map: remaining bytes (naked JXL codestream)
    """
    payload = bytearray()
    payload.append(0x00)                                    # jhgm_version
    payload.extend(len(metadata).to_bytes(2, 'big'))        # metadata_size (BE16)
    payload.extend(metadata)                                # ISO 21496-1 binary
    payload.append(len(color_encoding))                     # color_encoding_size
    payload.extend(color_encoding)                          # color encoding (optional)
    payload.extend((0).to_bytes(4, 'big'))                  # alt_icc_size = 0 (BE32)
    payload.extend(gain_map_codestream)                     # naked JXL codestream
    return bytes(payload)


def parse_isobmff_boxes(data: bytes) -> list:
    """Parse ISOBMFF boxes from JXL container (after signature).

    Returns list of (box_type, box_data) tuples where box_data includes
    the size and type header.
    """
    if data[:12] != JXL_SIGNATURE:
        raise ValueError("Not a JXL container (invalid signature)")

    boxes = []
    pos = 12  # Skip signature
    while pos < len(data):
        if pos + 8 > len(data):
            break
        size = int.from_bytes(data[pos:pos+4], 'big')
        box_type = data[pos+4:pos+8]
        if size == 0:  # Box extends to EOF
            boxes.append((box_type, data[pos:]))
            break
        elif size == 1:  # Extended size (64-bit)
            if pos + 16 > len(data):
                break
            size = int.from_bytes(data[pos+8:pos+16], 'big')
        boxes.append((box_type, data[pos:pos+size]))
        pos += size
    return boxes


def insert_jhgm_box(base_jxl: bytes, jhgm_bundle: bytes) -> bytes:
    """Insert jhgm box ONCE after last codestream box.

    JXL containers use ISOBMFF format. This function:
    1. Parses the container boxes
    2. Finds the last jxlc/jxlp codestream box
    3. Inserts the jhgm box after it
    4. Reassembles the container
    """
    if base_jxl[:12] != JXL_SIGNATURE:
        raise ValueError("Not a JXL container (invalid signature)")

    boxes = parse_isobmff_boxes(base_jxl)
    jhgm_box = (len(jhgm_bundle) + 8).to_bytes(4, 'big') + b'jhgm' + jhgm_bundle

    # Find LAST jxlc/jxlp box index (may have multiple jxlp for progressive)
    last_codestream_idx = -1
    for i, (box_type, _) in enumerate(boxes):
        if box_type in (b'jxlc', b'jxlp'):
            last_codestream_idx = i

    if last_codestream_idx == -1:
        raise ValueError("No codestream box found in container")

    # Reassemble: signature + boxes with jhgm after last codestream
    output = bytearray(JXL_SIGNATURE)
    for i, (box_type, box_data) in enumerate(boxes):
        output.extend(box_data)
        if i == last_codestream_idx:
            output.extend(jhgm_box)

    return bytes(output)


def verify_jxl_container(path: str) -> dict:
    """Verify JXL container has expected structure.

    Returns dict with:
    - has_codestream: bool
    - has_jhgm: bool
    - jhgm_count: int (should be exactly 1)
    - box_order: list of box type strings
    """
    with open(path, 'rb') as f:
        data = f.read()

    if data[:12] != JXL_SIGNATURE:
        return {"error": "Invalid JXL signature", "is_naked": data[:2] == b'\xff\x0a'}

    boxes = parse_isobmff_boxes(data)
    box_types = [t.decode('ascii', errors='replace') for t, _ in boxes]

    return {
        "has_codestream": any(t in ('jxlc', 'jxlp') for t in box_types),
        "has_jhgm": 'jhgm' in box_types,
        "jhgm_count": box_types.count('jhgm'),
        "box_order": box_types
    }
