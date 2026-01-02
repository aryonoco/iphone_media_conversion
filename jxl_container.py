#!/usr/bin/env python3
"""
JXL Container Parser/Writer Module.

Handles ISOBMFF-like JXL container format for reading, modifying, and writing
JXL files with gain map boxes (jhgm).

SPDX-License-Identifier: MPL-2.0
Copyright (c) 2025-2026 Aryan Ameri
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__: Final[list[str]] = [
    "JxlBox",
    "parse_jxl_container",
    "write_jxl_container",
    "insert_jhgm_box",
    "extract_naked_codestream",
    "JXL_SIGNATURE",
]

# JXL container signature (12 bytes)
JXL_SIGNATURE: Final[bytes] = bytes([
    0x00, 0x00, 0x00, 0x0C,  # size = 12
    0x4A, 0x58, 0x4C, 0x20,  # "JXL "
    0x0D, 0x0A, 0x87, 0x0A,  # magic bytes
])


@dataclass(frozen=True, slots=True)
class JxlBox:
    """Represents a single box in a JXL container.

    Attributes:
        box_type: 4-character ASCII box type (e.g., "JXL ", "ftyp", "jxlc", "jhgm")
        data: Raw box data (excludes size and type header)
    """
    box_type: str
    data: bytes

    def __repr__(self) -> str:
        return f"JxlBox(type={self.box_type!r}, size={len(self.data)})"


def parse_jxl_container(path: Path) -> list[JxlBox]:
    """Parse a JXL file into a list of boxes.

    Box structure:
    - size (4 bytes, big-endian): Total box size including header
    - type (4 bytes, ASCII): Box type identifier
    - data (size - 8 bytes): Box payload

    Special size values:
    - size = 0: Box extends to end of file
    - size = 1: Extended size (64-bit) follows type field

    Args:
        path: Path to the JXL file

    Returns:
        List of JxlBox instances

    Raises:
        ValueError: If file is not a valid JXL container
    """
    with open(path, "rb") as f:
        data = f.read()

    if not data.startswith(JXL_SIGNATURE):
        raise ValueError(f"Not a JXL container file: {path}")

    boxes: list[JxlBox] = []
    pos = 0

    while pos < len(data):
        if pos + 8 > len(data):
            break

        # Read size and type
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        box_type = data[pos + 4 : pos + 8].decode("ascii", errors="replace")

        header_size = 8
        box_data_start = pos + header_size

        if size == 0:
            # Box extends to EOF
            box_data = data[box_data_start:]
            pos = len(data)
        elif size == 1:
            # Extended size (64-bit)
            if pos + 16 > len(data):
                break
            extended_size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header_size = 16
            box_data_start = pos + header_size
            box_data = data[box_data_start : pos + extended_size]
            pos += extended_size
        else:
            # Normal size
            box_data = data[box_data_start : pos + size]
            pos += size

        boxes.append(JxlBox(box_type=box_type, data=box_data))

    return boxes


def write_jxl_container(path: Path, boxes: list[JxlBox]) -> None:
    """Write a list of boxes to a JXL container file.

    Args:
        path: Output file path
        boxes: List of JxlBox instances to write
    """
    with open(path, "wb") as f:
        for box in boxes:
            box_type_bytes = box.box_type.encode("ascii")
            if len(box_type_bytes) != 4:
                raise ValueError(f"Box type must be exactly 4 characters: {box.box_type!r}")

            total_size = 8 + len(box.data)

            if total_size > 0xFFFFFFFF:
                # Need extended size
                f.write(struct.pack(">I", 1))  # size = 1 indicates extended
                f.write(box_type_bytes)
                f.write(struct.pack(">Q", 16 + len(box.data)))  # extended size
            else:
                f.write(struct.pack(">I", total_size))
                f.write(box_type_bytes)

            f.write(box.data)


def insert_jhgm_box(boxes: list[JxlBox], jhgm_box: JxlBox) -> list[JxlBox]:
    """Insert a jhgm box after the last codestream box.

    Per libjxl ordering convention:
    1. JXL  - Signature
    2. ftyp - File type
    3. jxlc/jxlp - Codestream (may be multiple jxlp for partial codestream)
    4. jhgm - Gain map (INSERT HERE)
    5. Exif, xml  - Metadata

    Args:
        boxes: Original list of boxes
        jhgm_box: The jhgm box to insert

    Returns:
        New list with jhgm box inserted

    Raises:
        ValueError: If no codestream box is found
    """
    # Find the last codestream box (jxlc or jxlp)
    last_codestream_idx = -1
    for i, box in enumerate(boxes):
        if box.box_type in ("jxlc", "jxlp"):
            last_codestream_idx = i

    if last_codestream_idx == -1:
        raise ValueError("No codestream box (jxlc/jxlp) found in JXL container")

    # Remove any existing jhgm box
    filtered_boxes = [box for box in boxes if box.box_type != "jhgm"]

    # Recalculate insertion point after filtering
    last_codestream_idx = -1
    for i, box in enumerate(filtered_boxes):
        if box.box_type in ("jxlc", "jxlp"):
            last_codestream_idx = i

    # Insert jhgm after the last codestream box
    result = filtered_boxes[: last_codestream_idx + 1]
    result.append(jhgm_box)
    result.extend(filtered_boxes[last_codestream_idx + 1 :])

    return result


def extract_naked_codestream(jxl_path: Path) -> bytes:
    """Extract the naked codestream from a JXL container.

    If the file is already a naked codestream (starts with JXL codestream
    signature 0xFF0A), returns it directly.

    Args:
        jxl_path: Path to JXL file

    Returns:
        Naked codestream bytes
    """
    with open(jxl_path, "rb") as f:
        data = f.read()

    # Check for naked codestream signature (0xFF 0x0A)
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0x0A:
        return data

    # Parse container and extract codestream
    if not data.startswith(JXL_SIGNATURE):
        raise ValueError(f"Not a JXL file: {jxl_path}")

    boxes = parse_jxl_container(jxl_path)

    # Collect all codestream data
    codestream_parts: list[bytes] = []
    for box in boxes:
        if box.box_type == "jxlc":
            # Single codestream box
            return box.data
        elif box.box_type == "jxlp":
            # Partial codestream - first 4 bytes are sequence number
            if len(box.data) >= 4:
                codestream_parts.append(box.data[4:])

    if codestream_parts:
        return b"".join(codestream_parts)

    raise ValueError(f"No codestream found in JXL container: {jxl_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: jxl_container.py <file.jxl>")
        sys.exit(1)

    jxl_path = Path(sys.argv[1])
    boxes = parse_jxl_container(jxl_path)

    print(f"JXL Container: {jxl_path}")
    print(f"Total boxes: {len(boxes)}")
    print()

    for i, box in enumerate(boxes):
        print(f"  [{i}] {box.box_type!r}: {len(box.data)} bytes")
