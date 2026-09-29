"""Redact local home-directory usernames from PyTorch ZIP checkpoint metadata."""

import argparse
import pickletools
import re
import shutil
import struct
import zipfile
from pathlib import Path


HOME_USER = re.compile(rb"(?<=/home/)[A-Za-z0-9._-]+")
PICKLE_MEMBER = "archive/data.pkl"


def sanitize_pickle(data):
    output = bytearray(data)
    changed = 0
    for opcode, value, offset in pickletools.genops(data):
        if not isinstance(value, str) or "/home/" not in value:
            continue
        if opcode.name != "BINUNICODE":
            raise ValueError(f"Unsupported pickle string opcode: {opcode.name}")
        length = struct.unpack_from("<I", data, offset + 1)[0]
        start = offset + 5
        original = data[start:start + length]
        if original.decode("utf-8") != value:
            raise ValueError("Pickle string boundary mismatch")
        replacement, count = HOME_USER.subn(lambda match: b"x" * len(match.group()), original)
        if len(replacement) != len(original):
            raise ValueError("Redaction changed pickle string length")
        output[start:start + length] = replacement
        changed += count
    if changed == 0:
        raise ValueError("No home-directory username found in checkpoint metadata")
    list(pickletools.genops(output))
    return bytes(output), changed


def sanitize_archive(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(temporary, "w", allowZip64=True) as clean:
            if PICKLE_MEMBER not in original.namelist():
                raise ValueError(f"Missing {PICKLE_MEMBER}")
            clean.comment = original.comment
            changed = 0
            for info in original.infolist():
                with original.open(info) as input_file, clean.open(info, "w", force_zip64=True) as output_file:
                    if info.filename == PICKLE_MEMBER:
                        data, changed = sanitize_pickle(input_file.read())
                        output_file.write(data)
                    else:
                        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
        with zipfile.ZipFile(temporary) as check:
            if check.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
        temporary.replace(destination)
        print(f"Sanitized {changed} home-directory references in {destination.name}")
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    sanitize_archive(arguments.source, arguments.destination)
