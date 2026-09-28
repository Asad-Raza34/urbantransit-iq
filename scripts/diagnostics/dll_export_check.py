"""Interrogate the Windows export table of candidate hadoop.dll builds.

A DLL can contain the JNI symbol string without actually exporting it, so we
ask the Windows loader directly (GetProcAddress) instead of grepping bytes.
"""
import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

SYMBOLS = [
    "Java_org_apache_hadoop_io_nativeio_NativeIO_00024Windows_access0",
    "Java_org_apache_hadoop_io_nativeio_NativeIO_00024Windows_getOwner0",
    "Java_org_apache_hadoop_io_nativeio_NativeIO_00024Windows_chmodImpl0",
]

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.LoadLibraryW.restype = ctypes.c_void_p
kernel32.LoadLibraryW.argtypes = [wintypes.LPCWSTR]
kernel32.GetProcAddress.restype = ctypes.c_void_p
kernel32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


def check(dll_path: Path) -> None:
    handle = kernel32.LoadLibraryW(str(dll_path))
    if not handle:
        err = ctypes.get_last_error()
        print(f"{dll_path}: LOAD FAILED (winerror {err})")
        return
    print(f"{dll_path.name}: loaded (handle {handle:#x})")
    for sym in SYMBOLS:
        addr = kernel32.GetProcAddress(handle, sym.encode())
        print(f"  {'EXPORTED' if addr else 'MISSING '}  {sym}")
    ctypes.WinDLL(str(dll_path))  # keep a reference; frees on process exit


if __name__ == "__main__":
    targets = sys.argv[1:] or ["hadoop/bin/hadoop.dll"]
    for t in targets:
        p = Path(t)
        if p.exists():
            check(p)
        else:
            print(f"{t}: not found")
