#!/usr/bin/env python3
"""
Prepare USB deployment package — run THIS on your own machine before going to the cafe.

What it does:
  1. Downloads Python embeddable package (same version as current Python)
  2. Pre-downloads all pip wheels for offline install
  3. Downloads portable 7-Zip
  4. Creates a ready-to-run bundle in dist/

After running this, copy the entire portable/ folder to your USB drive.
On the cafe machine, just run: dist\\python\\python.exe run_download.py
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DIST_DIR = SCRIPT_DIR / "dist"
WHEELS_DIR = DIST_DIR / "wheels"
PYTHON_DIR = DIST_DIR / "python"


def get_python_embed_url() -> str:
    """Build download URL for Python embeddable package matching current version."""
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    arch = "amd64" if platform.machine().endswith("64") else "win32"
    return f"https://www.python.org/ftp/python/{ver}/python-{ver}-embed-{arch}.zip"


def download_file(url: str, dest: Path, desc: str = "") -> None:
    print(f"  Downloading {desc or url} ...")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  -> {dest.name} ({size_mb:.1f} MB)")


def setup_python_embed() -> None:
    """Download and configure Python embeddable for the USB."""
    if PYTHON_DIR.exists():
        print(f"[skip] {PYTHON_DIR} already exists")
        return

    PYTHON_DIR.mkdir(parents=True, exist_ok=True)
    url = get_python_embed_url()
    zip_path = DIST_DIR / "python_embed.zip"

    download_file(url, zip_path, "Python embeddable")

    print("  Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(PYTHON_DIR)
    zip_path.unlink()

    # Enable pip: uncomment "import site" in python*._pth
    pth_files = list(PYTHON_DIR.glob("python*._pth"))
    for pth in pth_files:
        text = pth.read_text()
        text = text.replace("#import site", "import site")
        pth.write_text(text)
        print(f"  Patched {pth.name} to enable site-packages")

    # Install pip into the embeddable Python
    get_pip_path = DIST_DIR / "get-pip.py"
    download_file("https://bootstrap.pypa.io/get-pip.py", get_pip_path, "get-pip.py")

    py_exe = PYTHON_DIR / "python.exe"
    subprocess.run([str(py_exe), str(get_pip_path), "--no-warn-script-location"],
                   check=True)
    get_pip_path.unlink()
    print("  pip installed into embeddable Python")


def download_wheels() -> None:
    """Pre-download all dependency wheels for offline install."""
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)

    py_exe = PYTHON_DIR / "python.exe"
    v = sys.version_info
    py_tag = f"cp{v.major}{v.minor}"
    arch = "win_amd64" if platform.machine().endswith("64") else "win32"

    for req_file, label in [
        (SCRIPT_DIR / "requirements.txt", "download-only"),
        (SCRIPT_DIR / "requirements_pipeline.txt", "pipeline"),
    ]:
        print(f"\n  Downloading wheels for {label} ({req_file.name}) ...")
        subprocess.run([
            str(py_exe), "-m", "pip", "download",
            "-r", str(req_file),
            "-d", str(WHEELS_DIR),
            "--only-binary=:all:",
            f"--python-version={v.major}.{v.minor}",
            f"--platform={arch}",
        ], check=True)


def install_download_deps() -> None:
    """Install the download-only deps directly into the embeddable Python."""
    py_exe = PYTHON_DIR / "python.exe"
    print("\n  Installing download-only deps into embeddable Python ...")
    subprocess.run([
        str(py_exe), "-m", "pip", "install",
        "-r", str(SCRIPT_DIR / "requirements.txt"),
        "--no-index",
        "--find-links", str(WHEELS_DIR),
        "--no-warn-script-location",
    ], check=True)


def setup_7z() -> None:
    """Download full portable 7-Zip (supports .rar/.zip/.7z)."""
    sevenz = SCRIPT_DIR / "7z.exe"
    sevenz_dll = SCRIPT_DIR / "7z.dll"
    if sevenz.exists() and sevenz_dll.exists():
        print(f"[skip] 7z.exe + 7z.dll already exist")
        return

    tmp_dir = DIST_DIR / "_7z_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: download 7zr.exe (minimal, can extract .7z archives)
    sevenzr = tmp_dir / "7zr.exe"
    download_file("https://www.7-zip.org/a/7zr.exe", sevenzr, "7zr.exe (bootstrap)")

    # Step 2: download 7-Zip Extra package (.7z) which contains full 7z.exe + 7z.dll
    extra_url = "https://github.com/ip7z/7zip/releases/download/26.00/7z2600-extra.7z"
    extra_archive = tmp_dir / "7z-extra.7z"
    download_file(extra_url, extra_archive, "7-Zip Extra (full, with .rar support)")

    # Step 3: extract with 7zr.exe
    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    subprocess.run(
        [str(sevenzr), "x", f"-o{extract_dir}", "-y", str(extra_archive)],
        check=True, capture_output=True,
    )

    # Step 4: find and copy 7z.exe + 7z.dll (they're in the x64/ subfolder)
    found = False
    for subdir in [extract_dir / "x64", extract_dir]:
        exe = subdir / "7z.exe" if (subdir / "7z.exe").exists() else subdir / "7za.exe"
        dll = subdir / "7z.dll"
        if exe.exists():
            shutil.copy2(exe, sevenz)
            if dll.exists():
                shutil.copy2(dll, sevenz_dll)
            found = True
            break

    # Cleanup temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if found:
        print(f"  [OK] 7z.exe + 7z.dll ready (supports .rar/.zip/.7z)")
    else:
        print(f"  [WARN] Could not find 7z.exe in extra package")
        print(f"  Manually place 7z.exe + 7z.dll in {SCRIPT_DIR}")


def create_launch_scripts() -> None:
    """Create .bat launchers that use the bundled Python."""
    for script, desc in [
        ("run_download.py", "download-only"),
        ("run_pipeline.py", "pipeline"),
    ]:
        bat = SCRIPT_DIR / f"launch_{script.replace('.py', '')}.bat"
        bat.write_text(
            f"@echo off\r\n"
            f"echo === CS Prophet Portable ({desc}) ===\r\n"
            f"set \"PYTHONNOUSERSITE=1\"\r\n"
            f'"%~dp0dist\\python\\python.exe" "%~dp0{script}" %*\r\n'
            f"if errorlevel 1 pause\r\n",
            encoding="utf-8",
        )
        print(f"  Created {bat.name}")


def main() -> None:
    print("=" * 60)
    print("CS Prophet — USB Deployment Preparation")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Arch: {platform.machine()}")
    print()

    print("[1/5] Setting up embeddable Python ...")
    setup_python_embed()

    print("\n[2/5] Downloading dependency wheels ...")
    download_wheels()

    print("\n[3/5] Installing download-only deps into embeddable Python ...")
    install_download_deps()

    print("\n[4/5] Setting up 7-Zip ...")
    setup_7z()

    print("\n[5/5] Creating launch scripts ...")
    create_launch_scripts()

    print()
    print("=" * 60)
    print("DONE. Copy the entire 'portable/' folder to your USB drive.")
    print()
    print("On the cafe machine:")
    print("  Double-click launch_run_download.bat   (download .dem files)")
    print("  Double-click launch_run_pipeline.bat   (download + parse)")
    print()
    print("If you need pipeline mode, first install extra deps on cafe:")
    print(f'  dist\\python\\python.exe -m pip install -r requirements_pipeline.txt '
          f'--no-index --find-links dist\\wheels')
    print("=" * 60)


if __name__ == "__main__":
    main()
