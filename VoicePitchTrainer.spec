# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None


def collect_package(package_name):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
        return package_datas, package_binaries, package_hiddenimports
    except Exception:
        return [], [], []


datas = []
binaries = []
hiddenimports = []

# Demucs needs package-data files to resolve names like "htdemucs".
for package_name in [
    "demucs",
    "dora",
    "julius",
    "diffq",
    "lameenc",
    "openunmix",
    "treetable",
]:
    package_datas, package_binaries, package_hiddenimports = collect_package(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

# Be explicit about Demucs data. This is the important part for:
# "htdemucs is neither a single pre-trained model or a bag of models."
datas += collect_data_files("demucs", include_py_files=False)
hiddenimports += collect_submodules("demucs")

# PyQt6 SVG/icon plugins are used by the UI.
package_datas, package_binaries, package_hiddenimports = collect_package("PyQt6")
datas += package_datas
binaries += package_binaries
hiddenimports += package_hiddenimports

# Torch hooks are heavy but reliable for onefile builds with Demucs.
package_datas, package_binaries, package_hiddenimports = collect_package("torch")
datas += package_datas
binaries += package_binaries
hiddenimports += package_hiddenimports

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="VoicePitchTrainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
