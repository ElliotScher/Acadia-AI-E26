# -*- mode: python ; coding: utf-8 -*-

import sys

a = Analysis(
    ['src/ui/main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Shown immediately while the heavier imports (torch,
# ultralytics, PySide6) load, since those otherwise leave the user staring
# at nothing for several seconds. main.py closes this once the main window
# is on screen.
splash = Splash(
    'assets/splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(20, 295),
    text_size=11,
    text_color='#ebeef2',
    text_default='Starting...',
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX-compressed executables are one of the most common triggers for
    # antivirus heuristics (packed/high-entropy binaries look like malware
    # droppers), so leave binaries uncompressed.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# onedir (COLLECT) rather than onefile: a onefile build self-extracts to a
# temp directory on every launch, which is the other big antivirus
# heuristics trigger for PyInstaller apps, and it also means every startup
# pays an unpack cost on top of the import cost the splash screen is
# already covering.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    splash.binaries,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='main',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='main.app',
        icon=None,
        bundle_identifier=None,
    )
