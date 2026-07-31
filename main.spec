# -*- mode: python ; coding: utf-8 -*-

import sys

a = Analysis(
    ['src/ui/main.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

splash = Splash(
    'assets/splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(27, 956),
    text_size=19,
    text_color='#ebeef2',
    text_default='Starting...',
    minify_script=True,
    always_on_top=True,
    # max_img_size=None is documented to disable the resize cap, but
    # PyInstaller 6.21.0 does `img_size > self.max_img_size` unconditionally
    # and crashes on None, so pass a bound at least as large as splash.png.
    max_img_size=(1330, 1001),
) if sys.platform != 'darwin' else None

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash if splash is not None else [],
    splash.binaries if splash is not None else [],
    [],
    name='ParkVision',
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
    icon='assets/icon.png',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='ParkVision.app',
        icon='assets/icon.png',
        bundle_identifier=None,
    )
