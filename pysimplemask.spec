# pysimplemask.spec
# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the pySimpleMask GUI (.exe / one-dir bundle)
#
# Usage:
#   pyinstaller pysimplemask.spec
#
# The spec uses a small entrypoint in packaging/ so the packaged binary
# stays stable even if the internal import path changes.

from pathlib import Path

# SPECPATH is the directory containing this spec file (set by PyInstaller),
# not the file path itself. This spec lives at the repo root alongside
# packaging/ and src/, so both anchors are that same directory.
spec_dir = Path(SPECPATH)
repo_root = spec_dir

a = Analysis(
    [str(spec_dir / 'packaging' / 'entrypoint.py')],
    pathex=[],
    binaries=[],
    datas=[
        # .ui file (Qt Designer source, kept for reference)
        (str(repo_root / 'src' / 'pysimplemask' / 'gui' / 'view' / 'mask.ui'),
         'pysimplemask/gui/view'),
        # SVG logo
        (str(repo_root / 'src' / 'pysimplemask' / 'resources'),
         'pysimplemask/resources'),
    ],
    hiddenimports=[
        # top-level package
        'pysimplemask',
        'pysimplemask.core',
        'pysimplemask.core.mask',
        'pysimplemask.core.model',
        'pysimplemask.core.qmap',
        'pysimplemask.core.partition',
        'pysimplemask.core.io',
        'pysimplemask.core.rasterize',
        'pysimplemask.core.find_center',
        'pysimplemask.core.outlier_removal',
        'pysimplemask.core.ellipse_util',
        'pysimplemask.core.file_handler',
        'pysimplemask.core.report',
        # readers
        'pysimplemask.core.reader',
        'pysimplemask.core.reader.base_reader',
        'pysimplemask.core.reader.io_utils',
        'pysimplemask.core.reader.metadata',
        'pysimplemask.core.reader.beamlines',
        'pysimplemask.core.reader.beamlines.aps_8idi',
        'pysimplemask.core.reader.beamlines.aps_9idd',
        'pysimplemask.core.reader.beamlines.native_files',
        'pysimplemask.core.reader.beamlines.xpcs_result',
        'pysimplemask.core.reader.formats',
        'pysimplemask.core.reader.formats.base',
        'pysimplemask.core.reader.formats.hdf',
        'pysimplemask.core.reader.formats.imm',
        'pysimplemask.core.reader.formats.rigaku',
        # gui
        'pysimplemask.gui',
        'pysimplemask.gui.app',
        'pysimplemask.gui.control',
        'pysimplemask.gui.control.main_window',
        'pysimplemask.gui.model',
        'pysimplemask.gui.model.roi_extract',
        'pysimplemask.gui.model.table_model',
        'pysimplemask.gui.view',
        'pysimplemask.gui.view.ui_mask',
        'pysimplemask.gui.view.widgets',
        'pysimplemask.gui.view.compile_ui',
        # cli
        'pysimplemask.cli',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest',
        'coverage',
        'setuptools',
        'pip',
        'wheel',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Platform-specific EXE settings
import sys

if sys.platform == 'win32':
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='pySimpleMask',
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
        icon=str(spec_dir / 'packaging' / 'icon.ico'),
    )
elif sys.platform == 'linux':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='pySimpleMask',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='pySimpleMask',
    )
else:
    # macOS or other — single-dir EXE
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='pySimpleMask',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )