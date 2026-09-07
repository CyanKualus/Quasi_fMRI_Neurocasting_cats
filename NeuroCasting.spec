# Build on Windows: python -m PyInstaller --noconfirm NeuroCasting.spec
from pathlib import Path

root = Path(SPECPATH)
a = Analysis(
    [str(root / "neurocasting_launcher.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(root / "assets"), "assets")],
    hiddenimports=["pyxdf"],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["QtAgg", "Agg"]}},
    runtime_hooks=[],
    excludes=["pytest", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="NeuroCasting",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(root / "assets" / "klh_intro.png"),
)
