# -*- mode: python ; coding: utf-8 -*-
import glob

# numpy 1.23.5 (MinGW/gcc build) 的 BLAS 依賴鏈需要這些 mingw-w64 執行期 DLL，
# 但 PyInstaller 的分析只會追蹤 Python 匯入層級的相依，追不到 DLL-對-DLL 的間接依賴，
# 所以要手動把整個資料夾的 DLL 加進去，才不會在啟動時出現
# "ImportError: DLL load failed while importing _multiarray_umath" 的錯誤。
_mingw_dlls = [(p, '.') for p in glob.glob(
    r'C:\Users\Tno\miniconda3\envs\py39\Library\mingw-w64\bin\*.dll')]

# cv2 (conda 版 opencv-python) 的 49 個 opencv_*4120.dll 實際放在 conda 環境的
# Library\bin\ 底下，不在 site-packages 裡，PyInstaller 的自動相依分析找不到，
# 造成 "ImportError: DLL load failed while importing cv2" (DLL initialization
# routine failed)。手動加進來才能正確打包。
_opencv_dlls = [(p, '.') for p in glob.glob(
    r'C:\Users\Tno\miniconda3\envs\py39\Library\bin\opencv_*.dll')]

a = Analysis(
    ['LipNetConnect1.py'],
    pathex=[],
    binaries=_mingw_dlls + _opencv_dlls,
    datas=[],
    hiddenimports=['ultralytics'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_dll_dirs.py'],
    excludes=['PyQt6', 'PySide2', 'PySide6'],
    noarchive=False,
    optimize=0,
)
# 2026-09-18修正：實機測試發現打包後的exe一啟動就當機(Windows事件記錄檔顯示
# c0000005記憶體存取違規)，Faulting模組是_internal\PyQt5\Qt5\bin\MSVCP140.dll——
# PyQt5自己也內建了一份舊版微軟C++執行期DLL(14.26.28720.3)，跟cv2需要的新版本
# 衝突，Windows誤抓到PyQt5那份舊的載入，cv2呼叫進去就直接崩潰。_internal根目錄
# 已經有一份正確版本的msvcp140.dll(及同系列的vcruntime140*.dll)，把PyQt5
# Qt5\bin底下這幾個重複、版本不對的執行期DLL從打包清單裡濾掉，讓系統改用根目錄
# 那份正確的，其餘Qt5\bin底下的Qt本身DLL(QtCore/QtGui等)都保留不動。
_dup_runtime_dll_names = {
    'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
    'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
    'vcruntime140.dll', 'vcruntime140_1.dll', 'vcruntime140_threads.dll',
    'concrt140.dll',
}
a.binaries = TOC([
    entry for entry in a.binaries
    if not (
        entry[0].lower().startswith('pyqt5\\qt5\\bin\\')
        and entry[0].lower().rsplit('\\', 1)[-1] in _dup_runtime_dll_names
    )
])

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LipNetConnect1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
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
    name='LipNetConnect1',
)
