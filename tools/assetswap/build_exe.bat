@echo off
rem Builds "Asset Swapper.exe" (a folder with the program and its libraries) with PyInstaller.
rem Needs Python 3.12-3.14 with:  pip install pyinstaller numpy pillow zstandard moderngl texture2ddecoder   (Python 3.15 is not supported by PyInstaller yet)
rem Usage:  build_exe.bat [python.exe] [output folder]
setlocal
set PY=%~1
if "%PY%"=="" set PY=python
set OUT=%~2
if "%OUT%"=="" set OUT=%~dp0..\..\..\AssetSwapper-dist
set SRC=%~dp0
"%PY%" -m PyInstaller --noconfirm --clean --onedir --windowed --name "Asset Swapper" --paths "%SRC%." --hidden-import glcontext --hidden-import texture2ddecoder ^
  --add-data "%SRC%blender_convert.py;." --add-data "%SRC%__main__.py;." ^
  --distpath "%OUT%" --workpath "%TEMP%\assetswap-build" --specpath "%TEMP%\assetswap-build" "%SRC%asset_swapper.py"
echo.
echo Built: %OUT%\Asset Swapper\Asset Swapper.exe
endlocal
