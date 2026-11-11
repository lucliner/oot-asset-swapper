"""Entry point of the Asset Swapper program (the exe is built from this file).

    Asset Swapper.exe                 opens the window
    Asset Swapper.exe <command> ...   the command-line tool (swap, inspect, preview, backup, ...), as `python tools/assetswap`
"""
import os
import sys

BASE = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# the modules are imported by name here so that the packager finds them all
import autoskin, backup, bfres, bfres_write, block_encode, bntx, bntx_flat, containers, gltf_io  # noqa: E402,F401,E401
import importers, posing, preview, restbl, retarget, settings, skeleton, swap, textures  # noqa: E402,F401,E401


def main():
    if len(sys.argv) > 1:
        import runpy
        runpy.run_path(os.path.join(BASE, '__main__.py'), run_name='__main__')
    else:
        import ui
        ui.main()


if __name__ == '__main__':
    main()
