"""Run inside Blender: convert a model Blender can import (FBX, DAE, OBJ, PMX with an add-on, ...) to a GLB with its skeleton.

    blender -b -P blender_convert.py -- <input> <output.glb>

The asset swapper calls this when it is given a format it does not read itself. Meters, +y up, skin weights kept.
"""
import os
import sys

import bpy

src, out = sys.argv[sys.argv.index('--') + 1:][:2]
ext = os.path.splitext(src)[1].lower()
bpy.ops.wm.read_factory_settings(use_empty=True)
if ext == '.fbx':
    bpy.ops.import_scene.fbx(filepath=src)
elif ext in ('.glb', '.gltf'):
    bpy.ops.import_scene.gltf(filepath=src)
elif ext == '.obj':
    bpy.ops.wm.obj_import(filepath=src)
elif ext == '.dae':
    bpy.ops.wm.collada_import(filepath=src)
else:
    raise SystemExit('unsupported input format: ' + ext)
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_skins=True, export_yup=True,
                          export_apply=False, export_image_format='AUTO', export_texcoords=True, export_normals=True)
print('CONVERTED', out)
