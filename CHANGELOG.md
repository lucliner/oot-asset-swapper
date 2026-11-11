# Changelog

## [1.0.0] - 2026-11-11

First release.

### Added
- Model import (glTF, OBJ; FBX/DAE via Blender), automatic retargeting to the game rig, automatic weights for unrigged models.
- Tk UI with bone map, parts, textures, fit and attachments, equipment, and backups tabs; command-line `swap`, `inspect`, `preview`, `backup`.
- In-place BC1/ASTC texture encoding that rewrites every mip level.
- Backups and revert of mods (`mods/_asset-swap-backups`).
- Profile options: `merge_parts`, `merge_into`, `flatten_back`, `fit_skeleton`, `arm_stiffness`, `joint_fit`, `hide_models`.
- Windows executable build (`build_exe.bat`).

### Fixed
- See-through holes: back-face winding repair and merging of gap-closing parts.
- Shield and sword clipping through the body (back flattening).
- Arms bending at the wrong pivots while jumping (arm-bone fitting) and zigzagging elbows (`arm_stiffness`).
- Textures reverting at distance (all mip levels rewritten).
- PyInstaller build script failing on a trailing backslash.
