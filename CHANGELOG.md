# Changelog

## [1.1.0] - 2026-11-11

### Added
- `arm_length` profile option and an *Arm length* field on the Fit tab: lengthens the arms of models whose proportions are larger than the game's skeleton.
- Worn equipment (Goron bracelet, gauntlets, boots, Agony Stone) is hidden automatically on a swapped player; weapons and shields still show (`hide_worn`, default on).
- `armpit_relief`: torso vertices beside the shoulder give their arm weight back to the spine, which removes the sucked-in armpit look (default on).
- `hand_follows_wrist` (0 to 1, default 0): at 0 the arms stay straight and the held sword sits near the hand; at 1 the sword fits the hand but the arms fold like Link's.
- `shift_held` (default off): experimental shifted copies of held weapon models.
- Link's skin thickness and gradient maps (`Thc`, `Gn2`) are written as flat colours so his UV layout no longer shows on an imported face (intended to remove the checkerboard on the face; not yet confirmed in game).

### Changed
- `arm_stiffness` keeps the hand rigid with the upper arm by default.

### Known issues
- With straight arms the sword floats near the hand while attacking, because the game attaches it to the animated wrist.
- Leg textures have not been investigated.

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
