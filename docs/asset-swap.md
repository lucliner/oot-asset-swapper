# Asset Swap

Replaces the geometry (and the colour textures) of a character model (Link, an NPC) with a model from another game, as a
**mod**: nothing in the game or its executable changes. `tools/assetswap` reads your own romfs and writes only the folder
you name.

## The window

Run `Asset Swapper.exe` (built by `tools/assetswap/build_exe.bat`; put the folder anywhere, it finds the game folder, or asks for it), or
`python -I tools/assetswap ui` from this folder.

1. **Replace**: the game's model (`Player_Child`, `Player_Adult`, an NPC). The game starts as the child.
2. **With model**: a `.fbx`, `.glb`/`.gltf`, `.obj` or `.dae`. Formats other than glTF and OBJ are converted with Blender
   (set `BLENDER` to `blender.exe` if it is not found; the default is `E:\Blender\blender.exe`).
3. **Load / reload** builds the plan and draws a preview: cyan are the game's body bones, orange the model's joints with a
   line to the game bone each one drives (a long line is a wrong mapping). *test pose* bends arms, a knee and the head to
   show whether the skin weights follow.
4. Correct what the automatic steps got wrong:

| Tab | What you can fix |
|---|---|
| **Bones** | which game bone each joint of the model drives. Names are matched automatically (MMD, Mixamo, Unity, Blender names); double-click a row to choose another bone or *(none)*. Unmapped joints follow their parent. |
| **Parts** | which of the game's shapes carries each part of the model. A part is drawn with that shape's material. |
| **Textures** | the base-colour image of each material (double-click, or *Find images next to the model*). |
| **Fit & attachments** | scale (default: the target's height), turn, offset; the weapon points (`Weapon_L`, `Weapon_R`, the sheath `Weapon_Root`): automatic (the end of the arm or wing), unchanged, or typed in. *Joint fit* (0 keeps the model's own proportions, 1 puts its joints exactly on the game's; low for torso and legs so they are not squashed), *arm drop* and *hand curl* (arms hang lower and the hands bend at the wrist instead of sticking out like wings). |
| **Equipment** | models of Link's equipment to hide. `Item_AgonyStone` is listed: the Stone of Agony is its own small model, and on a body smaller than Link's it sits inside it and glows through. The shield and sword are mounted on `Pod_A` and `Pod_B` on the back: they are pushed out of the model until clear of its surface (*leaving ... m clear*), and each can be moved by hand (metres, model space). |

5. **Build mod** writes `<game folder>\mods\<name>\romfs\...` and saves your corrections as `asset_swap_profile.json` there
   (load it again with *Load profile*). Start the game; rename the mod folder with a leading `-` to switch it off.

The same from the command line: `python -I tools/assetswap swap <romfs> Model/Player_Child.Player_Child.bfres.zs
<model.fbx> --mod <mods>\<name> --profile profile.json --preview out.png [--pose]`. Other commands: `inspect`, `preview`,
`preview-import`, `export-obj`.

## Backups and reverting

The game's own files are never changed: everything the tool makes is a mod folder under the game's `mods` folder, so
switching the mod off (a leading `-` on the folder) or removing it puts the game back as it was. Before every build the
existing mod is copied to `mods\_asset-swap-backups\<mod>\<date-time>` (the game skips folders that start with `_`; the last 12 are
kept). The **Backups & revert** tab lists the mods and backups: switch a mod on or off, remove it (a backup is kept first), restore
any backup (what is there is backed up first, so a restore can be undone) or back up by hand. The same from the command line:
`backup list|create|restore|on|off|remove --mods <mods folder> [--mod name] [--stamp when]`.

## What it does

1. **Reads** the model in metres. A rigged model keeps its own skin weights; an unrigged one gets automatic weights from
   the distance to the game's body bones.
2. **Fits** it to the target (height, ground, centre).
3. **Retargets** a rigged model: each joint is mapped to a game bone; the mesh is moved from the model's rest pose into the
   game's bind pose, bone by bone (direction and length matched), so the model's own weights stay valid and the game's
   animations drive it.
4. **Patches** the target's BFRES: the chosen shapes get the new geometry in their own vertex layout, the other shapes are
   emptied, the fields tied to geometry (counts, offsets, skin lists, bounding spheres, relocation table) are rewritten.
   Skeleton, materials and vertex layouts are kept.
5. **Textures**: the material of each shape names its texture files (`Tex/<Material>_<Model>_{Alb,Nrm,Rgh,AO}.bntx.zs`).
   The mod replaces those files: the albedo with your image, the other maps with neutral values. The pixels are written
   **in place** into the game's own file, in its own block-compressed format and at its own size (`block_encode.py` encodes
   your image as BC1 or ASTC, whichever the original uses), and every size and table of the file is left alone. Rewriting a
   texture's internal sizes, or writing a texture of another size or an uncompressed one, made the game stall on the loading
   screen. Textures shared with other models are not touched, and a part is only put in a shape whose albedo file no other
   part uses (one file holds one image).

## Limits

- The image has the donor texture's size (the child's skin atlas is 768 x 768, the vest 1536 x 1536), whatever size yours is.
  ASTC is encoded with one colour pair per block and a coarser weight grid, so fine detail inside a block is softened.
- A model with more parts than the target has shapes with an albedo of their own loses its smallest parts (the tool says so).
- Shapes keep the game's materials: a part is drawn with the shading of the shape it is put in (cloth, skin), not with the
  model's own materials. Normal, roughness and occlusion maps are neutral.
- Link's face animation, eyes and hair physics follow the game's rig, not the model's.
- 65,535 vertices per part; 8 skin influences per vertex at most (fewer in shapes with narrower layouts).

The layout notes the writer relies on are in the header of `tools/assetswap/bfres_write.py`.

## Notes from testing (Peter Griffin)

- **Equipment mounts cannot be moved.** The game takes the shield's and sword's positions from the animation, so editing `Pod_A`/`Pod_B` (`attach_offsets`, `equipment_clearance`) has no effect in game. A body deeper than Link's is instead pressed in at the back (`flatten_back`, default on; `flatten_margin` metres).
- **Parts that close gaps** (a model's cuffs, hems or caps kept as a separate mesh) are joined to the main mesh with `merge_parts` (names) and `merge_into` (host part): their texture colours are matched to texels of the host's atlas. Leaving them out leaves see-through holes.
- **Mip levels:** every level of a texture's mip chain is rewritten in place (sizes and offsets unchanged), so shrunk or steep surfaces do not show the game's original texture.
- **`fit_skeleton`**: moves the game's bones to the model's joints. For all bones (`true`) this distorted the body in game, so it is off by default; a list of bone name prefixes (`["Clavicle", "Arm_", "Wrist"]`) fits only those bones, which makes the arms bend at the model's own shoulders and elbows (jumping, reaching). Use `joint_fit` 0 for the same bones.
- **`arm_stiffness`** (0..1, default 0.6): moves that share of the forearm and hand weight to the upper arm, so Link's elbow and wrist bends do not fold a shorter, thicker arm into a zigzag.
- **`arm_length`** (default 1): a model whose arms are too short for its body under the game's rig gets longer arms: everything beyond the shoulder is moved out along the arm in proportion to its distance from the shoulder, and the arm bones are placed at the stretched joints (Fit tab, *Arm length*). Peter Griffin uses 1.2.
- **Worn equipment is hidden** (`hide_worn`, default on): the Goron bracelet, the gauntlets, the boots and the Goron and Zora tunics are drawn over Link's body by the game and would float on a new skin, so their models are emptied in the mod. Weapons and shields stay.
- **Held weapons** (swords, Deku stick, hammer in the right hand; bow and slingshot in the left): the animations set the weapon bones' positions, so the models themselves are shifted in the mod (`shift_held`, default on) by the distance from the game's weapon point on the new skeleton to the centre of the model's hand, measured before the arm weights are stiffened.
- **`armpit_relief`** (default 1, 0 = off): torso vertices that the model's weights hand to the arm (chest and ribs under the sleeve) follow the arm when the game swings it down, which sucks the armpit in. Everything beside the shoulder that is farther from the arm's axis than the arm is thick gives its arm weight back to `Spine_2`.
- **`hand_follows_wrist`** (0..1, default 0): 0 keeps the hand stiff with the arm (straight arms, but a sword mounted on the animated wrist floats a little off the hand); 1 lets the hand turn with the wrist (the sword stays in it, but Link's elbow and wrist bends fold a short thick arm). **`shift_held`** (default off) shifts the sword, hammer and bow models by the difference between the model's hand and Link's; in testing it did not help.

## Works with any recomp of the game

Any static recompilation of Ocarina of Time for Switch 2 (Hookshot and the projects built on it) uses the same romfs and the
same mod layout (`<game folder>/mods/<name>/romfs/...`). Point *Game romfs* at the `romfs` folder of the one you use: the mod
goes into the `mods` folder beside it. The tool finds `romfs` beside the program, in the last folder used, or you browse to it.

A model bigger than the game's own needs a larger entry in the resource size table, or the game goes on without the file.
The tool writes that table into the mod for you. Only one mod's table is used by the game, so if several mods each make a file
bigger, keep one table with all the entries in the mod whose folder name sorts last.
