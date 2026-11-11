"""Asset Swapper: a window for replacing the game's character models with models from elsewhere.

    python -I tools/assetswap ui

Pick the game's romfs, the model to replace (Link, an NPC), the model to put there, and check the preview. Everything the
automatic steps get wrong can be corrected here: which game bone each joint of the model drives (Bones), which of the
game's shapes carries each part (Parts), the images (Textures), the fit and the weapon attachment points (Fit), and
equipment models to hide (Equipment). Build writes a mod folder; the profile (all the corrections) is saved beside it.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preview  # noqa: E402
import swap  # noqa: E402

import backup  # noqa: E402
import settings  # noqa: E402

_GAME = settings.find_game()
DEFAULT_ROMFS = os.path.join(_GAME, 'romfs') if _GAME else ''
PROFILE_NAME = 'asset_swap_profile.json'
KNOWN_EQUIPMENT = ['Model/Item_AgonyStone.Item_AgonyStone.bfres.zs']
MODEL_TYPES = [('Models', '*.fbx *.glb *.gltf *.obj *.dae'), ('All files', '*.*')]


def list_targets(romfs):
    d = os.path.join(romfs, 'Model')
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.bfres.zs') and '.anim.' not in f and '.Skeleton.' not in f and re.match(r'(Player|Npc|Enemy|Horse|Animal)_', f):
            a, b = f.split('.')[:2]
            if a == b or a.startswith('Player'):
                out.append('Model/' + f)
    return out


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Sheikah Asset Swapper')
        self.geometry('1380x860')
        self.q = queue.Queue()
        self.plan = None
        self.profile = {}
        self.busy = False
        self.photo = None
        self.vars = {k: tk.StringVar() for k in ('romfs', 'target', 'model', 'mod')}
        self.vars['romfs'].set(DEFAULT_ROMFS)
        self.vars['mod'].set('asset-swap')
        self.v_pose, self.v_over, self.v_tex = tk.BooleanVar(), tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.v_yaw, self.v_pitch = tk.DoubleVar(value=20), tk.DoubleVar(value=0)
        self._build_ui()
        self.refresh_targets()
        self.fill_backups()
        self.after(100, self._poll)

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        top = ttk.Frame(self, padding=6)
        top.pack(fill='x')
        def row(r, label, var, browse=None, combo=None):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky='w')
            w = (ttk.Combobox(top, textvariable=var, width=70) if combo else ttk.Entry(top, textvariable=var, width=73))
            w.grid(row=r, column=1, sticky='we', padx=4, pady=1)
            if browse:
                ttk.Button(top, text='Browse', command=browse).grid(row=r, column=2)
            return w
        row(0, 'Game romfs', self.vars['romfs'], self.browse_romfs)
        self.cb_target = row(1, 'Replace', self.vars['target'], combo=True)
        row(2, 'With model', self.vars['model'], self.browse_model)
        row(3, 'Mod name', self.vars['mod'])
        top.columnconfigure(1, weight=1)
        btns = ttk.Frame(top)
        btns.grid(row=0, column=3, rowspan=4, padx=8)
        ttk.Button(btns, text='Load / reload', command=self.load).pack(fill='x', pady=1)
        ttk.Button(btns, text='Update preview', command=self.apply_and_preview).pack(fill='x', pady=1)
        ttk.Button(btns, text='Build mod', command=self.build).pack(fill='x', pady=1)
        ttk.Checkbutton(btns, text='write textures (experimental)', variable=self.v_tex).pack(anchor='w')
        ttk.Button(btns, text='Save profile', command=self.save_profile).pack(fill='x', pady=1)
        ttk.Button(btns, text='Load profile', command=self.load_profile).pack(fill='x', pady=1)
        ttk.Button(btns, text='Open mods folder', command=lambda: os.makedirs(self.mods_root, exist_ok=True) or os.startfile(self.mods_root)).pack(fill='x', pady=1)

        main = ttk.PanedWindow(self, orient='horizontal')
        main.pack(fill='both', expand=True, padx=6)
        nb = ttk.Notebook(main)
        main.add(nb, weight=1)
        right = ttk.Frame(main)
        main.add(right, weight=1)

        # --- bones
        t = ttk.Frame(nb, padding=4); nb.add(t, text='Bones')
        f = ttk.Frame(t); f.pack(fill='x')
        ttk.Label(f, text='Filter').pack(side='left'); self.v_filter = tk.StringVar()
        e = ttk.Entry(f, textvariable=self.v_filter); e.pack(side='left', fill='x', expand=True, padx=4)
        self.v_filter.trace_add('write', lambda *_: self.fill_bones())
        self.v_onlymapped = tk.BooleanVar()
        ttk.Checkbutton(f, text='only mapped', variable=self.v_onlymapped, command=self.fill_bones).pack(side='left')
        ttk.Button(f, text='Reset to automatic', command=self.reset_bones).pack(side='left', padx=4)
        self.tv_bones = ttk.Treeview(t, columns=('joint', 'game', 'src'), show='headings', height=20)
        for c, w, h in (('joint', 220, 'Model joint'), ('game', 170, 'Game bone'), ('src', 80, 'set by')):
            self.tv_bones.heading(c, text=h); self.tv_bones.column(c, width=w)
        self.tv_bones.pack(fill='both', expand=True)
        self.tv_bones.bind('<Double-1>', lambda e: self.edit_cell(self.tv_bones, e, self.bone_choices, self.set_bone))
        ttk.Label(t, text='Double-click a row to choose which game bone the joint drives. Unmapped joints follow their parent.',
                  wraplength=520).pack(anchor='w')

        # --- parts
        t = ttk.Frame(nb, padding=4); nb.add(t, text='Parts')
        self.tv_parts = ttk.Treeview(t, columns=('part', 'mat', 'verts', 'slot'), show='headings', height=10)
        for c, w, h in (('part', 200, 'Part'), ('mat', 160, 'Material'), ('verts', 60, 'Verts'), ('slot', 220, 'Game shape')):
            self.tv_parts.heading(c, text=h); self.tv_parts.column(c, width=w)
        self.tv_parts.pack(fill='both', expand=True)
        self.tv_parts.bind('<Double-1>', lambda e: self.edit_cell(self.tv_parts, e, self.slot_choices, self.set_slot))
        ttk.Label(t, text='Each part goes into one of the game\'s shapes and is drawn with that shape\'s material.\n'
                          'Double-click to pick another shape.', wraplength=520).pack(anchor='w')

        # --- textures
        t = ttk.Frame(nb, padding=4); nb.add(t, text='Textures')
        self.tv_tex = ttk.Treeview(t, columns=('mat', 'img'), show='headings', height=10)
        self.tv_tex.heading('mat', text='Material'); self.tv_tex.column('mat', width=220)
        self.tv_tex.heading('img', text='Image'); self.tv_tex.column('img', width=360)
        self.tv_tex.pack(fill='both', expand=True)
        self.tv_tex.bind('<Double-1>', self.pick_texture)
        ttk.Button(t, text='Find images next to the model', command=self.find_textures).pack(anchor='w', pady=2)

        # --- fit
        t = ttk.Frame(nb, padding=8); nb.add(t, text='Fit & attachments')
        self.fit = {k: tk.StringVar() for k in ('scale', 'rotate_y', 'ox', 'oy', 'oz')}
        for r, (k, label) in enumerate((('scale', 'Scale (blank = fit the target\'s height)'), ('rotate_y', 'Turn around y (degrees)'),
                                        ('ox', 'Offset x (m)'), ('oy', 'Offset y (m)'), ('oz', 'Offset z (m)'))):
            ttk.Label(t, text=label).grid(row=r, column=0, sticky='w'); ttk.Entry(t, textvariable=self.fit[k], width=10).grid(row=r, column=1, sticky='w')
        ttk.Separator(t).grid(row=5, column=0, columnspan=4, sticky='we', pady=8)
        self.v_att = tk.StringVar(value='auto')
        ttk.Label(t, text='Weapon points (sword, shield hand):').grid(row=6, column=0, sticky='w')
        ttk.Radiobutton(t, text='automatic: end of the arm/wing', variable=self.v_att, value='auto').grid(row=7, column=0, sticky='w')
        ttk.Radiobutton(t, text='game default (unchanged)', variable=self.v_att, value='off').grid(row=8, column=0, sticky='w')
        ttk.Radiobutton(t, text='manual (local position under the wrist, m)', variable=self.v_att, value='manual').grid(row=9, column=0, sticky='w')
        self.att = {}
        for r, b in enumerate(('Weapon_L', 'Weapon_R', 'Weapon_Root')):          # hands, and the sheath (Weapon_Root)
            ttk.Label(t, text=b).grid(row=10 + r, column=0, sticky='e')
            self.att[b] = [tk.StringVar() for _ in range(3)]
            for c, v in enumerate(self.att[b]):
                ttk.Entry(t, textvariable=v, width=9).grid(row=10 + r, column=1 + c, sticky='w')
        ttk.Button(t, text='Fill from current plan', command=self.fill_att).grid(row=13, column=0, sticky='w', pady=4)
        ttk.Separator(t).grid(row=14, column=0, columnspan=4, sticky='we', pady=8)
        ttk.Label(t, text='Joint fit: 0 keeps the model\'s own proportions, 1 puts its joints exactly on the game\'s').grid(
            row=15, column=0, columnspan=4, sticky='w')
        self.fit['body_fit'], self.fit['arm_fit'] = tk.StringVar(value='0.2'), tk.StringVar(value='0.3')
        for r, (k, label) in enumerate((('body_fit', 'Torso, legs, neck, head'), ('arm_fit', 'Arms (shoulder anchor)'))):
            ttk.Label(t, text=label).grid(row=16 + r, column=0, sticky='w')
            ttk.Entry(t, textvariable=self.fit[k], width=10).grid(row=16 + r, column=1, sticky='w')
        self.fit['arm_drop'], self.fit['hand_curl'] = tk.StringVar(value='-25'), tk.StringVar(value='15')
        for r, (k, label) in enumerate((('arm_drop', 'Arm drop (degrees, or auto)'), ('hand_curl', 'Hand curl at the wrist (degrees)'))):
            ttk.Label(t, text=label).grid(row=18 + r, column=0, sticky='w')
            ttk.Entry(t, textvariable=self.fit[k], width=10).grid(row=18 + r, column=1, sticky='w')
        ttk.Label(t, text='Negative = arms start raised. A body wider than the game character needs it: the idle animation swings the arms down about the shoulder joint, so a wide belly swallows them. Hands too straight: raise the curl.',
                  wraplength=480, foreground='#888').grid(row=20, column=0, columnspan=4, sticky='w')

        # --- equipment
        t = ttk.Frame(nb, padding=6); nb.add(t, text='Backups & revert')
        ttk.Label(t, text='The game\'s own files are never changed: everything lives in a mod folder. Switching a mod off or '
                          'removing it puts the game back as it was. Every build and every removal first keeps a copy.',
                  wraplength=520).pack(anchor='w')
        ttk.Label(t, text='Mods in the game', font=('', 9, 'bold')).pack(anchor='w', pady=(8, 0))
        self.tv_mods = ttk.Treeview(t, columns=('mod', 'state'), show='headings', height=4)
        self.tv_mods.heading('mod', text='Mod'); self.tv_mods.column('mod', width=260)
        self.tv_mods.heading('state', text='State'); self.tv_mods.column('state', width=100)
        self.tv_mods.pack(fill='x')
        fb = ttk.Frame(t); fb.pack(fill='x', pady=2)
        ttk.Button(fb, text='Switch on', command=lambda: self.mod_action('on')).pack(side='left')
        ttk.Button(fb, text='Switch off', command=lambda: self.mod_action('off')).pack(side='left', padx=2)
        ttk.Button(fb, text='Remove (back to the original game)', command=lambda: self.mod_action('remove')).pack(side='left')
        ttk.Label(t, text='Backups', font=('', 9, 'bold')).pack(anchor='w', pady=(8, 0))
        self.tv_back = ttk.Treeview(t, columns=('mod', 'when', 'files', 'why'), show='headings', height=8)
        for c, w, h in (('mod', 130, 'Mod'), ('when', 130, 'When'), ('files', 50, 'Files'), ('why', 200, 'Why')):
            self.tv_back.heading(c, text=h); self.tv_back.column(c, width=w)
        self.tv_back.pack(fill='both', expand=True)
        fb = ttk.Frame(t); fb.pack(fill='x', pady=2)
        ttk.Button(fb, text='Restore the selected backup', command=self.restore_backup).pack(side='left')
        ttk.Button(fb, text='Back up the selected mod now', command=self.backup_now).pack(side='left', padx=2)
        ttk.Button(fb, text='Delete the selected backup', command=self.delete_backup).pack(side='left')
        ttk.Button(fb, text='Refresh', command=self.fill_backups).pack(side='right')

        t = ttk.Frame(nb, padding=8); nb.add(t, text='Equipment')
        ttk.Label(t, text='Models of Link\'s equipment to hide (emptied in the mod). The Stone of Agony sits inside\n'
                          'a small model and glows through it.', wraplength=520).pack(anchor='w')
        self.lb_hide = tk.Listbox(t, height=8, selectmode='extended')
        self.lb_hide.pack(fill='x', pady=4)
        f = ttk.Frame(t); f.pack(fill='x')
        self.v_hide = tk.StringVar()
        ttk.Entry(f, textvariable=self.v_hide, width=60).pack(side='left')
        ttk.Button(f, text='Add', command=self.add_hide).pack(side='left', padx=2)
        ttk.Button(f, text='Remove selected', command=self.del_hide).pack(side='left')
        for m in KNOWN_EQUIPMENT:
            self.lb_hide.insert('end', m)
        ttk.Separator(t).pack(fill='x', pady=8)
        ttk.Label(t, text='Shield and sword (mounted on Pod_A / Pod_B on the back) must not sit inside the body:',
                  wraplength=520).pack(anchor='w')
        f2 = ttk.Frame(t); f2.pack(fill='x', pady=2)
        self.v_clear = tk.BooleanVar(value=True)
        ttk.Checkbutton(f2, text='push them out of the model, leaving', variable=self.v_clear).pack(side='left')
        self.v_clearance = tk.StringVar(value='0.20')
        ttk.Entry(f2, textvariable=self.v_clearance, width=6).pack(side='left', padx=4)
        ttk.Label(f2, text='m clear').pack(side='left')
        ttk.Label(t, text='Then move by hand (metres in model space: x right, y up, z forward; negative z = further back):',
                  wraplength=520).pack(anchor='w', pady=(6, 0))
        self.offs = {}
        for b in ('Pod_A', 'Pod_B', 'Weapon_Root'):
            fr = ttk.Frame(t); fr.pack(anchor='w')
            ttk.Label(fr, text=b, width=12).pack(side='left')
            self.offs[b] = [tk.StringVar() for _ in range(3)]
            for v in self.offs[b]:
                ttk.Entry(fr, textvariable=v, width=8).pack(side='left', padx=2)

        # --- preview
        pf = ttk.Frame(right); pf.pack(fill='x')
        ttk.Checkbutton(pf, text='test pose', variable=self.v_pose, command=self.show_preview).pack(side='left')
        ttk.Checkbutton(pf, text='skeletons', variable=self.v_over, command=self.show_preview).pack(side='left')
        ttk.Label(pf, text='  turn').pack(side='left')
        ttk.Scale(pf, from_=-180, to=180, variable=self.v_yaw, command=lambda _v: self._debounce()).pack(side='left', fill='x', expand=True)
        ttk.Label(pf, text=' tilt').pack(side='left')
        ttk.Scale(pf, from_=-60, to=60, variable=self.v_pitch, command=lambda _v: self._debounce()).pack(side='left', fill='x', expand=True)
        self.canvas = tk.Label(right, background='#18181c')
        self.canvas.pack(fill='both', expand=True, pady=4)
        ttk.Label(right, text='cyan: the game\'s bones   orange: the model\'s joints, joined to the game bone each one drives',
                  foreground='#888').pack(anchor='w')

        self.log_box = tk.Text(self, height=9, state='disabled', background='#101014', foreground='#cfd', font=('Consolas', 9))
        self.log_box.pack(fill='x', padx=6, pady=4)
        self._deb = None

    # ---------------------------------------------------------------- helpers
    def log(self, *a):
        self.q.put(('log', ' '.join(str(x) for x in a)))

    def _poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == 'log':
                    self.log_box.configure(state='normal'); self.log_box.insert('end', data + '\n')
                    self.log_box.see('end'); self.log_box.configure(state='disabled')
                elif kind == 'call':
                    data()
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _debounce(self):
        if self._deb:
            self.after_cancel(self._deb)
        self._deb = self.after(120, self.show_preview)

    def run_bg(self, fn, done=None):
        if self.busy:
            return
        self.busy = True
        self.title('Sheikah Asset Swapper (working...)')

        def work():
            try:
                res = fn()
                self.q.put(('call', (lambda: done(res)) if done else (lambda: None)))
            except Exception as e:
                import traceback
                self.log(traceback.format_exc())
                self.q.put(('call', lambda: messagebox.showerror('Asset Swapper', str(e))))
            finally:
                self.q.put(('call', self._idle))
        threading.Thread(target=work, daemon=True).start()

    def _idle(self):
        self.busy = False
        self.title('Sheikah Asset Swapper')

    def refresh_targets(self):
        t = list_targets(self.vars['romfs'].get())
        self.cb_target['values'] = t
        if t and not self.vars['target'].get():
            self.vars['target'].set(next((x for x in t if 'Player_Child.' in x), t[0]))

    def browse_romfs(self):
        d = filedialog.askdirectory(title='Choose the game folder (the one with sheikah.exe) or its romfs folder',
                                    initialdir=os.path.dirname(self.vars['romfs'].get()) or '.')
        if d:
            if os.path.isdir(os.path.join(d, 'romfs')):          # the game folder was chosen
                d = os.path.join(d, 'romfs')
            self.vars['romfs'].set(d)
            s = settings.load(); s['game'] = os.path.dirname(d); settings.save(s)
            self.refresh_targets(); self.fill_backups()

    @property
    def mods_root(self):
        """The mods folder of the game: beside the romfs folder."""
        return os.path.join(os.path.dirname(self.vars['romfs'].get().rstrip('\\/')), 'mods')

    def browse_model(self):
        p = filedialog.askopenfilename(filetypes=MODEL_TYPES)
        if p:
            self.vars['model'].set(p)

    @property
    def mod_dir(self):
        return os.path.join(self.mods_root, self.vars['mod'].get().strip() or 'asset-swap')

    # ---------------------------------------------------------------- profile <-> controls
    def collect(self):
        p = self.profile
        def num(s):
            s = s.get().strip()
            return float(s) if s else None
        p['scale'] = num(self.fit['scale'])
        p['rotate_y'] = num(self.fit['rotate_y']) or 0.0
        p['offset'] = [num(self.fit[k]) or 0.0 for k in ('ox', 'oy', 'oz')]
        mode = self.v_att.get()
        if mode == 'auto':
            p['attachments'] = 'auto'
        elif mode == 'off':
            p['attachments'] = None
        else:
            p['attachments'] = {b: [float(v.get() or 0) for v in vs] for b, vs in self.att.items()}
        p['hide_models'] = list(self.lb_hide.get(0, 'end'))
        p['equipment_auto_clear'] = bool(self.v_clear.get())
        try:
            p['equipment_clearance'] = float(self.v_clearance.get() or 0.28)
            p['attach_offsets'] = {b: [float(v.get() or 0) for v in vs] for b, vs in self.offs.items()}
        except ValueError:
            pass
        try:
            body, arm = float(self.fit['body_fit'].get() or 0.2), float(self.fit['arm_fit'].get() or 0.3)
            p['joint_fit'] = {'default': body, 'Clavicle': arm, 'Arm_': arm, 'Wrist': arm}
            v = self.fit['arm_drop'].get().strip().lower()
            p['arm_drop'] = 'auto' if v in ('', 'auto') else float(v)
            p['hand_curl'] = float(self.fit['hand_curl'].get() or 0)
        except ValueError:
            pass
        return p

    def distribute(self):
        p = self.profile
        self.fit['scale'].set('' if not p.get('scale') else str(p['scale']))
        self.fit['rotate_y'].set(str(p.get('rotate_y') or 0))
        for k, v in zip(('ox', 'oy', 'oz'), p.get('offset') or (0, 0, 0)):
            self.fit[k].set(str(v))
        self.v_clear.set(p.get('equipment_auto_clear', True)); self.v_clearance.set(str(p.get('equipment_clearance', 0.28)))
        for b, vs in (p.get('attach_offsets') or {}).items():
            if b in self.offs:
                for var, x in zip(self.offs[b], vs):
                    var.set(str(x))
        self.fit['arm_drop'].set(str(p.get('arm_drop', -25))); self.fit['hand_curl'].set(str(p.get('hand_curl', 15)))
        jf = p.get('joint_fit') or {}
        self.fit['body_fit'].set(str(jf.get('default', 0.2))); self.fit['arm_fit'].set(str(jf.get('Arm_', 0.3)))
        a = p.get('attachments', 'auto')
        self.v_att.set('auto' if a == 'auto' else 'off' if a is None else 'manual')
        if isinstance(a, dict):
            for b, vs in a.items():
                if b in self.att:
                    for var, x in zip(self.att[b], vs):
                        var.set(str(x))
        if 'hide_models' in p:
            self.lb_hide.delete(0, 'end')
            for m in p['hide_models']:
                self.lb_hide.insert('end', m)

    def save_profile(self):
        self.collect()
        os.makedirs(self.mod_dir, exist_ok=True)
        path = os.path.join(self.mod_dir, PROFILE_NAME)
        data = {'model': self.vars['model'].get(), 'target': self.vars['target'].get(), 'profile': self.profile}
        json.dump(data, open(path, 'w', encoding='utf-8'), indent=1)
        self.log('profile saved: ' + path)

    def load_profile(self):
        p = filedialog.askopenfilename(initialdir=self.mod_dir, filetypes=[('Profile', '*.json')])
        if not p:
            return
        d = json.load(open(p, encoding='utf-8'))
        self.vars['model'].set(d.get('model', '')); self.vars['target'].set(d.get('target', ''))
        self.profile = d.get('profile', {})
        self.distribute()
        self.load()

    # ---------------------------------------------------------------- loading and preview
    def load(self):
        if not self.vars['model'].get() or not self.vars['target'].get():
            messagebox.showinfo('Asset Swapper', 'Choose the model to replace and the model to put there first.')
            return
        self.collect()
        romfs, target, model = self.vars['romfs'].get(), self.vars['target'].get(), self.vars['model'].get()

        def work():
            self.log('loading %s ...' % os.path.basename(model))
            return swap.Plan(romfs, target, model, self.profile, self.log)
        self.run_bg(work, self._loaded)

    def _loaded(self, plan):
        self.plan = plan
        self.fill_bones(); self.fill_parts(); self.fill_textures()
        self.show_preview()

    def apply_and_preview(self):
        if self.plan is None:
            return self.load()
        self.load()

    def show_preview(self):
        if self.plan is None:
            return
        plan, pose, over = self.plan, self.v_pose.get(), self.v_over.get()
        yaw, pitch = self.v_yaw.get(), self.v_pitch.get()
        try:
            tris = plan.triangles(pose)
            img = preview.render_image(tris, yaw, pitch, 620, plan.overlays() if over and not pose else (), ref=tris)
        except Exception as e:
            self.log('preview failed: %s' % e)
            return
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.photo)

    # ---------------------------------------------------------------- bones tab
    def game_bone_names(self):
        return sorted(b.name for b in self.plan.f.bones if b.smooth >= 0)

    def bone_choices(self, _row):
        return ['(none)'] + self.game_bone_names()

    def fill_bones(self):
        self.tv_bones.delete(*self.tv_bones.get_children())
        if self.plan is None:
            return
        if self.plan.rig is None:
            self.tv_bones.insert('', 'end', values=('(this model has no skeleton)', '', 'weights are automatic'))
            return
        flt = self.v_filter.get().lower()
        manual = self.profile.get('bone_map') or {}
        for n in self.plan.rig.names:
            g = self.plan.bone_map.get(n, '')
            if self.v_onlymapped.get() and not g:
                continue
            if flt and flt not in n.lower() and flt not in g.lower():
                continue
            self.tv_bones.insert('', 'end', iid=n, values=(n, g, 'you' if n in manual else ('name' if g else '')))

    def set_bone(self, joint, value):
        bm = self.profile.setdefault('bone_map', {})
        bm[joint] = None if value in ('(none)', '') else value
        self.load()

    def reset_bones(self):
        self.profile['bone_map'] = {}
        self.load()

    # ---------------------------------------------------------------- parts / textures
    def slot_choices(self, _row):
        f = self.plan.f
        return ['-1  (leave this part out)'] + ['%d  %s  [%s]' % (s.index, s.name[:40], f.materials[s.material][:24]) for s in f.shapes if s.vertex_count and s.skin_count >= 2]

    def fill_parts(self):
        self.tv_parts.delete(*self.tv_parts.get_children())
        if self.plan is None:
            return
        f = self.plan.f
        for m in self.plan.meshes:
            si = self.plan.part_slot.get(m.name)
            self.tv_parts.insert('', 'end', iid=m.name, values=(m.name, m.material, len(m.pos), '%s  %s' % (si, f.shapes[si].name[:30]) if si is not None else ''))

    def set_slot(self, part, value):
        self.profile.setdefault('slots', {})[part] = int(value.split()[0])
        self.load()

    def fill_textures(self):
        self.tv_tex.delete(*self.tv_tex.get_children())
        if self.plan is None:
            return
        tm = self.profile.get('texture_map') or {}
        seen = set()
        for m in self.plan.meshes:
            if m.material in seen:
                continue
            seen.add(m.material)
            self.tv_tex.insert('', 'end', iid=m.material or '(none)', values=(m.material, tm.get(m.material) or m.texture or ''))

    def pick_texture(self, e):
        row = self.tv_tex.identify_row(e.y)
        if not row:
            return
        p = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg *.tga *.bmp'), ('All', '*.*')])
        if p:
            self.profile.setdefault('texture_map', {})[row] = p
            self.fill_textures()

    def find_textures(self):
        if self.plan is None:
            return
        base = os.path.dirname(os.path.dirname(self.vars['model'].get()))
        imgs = []
        for root, _d, files in os.walk(base):
            imgs += [os.path.join(root, x) for x in files if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tga'))]
        norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
        tm = self.profile.setdefault('texture_map', {})
        for m in self.plan.meshes:
            if tm.get(m.material) or m.texture:
                continue
            key = norm(re.sub(r'_?\d{3}$', '', m.material))
            hits = [i for i in imgs if key and key in norm(os.path.basename(i)) and '_n' not in os.path.basename(i).lower()[-8:]]
            if hits:
                tm[m.material] = sorted(hits)[0]
        self.fill_textures()
        self.log('textures matched by name where possible; double-click a row to choose another')

    # ---------------------------------------------------------------- fit / equipment
    def fill_att(self):
        if self.plan is None:
            return
        for b, vs in self.att.items():
            idx = next((x.index for x in self.plan.f.bones if x.name == b), None)
            pos = self.plan.bone_positions.get(idx)
            if pos:
                for var, x in zip(vs, pos):
                    var.set('%.4f' % x)

    def add_hide(self):
        v = self.v_hide.get().strip()
        if v:
            self.lb_hide.insert('end', v); self.v_hide.set('')

    def del_hide(self):
        for i in reversed(self.lb_hide.curselection()):
            self.lb_hide.delete(i)

    # ---------------------------------------------------------------- backups and revert
    def fill_backups(self):
        if not hasattr(self, 'tv_mods'):
            return
        self.tv_mods.delete(*self.tv_mods.get_children())
        self.tv_back.delete(*self.tv_back.get_children())
        md = self.mods_root
        for name, on in backup.mods(md):
            self.tv_mods.insert('', 'end', iid=name, values=(name, 'on' if on else 'off'))
        for mod, stamp, path, meta in backup.list_snapshots(md):
            when = '%s-%s-%s %s:%s' % (stamp[0:4], stamp[4:6], stamp[6:8], stamp[9:11], stamp[11:13])
            self.tv_back.insert('', 'end', iid=path, values=(mod, when, meta.get('files', ''), meta.get('reason', '')))

    def _selected_mod(self):
        s = self.tv_mods.selection()
        if s:
            return s[0]
        name = self.vars['mod'].get().strip()
        return name or None

    def mod_action(self, action):
        mod = self._selected_mod()
        if not mod:
            return messagebox.showinfo('Asset Swapper', 'Select a mod first.')
        try:
            if action == 'remove':
                if not messagebox.askyesno('Remove mod', 'Remove "%s" from the game? The game goes back to the original models. '
                                                         'A backup is kept first, so you can restore it.' % mod):
                    return
                snap = backup.remove(self.mods_root, mod)
                self.log('removed %s; backup kept at %s' % (mod, snap))
            else:
                self.log('%s switched %s: %s' % (mod, action, backup.set_enabled(self.mods_root, mod, action == 'on')))
        except Exception as e:
            messagebox.showerror('Asset Swapper', str(e))
        self.fill_backups()

    def restore_backup(self):
        s = self.tv_back.selection()
        if not s:
            return messagebox.showinfo('Asset Swapper', 'Select a backup first.')
        if messagebox.askyesno('Restore', 'Replace the current mod with this backup? What is there now is backed up first.'):
            self.log('restored: ' + backup.restore(self.mods_root, s[0]))
            self.fill_backups()

    def backup_now(self):
        mod = self._selected_mod()
        snap = backup.snapshot(self.mods_root, mod, 'by hand') if mod else None
        self.log('backed up to %s' % snap if snap else 'nothing to back up')
        self.fill_backups()

    def delete_backup(self):
        s = self.tv_back.selection()
        if s and messagebox.askyesno('Delete backup', 'Delete this backup for good?'):
            backup.delete_snapshot(s[0])
            self.fill_backups()

    # ---------------------------------------------------------------- generic cell editor
    def edit_cell(self, tv, event, choices, setter):
        row, col = tv.identify_row(event.y), tv.identify_column(event.x)
        if not row or self.plan is None:
            return
        top = tk.Toplevel(self)
        top.title('Choose')
        top.geometry('+%d+%d' % (event.x_root, event.y_root))
        opts = choices(row)
        lb = tk.Listbox(top, height=min(18, len(opts)), width=max(40, max(len(o) for o in opts) + 2))
        for o in opts:
            lb.insert('end', o)
        lb.pack()
        def ok(_e=None):
            s = lb.curselection()
            if s:
                setter(row, opts[s[0]])
            top.destroy()
        lb.bind('<Double-1>', ok); lb.bind('<Return>', ok)
        top.bind('<Escape>', lambda _e: top.destroy())
        lb.focus_set()

    # ---------------------------------------------------------------- build
    def build(self):
        if self.plan is None:
            return self.load()
        self.collect()
        self.save_profile()
        mod = self.mod_dir
        tex = self.v_tex.get()
        romfs, target, model = self.vars['romfs'].get(), self.vars['target'].get(), self.vars['model'].get()

        def work():
            plan = swap.Plan(romfs, target, model, self.profile, self.log)       # fresh, from the current profile
            plan.write(mod, tex, self.log)
            return plan
        def done(plan):
            self.plan = plan
            self.log('Built: %s  (restart the game to use it; the Backups tab switches it off or removes it)' % mod)
            self.fill_backups()
        self.run_bg(work, done)


def main():
    App().mainloop()


if __name__ == '__main__':
    main()
