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

import numpy as np
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


# ---- look: a flat dark theme on ttk's clam
BG, PANEL, FIELD, EDGE = '#14161b', '#1c1f26', '#252932', '#303541'
FG, MUTED, ACCENT, ACCENT_HI = '#e6e9ef', '#8b93a3', '#4f8cff', '#6aa0ff'
OK, WARN = '#4cc38a', '#f0b429'
FONT = ('Segoe UI', 10)

# the body parts a person thinks in, in the order they read: (label, game bone, sided)
BODY = [('Head', 'Head', False), ('Neck', 'Neck', False), ('Chest', 'Spine_2', False), ('Belly / lower back', 'Spine_1', False),
        ('Hips', 'Waist', False),
        ('Shoulder', 'Clavicle', True), ('Upper arm', 'Arm_1', True), ('Forearm', 'Arm_2', True), ('Hand', 'Wrist', True),
        ('Thigh', 'Leg_1', True), ('Shin', 'Leg_2', True), ('Foot', 'Ankle', True), ('Toes', 'Toe', True)]


def _indicators(root, st):
    """Check box and radio button images drawn in the theme's colours, scaled to the screen."""
    from PIL import Image, ImageDraw
    k = max(1.0, float(root.tk.call('tk', 'scaling')) / 1.333)
    n = int(round(18 * k))
    big = n * 4

    def make(kind, on, hot):
        im = Image.new('RGBA', (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        fill = ACCENT if on else FIELD
        edge = ACCENT_HI if (hot or on) else MUTED
        box = [big * 0.08, big * 0.08, big * 0.92, big * 0.92]
        if kind == 'check':
            d.rounded_rectangle(box, radius=big * 0.18, fill=fill, outline=edge, width=int(big * 0.07))
            if on:
                d.line([(big * 0.26, big * 0.52), (big * 0.43, big * 0.68), (big * 0.75, big * 0.32)], fill='#ffffff', width=int(big * 0.13), joint='curve')
        else:
            d.ellipse(box, fill=fill, outline=edge, width=int(big * 0.07))
            if on:
                d.ellipse([big * 0.32, big * 0.32, big * 0.68, big * 0.68], fill='#ffffff')
        return ImageTk.PhotoImage(im.resize((n, n), Image.LANCZOS))

    root._ind = {(kind, on, hot): make(kind, on, hot) for kind in ('check', 'radio') for on in (False, True) for hot in (False, True)}
    for kind, style in (('check', 'Checkbutton'), ('radio', 'Radiobutton')):
        i = root._ind
        st.element_create('Flat.%s.indicator' % style, 'image', i[(kind, False, False)],
                          ('selected', 'active', i[(kind, True, True)]), ('selected', i[(kind, True, False)]), ('active', i[(kind, False, True)]),
                          sticky='w', padding=(0, 0, 6, 0))
        st.layout('T' + style, [('%s.padding' % style, {'sticky': 'nswe', 'children': [
            ('Flat.%s.indicator' % style, {'side': 'left', 'sticky': ''}),
            ('%s.focus' % style, {'side': 'left', 'sticky': 'w', 'children': [('%s.label' % style, {'sticky': 'nswe'})]})]})])


def apply_theme(root):
    st = ttk.Style(root)
    st.theme_use('clam')
    _indicators(root, st)
    root.configure(background=BG)
    root.option_add('*Font', FONT)
    root.option_add('*TCombobox*Listbox.background', FIELD)
    root.option_add('*TCombobox*Listbox.foreground', FG)
    root.option_add('*TCombobox*Listbox.selectBackground', ACCENT)
    st.configure('.', background=BG, foreground=FG, fieldbackground=FIELD, bordercolor=EDGE, lightcolor=BG, darkcolor=BG,
                 troughcolor=PANEL, focuscolor=ACCENT, font=FONT)
    st.configure('TFrame', background=BG)
    st.configure('Panel.TFrame', background=PANEL)
    st.configure('TLabel', background=BG, foreground=FG)
    st.configure('Muted.TLabel', foreground=MUTED)
    st.configure('Title.TLabel', font=('Segoe UI Semibold', 15), foreground=FG)
    st.configure('Sub.TLabel', foreground=MUTED)
    st.configure('Hint.TLabel', background=PANEL, foreground=FG, padding=10)
    st.configure('TEntry', fieldbackground=FIELD, foreground=FG, insertcolor=FG, bordercolor=EDGE, padding=3)
    st.configure('Small.TButton', padding=(9, 3))
    st.map('TEntry', bordercolor=[('focus', ACCENT)])
    st.configure('TCombobox', fieldbackground=FIELD, background=FIELD, foreground=FG, arrowcolor=MUTED, bordercolor=EDGE, padding=4)
    st.map('TCombobox', bordercolor=[('focus', ACCENT)], fieldbackground=[('readonly', FIELD)])
    st.configure('TButton', background=FIELD, foreground=FG, bordercolor=EDGE, padding=(12, 6), relief='flat')
    st.map('TButton', background=[('active', EDGE), ('disabled', PANEL)], foreground=[('disabled', MUTED)])
    st.configure('Accent.TButton', background=ACCENT, foreground='#ffffff', bordercolor=ACCENT, padding=(12, 8), font=('Segoe UI Semibold', 10))
    st.map('Accent.TButton', background=[('active', ACCENT_HI), ('disabled', EDGE)])
    st.configure('TCheckbutton', background=BG, foreground=FG, indicatorcolor=FIELD, indicatorbackground=FIELD, indicatorforeground='#ffffff', bordercolor=MUTED, padding=(2, 2))
    st.map('TCheckbutton', indicatorcolor=[('selected', ACCENT), ('active', EDGE)], indicatorforeground=[('selected', '#ffffff'), ('!selected', FIELD)], background=[('active', BG)])
    st.configure('TRadiobutton', background=BG, foreground=FG, indicatorcolor=FIELD, indicatorforeground='#ffffff')
    st.map('TRadiobutton', indicatorcolor=[('selected', ACCENT)], indicatorforeground=[('selected', '#ffffff')], background=[('active', BG)])
    st.configure('TNotebook', background=BG, bordercolor=BG, tabmargins=(0, 0, 0, 0))
    st.configure('TNotebook.Tab', background=BG, foreground=MUTED, padding=(8, 6), bordercolor=BG, font=FONT)
    st.map('TNotebook.Tab', background=[('selected', PANEL)], foreground=[('selected', FG)])
    st.configure('Treeview', background=PANEL, fieldbackground=PANEL, foreground=FG, bordercolor=EDGE, rowheight=28, borderwidth=0)
    st.map('Treeview', background=[('selected', ACCENT)], foreground=[('selected', '#ffffff')])
    st.configure('Treeview.Heading', background=FIELD, foreground=MUTED, bordercolor=EDGE, padding=(8, 6), font=('Segoe UI Semibold', 9), relief='flat')
    st.map('Treeview.Heading', background=[('active', EDGE)])
    st.configure('TScale', background=BG, troughcolor=FIELD)
    st.configure('Horizontal.TScrollbar', background=FIELD, troughcolor=PANEL, arrowcolor=MUTED, bordercolor=PANEL)
    st.configure('Vertical.TScrollbar', background=FIELD, troughcolor=PANEL, arrowcolor=MUTED, bordercolor=PANEL)
    st.configure('Horizontal.TProgressbar', background=ACCENT, troughcolor=FIELD, bordercolor=FIELD, thickness=4)
    st.configure('TSeparator', background=EDGE)
    st.configure('TPanedwindow', background=BG)
    st.configure('Sash', background=BG, sashthickness=8)


KNOWN_NAMES = {'Player_Child': 'Link (child)', 'Player_Adult': 'Link (adult)', 'Npc_Navi': 'Navi (the fairy)'}


def friendly(rel):
    """'Model/Npc_Zelda_Child.Npc_Zelda_Child.bfres.zs' -> 'Zelda Child'."""
    stem = os.path.basename(rel).split('.')[0]
    if stem in KNOWN_NAMES:
        return KNOWN_NAMES[stem]
    kind, _, rest = stem.partition('_')
    rest = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', rest.replace('_', ' ')) or stem
    return {'Player': 'Link: ', 'Enemy': 'Enemy: ', 'Horse': 'Horse: ', 'Animal': 'Animal: '}.get(kind, '') + rest


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
        apply_theme(self)
        self.title('Asset Swapper')
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry('%dx%d+%d+%d' % (min(1440, sw - 60), min(900, sh - 110), 20, 20))       # always inside the screen
        self.minsize(960, 600)
        self.q = queue.Queue()
        self.plan = None
        self.profile = {}
        self.busy = False
        self.photo = None
        self.vars = {k: tk.StringVar() for k in ('romfs', 'target', 'model', 'mod')}
        self.vars['romfs'].set(DEFAULT_ROMFS)
        self.vars['mod'].set('asset-swap')
        self.v_pose, self.v_over, self.v_tex = tk.BooleanVar(), tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.v_shade = tk.BooleanVar(value=True)
        self.v_xray = tk.BooleanVar(value=False)
        self.v_ghost = tk.BooleanVar(value=False)
        self.tf_A = np.eye(4)                       # the move / turn / scale of the model not yet applied to the fit
        self.tf_c = np.zeros(3)                     # its pivot: the middle of the model
        self.v_mode = tk.StringVar(value='off')     # transform gizmo: off, move, rotate, scale
        self._tf_job = self._fast_job = None
        self._giz = self._gdrag = self._odrag = self._parts_cache = self._ref = None
        self._r_giz = 0.3
        self._giz_active = None
        self._src_labels = {}
        self._copy_info = None
        self.v_yaw, self.v_pitch = tk.DoubleVar(value=20), tk.DoubleVar(value=0)
        self._build_ui()
        self.refresh_targets()
        self.fill_backups()
        self.after(100, self._poll)

    # ---------------------------------------------------------------- layout
    def _toggle_log(self):
        if self.v_showlog.get():
            self.log_box.pack(fill='x', side='bottom', padx=16, pady=(0, 4), before=self._bar)
        else:
            self.log_box.pack_forget()

    def _scrolled(self, nb, title):
        """A tab whose contents scroll when the window is too short to show them all; returns the frame to put them in."""
        outer = ttk.Frame(nb)
        nb.add(outer, text=title)
        cv = tk.Canvas(outer, background=BG, highlightthickness=0, borderwidth=0)
        sb = ttk.Scrollbar(outer, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y'); cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv, padding=8)
        win = cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda _e: cv.configure(scrollregion=cv.bbox('all')))
        cv.bind('<Configure>', lambda e: cv.itemconfigure(win, width=e.width))
        def wheel(e):
            cv.yview_scroll(-1 * (e.delta // 120), 'units')
        outer.bind('<Enter>', lambda _e: cv.bind_all('<MouseWheel>', wheel))
        outer.bind('<Leave>', lambda _e: cv.unbind_all('<MouseWheel>'))
        return inner

    def _build_ui(self):
        head = ttk.Frame(self, padding=(16, 8, 16, 0))
        head.pack(fill='x')
        ttk.Label(head, text='Asset Swapper', style='Title.TLabel').pack(side='left')
        ttk.Label(head, text='   replace the game"s models with your own, as a mod', style='Sub.TLabel').pack(side='left', pady=(4, 0))
        top = ttk.Frame(self, padding=(16, 6, 16, 4))
        top.pack(fill='x')
        def row(r, label, var, browse=None, combo=None, btn='Browse'):
            ttk.Label(top, text=label, style='Muted.TLabel').grid(row=r, column=0, sticky='w', padx=(0, 10))
            w = (ttk.Combobox(top, textvariable=var, width=70) if combo else ttk.Entry(top, textvariable=var, width=73))
            w.grid(row=r, column=1, sticky='we', padx=4, pady=1)
            if browse:
                ttk.Button(top, text=btn, command=browse).grid(row=r, column=2, padx=(0, 4), sticky='we')
            return w
        row(0, 'Game romfs', self.vars['romfs'], self.browse_romfs)
        self.v_target_label = tk.StringVar()
        self.cb_target = row(1, 'Replace', self.v_target_label, self.open_picker, combo=True, btn='Pick model...')
        self.cb_target.configure(state='readonly')
        self.cb_target.bind('<<ComboboxSelected>>', self._target_picked)
        self.vars['target'].trace_add('write', lambda *_: self._target_shown())
        row(2, 'With model', self.vars['model'])
        src = ttk.Frame(top)
        src.grid(row=2, column=2, padx=(0, 4), sticky='we')
        ttk.Button(src, text='Browse', command=self.browse_model).pack(side='left', fill='x', expand=True)
        ttk.Button(src, text='Game model...', command=self.open_source_picker).pack(side='left', fill='x', expand=True, padx=(4, 0))
        row(3, 'Mod name', self.vars['mod'])
        self.v_copy = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='Copy whole model', variable=self.v_copy, command=self._copy_toggled).grid(row=3, column=2, padx=(0, 4), sticky='w')
        top.columnconfigure(1, weight=1)
        btns = ttk.Frame(top)
        btns.grid(row=0, column=3, rowspan=4, padx=(12, 0))
        ttk.Button(btns, text='Load / reload', command=self.load).pack(fill='x', pady=1)
        ttk.Button(btns, text='Update preview', command=self.apply_and_preview).pack(fill='x', pady=1)
        ttk.Button(btns, text='Build mod', style='Accent.TButton', command=self.build).pack(fill='x', pady=(0, 6))
        ttk.Checkbutton(btns, text='write textures (experimental)', variable=self.v_tex).pack(anchor='w')
        ttk.Button(btns, text='Save profile', command=self.save_profile).pack(fill='x', pady=1)
        ttk.Button(btns, text='Load profile', command=self.load_profile).pack(fill='x', pady=1)
        ttk.Button(btns, text='Open mods folder', command=lambda: os.makedirs(self.mods_root, exist_ok=True) or os.startfile(self.mods_root)).pack(fill='x', pady=1)

        main = ttk.PanedWindow(self, orient='horizontal')
        main.pack(fill='both', expand=True, padx=16, pady=(4, 0))
        nb = ttk.Notebook(main)
        main.add(nb, weight=2)
        right = ttk.Frame(main)
        main.add(right, weight=3)
        self.after(200, lambda: main.sashpos(0, int(self.winfo_width() * 0.36)))

        # --- bones
        t = ttk.Frame(nb, padding=4); nb.add(t, text='Bones')
        self.v_hint = tk.StringVar()
        ttk.Label(t, textvariable=self.v_hint, style='Hint.TLabel', wraplength=520, justify='left').pack(fill='x', pady=(0, 6))
        self.tv_body = ttk.Treeview(t, columns=('part', 'joint', 'state'), show='headings', height=13, selectmode='browse')
        for c, w, h in (('part', 170, 'Body part'), ('joint', 230, 'Matched to your model'), ('state', 100, '')):
            self.tv_body.heading(c, text=h); self.tv_body.column(c, width=w)
        self.tv_body.tag_configure('ok', foreground=OK); self.tv_body.tag_configure('miss', foreground=WARN)
        self.tv_body.pack(fill='both', expand=True)
        self.tv_body.bind('<Double-1>', self.edit_body)
        ttk.Label(t, text='Double-click a row to pick a different part of your model for it. Only rows marked check this may need a look.',
                  style='Muted.TLabel', wraplength=520).pack(anchor='w', pady=(4, 0))
        self.v_adv = tk.BooleanVar(value=False)
        fa = ttk.Frame(t); fa.pack(fill='x', pady=(6, 0))
        ttk.Checkbutton(fa, text='Show every joint (advanced)', variable=self.v_adv, command=self._toggle_adv).pack(side='left')
        ttk.Button(fa, text='Undo skeleton drags', command=self.reset_skeleton_edits).pack(side='right')
        self.adv = ttk.Frame(t)
        f = ttk.Frame(self.adv); f.pack(fill='x')
        ttk.Label(f, text='Filter').pack(side='left'); self.v_filter = tk.StringVar()
        e = ttk.Entry(f, textvariable=self.v_filter); e.pack(side='left', fill='x', expand=True, padx=4)
        self.v_filter.trace_add('write', lambda *_: self.fill_bones())
        self.v_onlymapped = tk.BooleanVar()
        ttk.Checkbutton(f, text='only mapped', variable=self.v_onlymapped, command=self.fill_bones).pack(side='left')
        ttk.Button(f, text='Reset to automatic', command=self.reset_bones).pack(side='left', padx=4)
        self.tv_bones = ttk.Treeview(self.adv, columns=('joint', 'game', 'src'), show='headings', height=8)
        for c, w, h in (('joint', 220, 'Model joint'), ('game', 170, 'Game bone'), ('src', 80, 'set by')):
            self.tv_bones.heading(c, text=h); self.tv_bones.column(c, width=w)
        self.tv_bones.pack(fill='both', expand=True)
        self.tv_bones.bind('<Double-1>', lambda e: self.edit_cell(self.tv_bones, e, self.bone_choices, self.set_bone))
        ttk.Label(self.adv, text='Double-click a row to choose which game bone the joint drives. Unmapped joints follow their parent.',
                  style='Muted.TLabel', wraplength=520).pack(anchor='w')

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
        t = self._scrolled(nb, 'Fit')
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
                  wraplength=480, style='Muted.TLabel').grid(row=20, column=0, columnspan=4, sticky='w')
        self.fit['arm_length'] = tk.StringVar(value='1.0')
        ttk.Label(t, text='Arm length (1 = as the model; more = longer arms)').grid(row=21, column=0, sticky='w')
        ttk.Entry(t, textvariable=self.fit['arm_length'], width=10).grid(row=21, column=1, sticky='w')

        # --- equipment
        t = ttk.Frame(nb, padding=6); nb.add(t, text='Backups')
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

        t = self._scrolled(nb, 'Equipment')
        ttk.Label(t, text='Models of Link\'s equipment to hide (emptied in the mod). The Stone of Agony sits inside\n'
                          'a small model and glows through it.', wraplength=520).pack(anchor='w')
        self.lb_hide = tk.Listbox(t, height=8, selectmode='extended', background=FIELD, foreground=FG, selectbackground=ACCENT, borderwidth=0, highlightthickness=1, highlightbackground=EDGE)
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
        ttk.Checkbutton(pf, text='skeletons', variable=self.v_over, command=self.show_preview).pack(side='left', padx=(10, 0))
        ttk.Checkbutton(pf, text='textures', variable=self.v_shade, command=self.show_preview).pack(side='left', padx=(10, 0))
        ttk.Checkbutton(pf, text='X-ray', variable=self.v_xray, command=self._xray_changed).pack(side='left', padx=(10, 0))
        ttk.Checkbutton(pf, text='Original', variable=self.v_ghost, command=self.show_preview).pack(side='left', padx=(10, 0))
        tf = ttk.Frame(right); tf.pack(fill='x', pady=(6, 0))
        ttk.Label(tf, text='Mode', style='Muted.TLabel').pack(side='left')
        for text, val in (('Off', 'off'), ('Move', 'move'), ('Rotate', 'rotate'), ('Scale', 'scale')):
            ttk.Radiobutton(tf, text=text, value=val, variable=self.v_mode, command=self._mode_changed).pack(side='left', padx=(10, 0))
        self.v_sizeinfo = tk.StringVar()
        ttk.Button(tf, text='-', width=2, style='Small.TButton', command=lambda: self._tf_scale(1 / 1.05)).pack(side='left', padx=(18, 0))
        ttk.Button(tf, text='+', width=2, style='Small.TButton', command=lambda: self._tf_scale(1.05)).pack(side='left', padx=(2, 0))
        ttk.Label(tf, textvariable=self.v_sizeinfo, style='Muted.TLabel').pack(side='left', padx=8)
        ttk.Button(tf, text='Match height', style='Small.TButton', command=self._tf_auto).pack(side='left')
        ttk.Button(tf, text='Reset', style='Small.TButton', command=self._tf_reset).pack(side='left', padx=(6, 0))
        holder = tk.Frame(right, background='#0f1115', height=340, width=400)
        holder.pack(fill='both', expand=True, pady=4)
        holder.pack_propagate(False)                         # the picture follows the space, not the other way round
        holder.bind('<Configure>', self._canvas_resized)
        self.canvas = tk.Label(holder, background='#0f1115', borderwidth=0, anchor='nw')
        self.canvas.place(relx=0.5, rely=0.0, anchor='n')
        self.canvas.bind('<ButtonPress-1>', self._press)
        self.canvas.bind('<B1-Motion>', self._motion)
        self.canvas.bind('<ButtonRelease-1>', self._release)
        self.canvas.bind('<Motion>', self._hover)
        self.v_xhint = tk.StringVar()
        self._hint = ttk.Label(right, textvariable=self.v_xhint, style='Muted.TLabel', wraplength=620, justify='left')
        self._hint.pack(anchor='w', side='bottom', before=holder)
        right.bind('<Configure>', lambda e: self._hint.configure(wraplength=max(200, e.width - 8)))

        self.log_box = tk.Text(self, height=5, state='disabled', background='#0f1115', foreground='#b8c4d6', font=('Consolas', 9),
                               borderwidth=0, highlightthickness=1, highlightbackground=EDGE, padx=8, pady=6, insertbackground=FG)
        bar = self._bar = ttk.Frame(self, padding=(16, 2, 16, 10)); bar.pack(fill='x', side='bottom', before=main)          # (packed before the
                                                                                                                # main area, so a small window cuts that, not these)
        self.status = tk.StringVar(value='Ready')
        self.v_showlog = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text='Log', variable=self.v_showlog, command=self._toggle_log).pack(side='right', padx=(10, 0))
        ttk.Label(bar, textvariable=self.status, style='Muted.TLabel').pack(side='left')
        self.progress = ttk.Progressbar(bar, mode='indeterminate', length=160)
        self.progress.pack(side='right')
        self._deb = None
        self._base = self._cam = None
        self._handles = {}
        self._drag = None
        self._pose_view = False

    # ---------------------------------------------------------------- helpers
    def log(self, *a):
        self.q.put(('log', ' '.join(str(x) for x in a)))

    def busy_msg_locked(self):
        return False

    def _poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == 'log':
                    line = data.strip().splitlines()[-1] if data.strip() else ''
                    if line and not self.busy_msg_locked():
                        self.status.set(line[:140])
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
            self.status.set('Still working on the previous step, please wait...')
            return
        self.busy = True
        self.status.set('Working...')
        self.progress.start(12)
        self.title('Asset Swapper (working...)')

        def work():
            err = None
            try:
                res = fn()
            except Exception as e:
                import traceback
                self.log(traceback.format_exc())
                err = str(e)
            self.q.put(('call', self._idle))                    # first, so a dialog (which waits for a click) never leaves it "working"
            if err is not None:
                self.q.put(('call', lambda: self._show_error(err)))
            else:
                self.q.put(('call', (lambda: done(res)) if done else (lambda: None)))
        threading.Thread(target=work, daemon=True).start()

    def _show_error(self, text):
        self.status.set('Something went wrong: ' + text[:120])
        self.lift()
        messagebox.showerror('Asset Swapper', text, parent=self)

    def _idle(self):
        self.busy = False
        self.progress.stop()
        self.status.set('Ready')
        self.title('Asset Swapper')

    def open_picker(self):
        import picker
        romfs = self.vars['romfs'].get()
        targets = list_targets(romfs)
        if not targets:
            return messagebox.showinfo('Asset Swapper', 'Set the game romfs folder first.')
        theme = {'BG': BG, 'PANEL': PANEL, 'FIELD': FIELD, 'EDGE': EDGE, 'FG': FG, 'MUTED': MUTED, 'ACCENT': ACCENT}
        picker.ModelPicker(self, romfs, targets, friendly, theme, self.vars['target'].get(), self._picked)

    def open_source_picker(self):
        import picker
        romfs = self.vars['romfs'].get()
        targets = list_targets(romfs)
        if not targets:
            return messagebox.showinfo('Asset Swapper', 'Set the game romfs folder first.')
        theme = {'BG': BG, 'PANEL': PANEL, 'FIELD': FIELD, 'EDGE': EDGE, 'FG': FG, 'MUTED': MUTED, 'ACCENT': ACCENT}
        cur = self._src_labels.get(self.vars['model'].get())
        picker.ModelPicker(self, romfs, targets, friendly, theme, cur, self._source_picked, title='Choose the model to put in')

    def _source_picked(self, rel):
        label = 'Game model: ' + friendly(rel)
        self._src_labels[label] = rel
        self.vars['model'].set(label)
        self.profile = {}
        if self.vars['target'].get():
            self.after(50, self.load)

    def _source_label(self, rel):
        label = 'Game model: ' + friendly(rel)
        self._src_labels[label] = rel
        self.vars['model'].set(label)

    def model_value(self):
        """The model to put in as the program names it: a file path, or game:<romfs path> for a model of the game."""
        v = self.vars['model'].get().strip()
        rel = self._src_labels.get(v)
        return 'game:' + rel if rel else v

    def _picked(self, rel):
        self.vars['target'].set(rel)
        if self.vars['model'].get().strip():
            self.after(50, self.load)

    def _target_picked(self, _e=None):
        rel = getattr(self, '_labels', {}).get(self.v_target_label.get())
        if rel:
            self.vars['target'].set(rel)
            if self.vars['model'].get().strip():
                self.after(50, self.load)

    def _target_shown(self):
        rel = self.vars['target'].get()
        lab = next((k for k, v in getattr(self, '_labels', {}).items() if v == rel), None)
        self.v_target_label.set(lab or (friendly(rel) if rel else ''))

    def refresh_targets(self):
        t = list_targets(self.vars['romfs'].get())
        self._labels = {}
        for rel in sorted(t, key=lambda r: (not os.path.basename(r).startswith('Player'), friendly(r).lower())):
            lab = friendly(rel)
            if lab in self._labels:
                lab = '%s (%s)' % (lab, os.path.basename(rel).split('.')[0])
            self._labels[lab] = rel
        self.cb_target['values'] = list(self._labels)
        self._target_shown()
        if t and not self.vars['target'].get():
            self.vars['target'].set(next((x for x in t if 'Player_Child.' in x), t[0]))

    def browse_romfs(self):
        d = filedialog.askdirectory(title='Choose the game folder (the one with the recomp program and the romfs folder) or the romfs folder itself',
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
            self.profile = {}                    # a different model: the corrections made for the last one do not apply
            self.after(50, self.load)            # and the preview shows it straight away

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
            p['arm_length'] = float(self.fit['arm_length'].get() or 1)
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
        self.fit['arm_length'].set(str(p.get('arm_length', 1.0)))
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
        data = {'model': self.model_value(), 'target': self.vars['target'].get(), 'profile': self.profile}
        json.dump(data, open(path, 'w', encoding='utf-8'), indent=1)
        self.log('profile saved: ' + path)

    def load_profile(self):
        p = filedialog.askopenfilename(initialdir=self.mod_dir, filetypes=[('Profile', '*.json')])
        if not p:
            return
        d = json.load(open(p, encoding='utf-8'))
        m = d.get('model', '')
        if m.startswith('game:'):
            self._source_label(m[5:])
        else:
            self.vars['model'].set(m)
        self.vars['target'].set(d.get('target', ''))
        self.profile = d.get('profile', {})
        self.distribute()
        self.load()

    # ---------------------------------------------------------------- loading and preview
    def copy_mode(self):
        """True when the whole model file of a game model is to be copied over the target's (the other way to swap)."""
        return bool(self.v_copy.get()) and self.model_value().startswith('game:')

    def _copy_toggled(self):
        if self.model_value().startswith('game:') and self.vars['target'].get():
            self.load()
        elif self.v_copy.get():
            messagebox.showinfo('Asset Swapper', 'Copying a whole model needs a model of the game as the model to put in: use Game model...')
            self.v_copy.set(False)

    def _load_copy(self, romfs, target, model):
        import gametex
        from containers import Romfs

        def work():
            rom = Romfs(romfs)
            self.log('loading %s ...' % os.path.basename(model))
            src, facts = gametex.textured_parts(rom, model[5:], 512)
            orig, _f = gametex.textured_parts(rom, target, 128)
            return src, np.concatenate([q['tris'] for q in orig]) if orig else np.zeros((0, 3, 3)), facts

        def done(res):
            self.plan = None
            self._copy_info = res
            self._base = self._cam = None
            self.fill_bones(); self.fill_parts(); self.fill_textures()
            self.show_preview()
        self.run_bg(work, done)

    def _render_copy(self):
        parts, orig, _facts = self._copy_info
        tris = np.concatenate([q['tris'] for q in parts]) if parts else np.zeros((0, 3, 3))
        ghost = orig if self.v_ghost.get() else None
        ref = np.concatenate([tris, orig])
        try:
            self._base, self._cam = preview.render_base(tris, self.v_yaw.get(), self.v_pitch.get(), self._view_size, ref, None,
                                                         ghost=ghost, parts=parts)
        except Exception as e:
            self.log('preview failed: %s' % e)
            return
        self._pose_view = False
        self._handles, self._giz = {}, None
        self.v_xhint.set('Copy mode: the whole model file is copied over the one it replaces, at its own size (tick Original to see the '
                         'model it replaces). Nothing is refitted.')
        self.photo = ImageTk.PhotoImage(self._base)
        self.canvas.configure(image=self.photo)

    def load(self):
        if not self.vars['model'].get() or not self.vars['target'].get():
            messagebox.showinfo('Asset Swapper', 'Choose the model to replace and the model to put there first.')
            return
        self.collect()
        romfs, target, model = self.vars['romfs'].get(), self.vars['target'].get(), self.model_value()
        if self.copy_mode():
            return self._load_copy(romfs, target, model)
        self._copy_info = None

        def work():
            self.log('loading %s ...' % os.path.basename(model))
            return swap.Plan(romfs, target, model, self.profile, self.log)
        self.run_bg(work, self._loaded)

    def _loaded(self, plan):
        self.plan = plan
        self.tf_A = np.eye(4)
        self._parts_cache = self._ref = None
        base = plan.triangles(False).reshape(-1, 3)
        if len(base):
            self.tf_c = (base.min(0) + base.max(0)) / 2
            self._r_giz = 0.55 * float(np.ptp(base, axis=0).max())
        self._tf_info()
        self.fill_bones(); self.fill_parts(); self.fill_textures()
        self.show_preview()

    def apply_and_preview(self):
        if self.plan is None:
            return self.load()
        self.load()

    # ---- the imported model against the game's own: the original stays where it is; the new one is moved, turned, scaled
    def _live(self, P):
        """Points of the loaded plan with the not-yet-applied transform on them."""
        A = self.tf_A
        return np.asarray(P, np.float64) @ A[:3, :3].T + A[:3, 3]

    def _identity(self):
        return np.allclose(self.tf_A, np.eye(4), atol=1e-9)

    def _k_now(self):
        return float(np.cbrt(abs(np.linalg.det(self.tf_A[:3, :3]))))

    @staticmethod
    def _about(c, M3):
        """4x4 that applies the 3x3 M3 about the point c."""
        m = np.eye(4)
        m[:3, :3] = M3
        m[:3, 3] = np.asarray(c) - M3 @ np.asarray(c)
        return m

    @staticmethod
    def _axis_turn(axis, ang):
        axis = np.asarray(axis, np.float64)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

    def _pivot_now(self):
        return (self.tf_A @ np.append(self.tf_c, 1.0))[:3]

    def _tf_info(self):
        if self.plan is not None:
            sc = self.plan.scale * self._k_now()
            self.v_sizeinfo.set('%d%%' % round(100 * sc / self.plan.auto_scale))

    def _mode_changed(self):
        self.show_preview()

    def _tf_scale(self, f):
        if self.plan is None:
            return
        self.tf_A = self._about(self._pivot_now(), np.eye(3) * f) @ self.tf_A
        self._tf_info(); self.show_preview(); self._tf_schedule(900)

    def _tf_auto(self):
        if self.plan is None:
            return
        self._tf_scale(self.plan.auto_scale / (self.plan.scale * self._k_now()))

    def _tf_reset(self):
        self.profile.pop('rotation', None)
        self.fit['scale'].set('')
        for k in ('ox', 'oy', 'oz'):
            self.fit[k].set('0')
        self.tf_A = np.eye(4)
        self.load()

    def _tf_schedule(self, ms):
        if self._tf_job:
            self.after_cancel(self._tf_job)
        self._tf_job = self.after(max(ms, 1), self._tf_commit)

    def _tf_apply_profile(self):
        """Make the live transform the fit's (scale, rotation, offset) and clear it; no rebuild."""
        if self.plan is None or self._identity():
            return
        v = self.plan.refit(self.tf_A[:3, :3], self.tf_A[:3, 3])
        self.profile['rotation'] = v['rotation']
        self.fit['scale'].set('%.5f' % v['scale'])
        for k, x in zip(('ox', 'oy', 'oz'), v['offset']):
            self.fit[k].set('%.5f' % x)
        self.tf_A = np.eye(4)

    def _tf_commit(self):
        self._tf_job = None
        if self.plan is None or self._identity():
            return
        if self.busy:
            self._tf_schedule(500)
            return
        self._tf_apply_profile()
        self.load()

    _view_size = 560

    def _canvas_resized(self, e):
        size = int(max(300, min(e.width, e.height) - 4))
        if abs(size - self._view_size) > 24:
            self._view_size = size
            self._debounce()

    def _xray_changed(self):
        self.show_preview()

    def _queue_preview(self):
        """Redraw soon, but not once per mouse event."""
        if self._fast_job is None:
            self._fast_job = self.after(12, self._run_preview)

    def _run_preview(self):
        self._fast_job = None
        self.show_preview()

    def _parts(self, pose):
        key = (id(self.plan), pose)
        if self._parts_cache is None or self._parts_cache[0] != key:
            self._parts_cache = (key, self.plan.textured_parts(pose))
        return self._parts_cache[1]

    def show_preview(self):
        if self.plan is None:
            if self._copy_info is not None:
                self._render_copy()
            return
        plan, pose = self.plan, self.v_pose.get()
        yaw, pitch = self.v_yaw.get(), self.v_pitch.get()
        try:
            import glview
            gl = glview.get() is not None
            parts, xform = None, None
            if self.v_shade.get() and gl:
                parts = self._parts(pose)                       # drawn as they are; the move / turn / scale is done on the GPU
                tris = np.concatenate([q['tris'] for q in parts]) if parts else np.zeros((0, 3, 3))
                cols = None
                xform = self.tf_A
            else:
                tris = plan.triangles(pose)
                cols = plan.triangle_colors() if self.v_shade.get() else None
                if gl:
                    xform = self.tf_A
                else:
                    tris = self._live(tris.reshape(-1, 3)).reshape(-1, 3, 3)
            xray = self.v_xray.get() and not pose
            orig = plan.original_triangles()
            ghost = orig if self.v_ghost.get() else None
            if self._ref is None:                               # the view is framed on the fitted model and the original, never on
                base = tris if gl else plan.triangles(pose)     # the part being moved, so nothing jumps while you drag
                self._ref = np.concatenate([base, orig])
            self._base, self._cam = preview.render_base(tris, yaw, pitch, self._view_size, self._ref, cols, xray=xray, ghost=ghost,
                                                         parts=parts, xform=xform)
        except Exception as e:
            self.log('preview failed: %s' % e)
            return
        self._pose_view = pose
        mode = self.v_mode.get()
        hint = {'move': 'Move: drag an arrow to slide along that axis, the square to slide in the view.',
                'rotate': 'Rotate: drag a ring to turn the model about that axis (red x, green y, blue z).',
                'scale': 'Scale: drag away from the centre to make it bigger, toward it to make it smaller.'}.get(mode, '')
        if xray:
            hint += ('  ' if hint else '') + 'X-ray: drag an orange dot to where that joint really is.'
        if not hint and not pose:
            hint = 'Drag the picture to turn the view. Move, Rotate and Scale fit your model to the original. Cyan: the game\'s bones; orange: your model\'s joints.'
        self.v_xhint.set(hint)
        self._tf_info()
        self._draw_skeleton()

    def _skeleton_parts(self, moved=None):
        """(overlays, handles) for the current view; moved = (joint name, world point) while a joint is being dragged."""
        plan = self.plan
        if self._pose_view or not (self.v_over.get() or self.v_xray.get()):
            return [], []
        cyan = plan.overlays()[0]
        handles, links = [], []
        for n, p, g in plan.joint_handles():
            p = self._live(np.asarray(p, np.float64)[None])[0]
            if moved and moved[0] == n:
                p = moved[1]
            handles.append((n, p, (255, 150, 60)))
            links.append((p, g))
        ov = [cyan, ('lines', links, (255, 150, 60))]
        return ov, (handles if self.v_xray.get() else [])

    def _draw_skeleton(self, moved=None, active=None, gizmo_active=None):
        if self._base is None:
            return
        ov, handles = self._skeleton_parts(moved)
        img, self._handles = preview.draw_skeleton(self._base, self._cam, ov, handles, active)
        mode = self.v_mode.get()
        self._giz = None
        if mode != 'off' and not self._pose_view:
            img, self._giz = preview.draw_gizmo(img, self._cam, self._pivot_now(), self._r_giz, mode, gizmo_active or self._giz_active)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.photo)

    # ---- the mouse: gizmo handles first, then joint handles, else the view turns
    def _nearest_handle(self, x, y):
        best, bd = None, 14.0
        for k, (hx, hy) in self._handles.items():
            d = ((hx - x) ** 2 + (hy - y) ** 2) ** 0.5
            if d < bd:
                best, bd = k, d
        return best

    def _gizmo_hit(self, x, y):
        g = self._giz
        if not g:
            return None
        best, bd = None, 13.0
        if self.v_mode.get() == 'rotate':
            for k, px in g['rings'].items():
                d = float(np.min(np.hypot(px[:, 0] - x, px[:, 1] - y)))
                if d < min(bd, 10.0):
                    best, bd = k, d
            return best
        for k, (hx, hy) in g['handles'].items():
            d = ((hx - x) ** 2 + (hy - y) ** 2) ** 0.5
            if d < bd:
                best, bd = k, d
        return best

    def _hover(self, e):
        if self._gdrag is None and self._drag is None and self._odrag is None:
            hot = self._gizmo_hit(e.x, e.y) or (self.v_xray.get() and self._nearest_handle(e.x, e.y))
            self.canvas.configure(cursor='hand2' if hot else 'fleur')

    def _press(self, e):
        if self.plan is None or self._base is None:
            return
        if self.v_mode.get() != 'off' and not self._pose_view:
            k = self._gizmo_hit(e.x, e.y)
            if k:
                c = self._pivot_now()
                cpx = np.array(self._giz['center'])
                d = {'k': k, 'mode': self.v_mode.get(), 'x': e.x, 'y': e.y, 'A0': self.tf_A.copy(), 'c': c, 'cpx': cpx}
                if d['mode'] == 'rotate':
                    # the ring is grabbed at a point; the turn follows the mouse along the ring's direction there (works however
                    # the ring is seen, even edge on)
                    px = self._giz['rings'][k]
                    i = int(np.argmin(np.hypot(px[:, 0] - e.x, px[:, 1] - e.y)))
                    n = len(px) - 1
                    T = px[(i + 1) % n] - px[(i - 1) % n]
                    step = 2 * (2 * np.pi / n)
                    ppr = max(float(np.hypot(*T)) / step, 0.35 * float(np.hypot(*(px[0] - cpx))) + 25.0)     # pixels per radian
                    d['T'] = T / (np.hypot(*T) + 1e-9)
                    d['ppr'] = ppr
                    d['sgn'] = -1.0 if k == 'y' else 1.0
                elif d['mode'] == 'scale':
                    d['r0'] = max(float(np.hypot(e.x - cpx[0], e.y - cpx[1])), 10.0)
                self._gdrag = d
                self._draw_skeleton(gizmo_active=k)
                return
        if self.v_xray.get() and not self._pose_view:
            k = self._nearest_handle(e.x, e.y)
            if k is not None:
                world = self._live(np.asarray(next(p for n, p, _g in self.plan.joint_handles() if n == k), np.float64)[None])[0]
                self._drag = {'joint': k, 'x': e.x, 'y': e.y, 'world': np.asarray(world, np.float64), 'now': np.asarray(world, np.float64)}
                self._draw_skeleton((k, self._drag['now']), k)
                return
        self._odrag = (e.x, e.y, self.v_yaw.get(), self.v_pitch.get())

    def _motion(self, e):
        cam = self._cam
        g = self._gdrag
        if g:
            c, A0 = g['c'], g['A0']
            if g['mode'] == 'rotate':
                axis = preview.AXES[g['k']][0]
                ang = g['sgn'] * float((e.x - g['x']) * g['T'][0] + (e.y - g['y']) * g['T'][1]) / g['ppr']
                self.tf_A = self._about(c, self._axis_turn(axis, ang)) @ A0
            elif g['mode'] == 'scale':
                f = float(np.hypot(e.x - g['cpx'][0], e.y - g['cpx'][1])) / g['r0']
                f = min(max(f, 0.05), 20.0)
                self.tf_A = self._about(c, np.eye(3) * f) @ A0
            else:
                dx, dy = e.x - g['x'], e.y - g['y']
                if g['k'] == 'c':
                    d = cam.R.T @ np.array([dx / cam.scale, -dy / cam.scale, 0.0])
                else:
                    axis = preview.AXES[g['k']][0]
                    v = (cam.R @ axis)[:2] * np.array([1.0, -1.0]) * cam.scale          # the axis in pixels per unit
                    n2 = float(v @ v)
                    d = axis * ((dx * v[0] + dy * v[1]) / n2) if n2 > 1e-6 else np.zeros(3)
                T = np.eye(4); T[:3, 3] = d
                self.tf_A = T @ A0
            self._tf_info()
            self._queue_preview_with(g['k'])
            return
        d = self._drag
        if d:
            q = np.array([(e.x - d['x']) / cam.scale, -(e.y - d['y']) / cam.scale, 0.0])
            d['now'] = d['world'] + cam.R.T @ q                      # the view plane, back in model space
            self._draw_skeleton((d['joint'], d['now']), d['joint'])
            return
        o = self._odrag
        if o:
            x0, y0, yaw0, pitch0 = o
            self.v_yaw.set(((yaw0 + (e.x - x0) * 0.6 + 180) % 360) - 180)
            self.v_pitch.set(max(-60.0, min(60.0, pitch0 - (e.y - y0) * 0.4)))
            self._queue_preview()

    def _queue_preview_with(self, active):
        self._giz_active = active
        self._queue_preview()

    def _release(self, e):
        g, self._gdrag = self._gdrag, None
        d, self._drag = self._drag, None
        self._odrag = None
        self._giz_active = None
        if g:
            self.show_preview()
            self._tf_schedule(350)                                # then the fit takes it over
            return
        if not d:
            return
        delta = d['now'] - d['world']
        if np.linalg.norm(delta) < 1e-4:
            self._draw_skeleton()
            return
        # the plan's fitted model, with what is live on it, back into the imported model's own units
        src = np.linalg.solve(self.tf_A[:3, :3] @ np.asarray(self.plan.M, np.float64)[:3, :3], delta)
        self._tf_apply_profile()
        offs = self.profile.setdefault('joint_offsets', {})
        offs[d['joint']] = [float(a + b) for a, b in zip(offs.get(d['joint'], (0, 0, 0)), src)]
        self.log('moved joint %s' % d['joint'])
        self.load()

    def reset_skeleton_edits(self):
        self.profile.pop('joint_offsets', None)
        self.load()

    # ---------------------------------------------------------------- bones tab
    def game_bone_names(self):
        return sorted(b.name for b in self.plan.f.bones if b.smooth >= 0)

    def bone_choices(self, _row):
        return ['(none)'] + self.game_bone_names()

    def _toggle_adv(self):
        if self.v_adv.get():
            self.adv.pack(fill='both', expand=True, pady=(4, 0))
        else:
            self.adv.pack_forget()

    def _body_rows(self):
        """[(iid, label, game bone)] for the body parts this skeleton has."""
        have = {b.name for b in self.plan.f.bones if b.smooth >= 0}
        rows = []
        for label, bone, sided in BODY:
            if sided:
                for side, word in (('L', 'Left'), ('R', 'Right')):
                    if '%s_%s' % (bone, side) in have:
                        rows.append(('%s_%s' % (bone, side), '%s %s' % (word, label.lower()), '%s_%s' % (bone, side)))
            elif bone in have:
                rows.append((bone, label, bone))
        if len(rows) < 8:
            # a creature or object rig (a fairy, an animal): list its own animated bones by name instead
            skip = re.compile(r'^(Root|Skl_Root|Ph_|Z_Ph_|LookIK|Z_|Dummy)', re.I)
            taken = {b for _i, _l, b in rows}
            for b in self.plan.f.bones:
                if b.smooth >= 0 and b.name not in taken and not skip.search(b.name):
                    rows.append((b.name, b.name.replace('_', ' '), b.name))
        return rows

    def fill_body(self):
        self.tv_body.delete(*self.tv_body.get_children())
        if self.plan is None:
            self.v_hint.set('Load a model first.')
            return
        if self.plan.rig is None:
            self.v_hint.set('This model has no skeleton, so its parts are matched to the game automatically from their shape. Nothing to do here.')
            return
        by_bone = {}
        for joint, g in self.plan.bone_map.items():
            if g:
                by_bone.setdefault(g, []).append(joint)
        manual = self.profile.get('bone_map') or {}
        rows = self._body_rows()
        found = 0
        for iid, label, bone in rows:
            js = by_bone.get(bone, [])
            if js:
                found += 1
                edited = any(j in manual for j in js)
                self.tv_body.insert('', 'end', iid=iid, values=(label, js[0], 'edited by you' if edited else 'matched'), tags=('ok',))
            else:
                self.tv_body.insert('', 'end', iid=iid, values=(label, '(not found)', 'check this'), tags=('miss',))
        miss = len(rows) - found
        if not rows:
            self.v_hint.set('This game model has no standard body parts to match. Use the advanced list below.')
        elif miss == 0:
            self.v_hint.set('All %d %s were matched automatically. If the preview looks right you can skip this tab and build.' % (len(rows), 'body part' if len(rows) == 1 else 'body parts'))
        else:
            self.v_hint.set('%d of %d body parts matched automatically. The %d marked "check this" had no joint with a recognisable name: '
                            'double-click each one and pick the part of your model that does that job, or leave it and the nearest joint is used.'
                            % (found, len(rows), miss))

    def edit_body(self, event):
        row = self.tv_body.identify_row(event.y)
        if not row or self.plan is None or self.plan.rig is None:
            return
        bone = next(b for i, _l, b in self._body_rows() if i == row)
        names = list(self.plan.rig.names)
        top = tk.Toplevel(self)
        top.title('Pick the joint')
        top.configure(background=BG)
        top.geometry('+%d+%d' % (event.x_root, event.y_root))
        ttk.Label(top, text='Which part of your model is the %s?' % self.tv_body.set(row, 'part').lower(), padding=10).pack(anchor='w')
        lb = tk.Listbox(top, height=min(18, len(names) + 1), width=44, background=FIELD, foreground=FG, selectbackground=ACCENT,
                        borderwidth=0, highlightthickness=1, highlightbackground=EDGE)
        for o in ['(nothing - leave it to the nearest joint)'] + names:
            lb.insert('end', o)
        lb.pack(padx=10, pady=(0, 10))
        def ok(_e=None):
            sel = lb.curselection()
            top.destroy()
            if not sel:
                return
            bm = self.profile.setdefault('bone_map', {})
            for j, g in list(self.plan.bone_map.items()):
                if g == bone:
                    bm[j] = None                                  # one joint per body part
            if sel[0] > 0:
                bm[names[sel[0] - 1]] = bone
            self.load()
        lb.bind('<Double-1>', ok); lb.bind('<Return>', ok)
        top.bind('<Escape>', lambda _e: top.destroy())
        lb.focus_set()

    def fill_bones(self):
        self.fill_body()
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
        if self.model_value().startswith('game:'):
            return self.log('this model is from the game: it has no image files to look for')
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
        if self.copy_mode():
            if self._copy_info is None:
                return self.load()
            self.save_profile()
            mod, romfs, target, model = self.mod_dir, self.vars['romfs'].get(), self.vars['target'].get(), self.model_value()

            def work_copy():
                import backup
                snap = backup.snapshot(os.path.dirname(mod), os.path.basename(mod), 'before a build')
                if snap:
                    self.log('backed up the existing mod first: %s' % snap)
                return swap.copy_model(romfs, target, model[5:], mod, self.log)
            self.run_bg(work_copy, lambda _r: (self.log('Built: %s  (restart the game to use it)' % mod), self.fill_backups()))
            return
        if self.plan is None:
            return self.load()
        self.collect()
        self.save_profile()
        mod = self.mod_dir
        tex = self.v_tex.get()
        romfs, target, model = self.vars['romfs'].get(), self.vars['target'].get(), self.model_value()

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
