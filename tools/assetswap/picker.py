"""Model picker: a popup that shows every model the tool can replace as a 3D thumbnail with its name, with a search box,
a category filter and a sort, and a larger preview you can turn with the mouse.

Thumbnails are drawn on a background thread from the game's own files (bind pose, flat shaded) and kept in a cache folder, so
the second opening is instant. Nothing here writes to the game folder.
"""
import json
import os
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

import gametex
import glview
import preview
from bfres import Bfres
from containers import Romfs
from skeleton import world_matrices

THUMB = 150
CACHE = os.path.join(tempfile.gettempdir(), 'assetswap_thumbs_v3')
CATEGORIES = ['All', 'Link', 'Characters', 'Enemies', 'Animals', 'Other']
SORTS = ['Name (A to Z)', 'Name (Z to A)', 'Tallest first', 'Smallest first', 'Most detailed first']


def category(rel):
    stem = os.path.basename(rel).split('.')[0]
    kind = stem.split('_')[0]
    return {'Player': 'Link', 'Npc': 'Characters', 'Enemy': 'Enemies', 'Horse': 'Animals', 'Animal': 'Animals'}.get(kind, 'Other')


def model_triangles(rom, rel, limit=22000):
    """(triangles (n, 3, 3) in the bind pose, facts) of a game model; triangles are thinned to about `limit` for drawing."""
    f = Bfres(rom.read(rel))
    W = world_matrices(f)
    tris = []
    total = 0
    for sh in f.shapes:
        if not sh.vertex_count:
            continue
        pos = f.read_attr(sh, '_p0')[:, :3].astype(np.float64)
        if sh.skin_count <= 1:
            h = np.c_[pos, np.ones(len(pos))]
            if sh.skin_count == 0 or not sh.skin_bones:
                pos = (h @ W[sh.bone].T)[:, :3]
            else:
                vb = f.read_attr(sh, '_i0')[:, 0].astype(int)
                q = np.empty_like(pos)
                for k in np.unique(vb):
                    sel = vb == k
                    q[sel] = (h[sel] @ W[f.matrix_to_bone[k]].T)[:, :3]
                pos = q
        idx = f.indices(sh).reshape(-1, 3)
        total += len(idx)
        tris.append(pos[idx])
    if not tris:
        return np.zeros((0, 3, 3)), {'tris': 0, 'height': 0.0, 'bones': len(f.bones), 'shapes': 0}
    t = np.concatenate(tris)
    pts = t.reshape(-1, 3)
    facts = {'tris': int(total), 'height': float(np.ptp(pts[:, 1])), 'bones': len(f.bones),
             'shapes': len([s for s in f.shapes if s.vertex_count])}
    if len(t) > limit:
        t = t[np.linspace(0, len(t) - 1, limit).astype(int)]
    return t, facts


class ModelPicker(tk.Toplevel):
    def __init__(self, parent, romfs, targets, friendly, theme, current=None, on_pick=None):
        super().__init__(parent)
        self.title('Choose the model to replace')
        self.geometry('1180x740')
        self.minsize(900, 560)
        self.configure(background=theme['BG'])
        self.transient(parent)
        self.th = theme
        self.blank = ImageTk.PhotoImage(Image.new('RGB', (THUMB, THUMB), (15, 17, 21)))
        self.blank_big = ImageTk.PhotoImage(Image.new('RGB', (440, 440), (15, 17, 21)))
        self.rom = Romfs(romfs)
        self.on_pick = on_pick
        self.items = [{'rel': r, 'name': friendly(r), 'file': os.path.basename(r).split('.')[0], 'cat': category(r)} for r in targets]
        self.by_rel = {i['rel']: i for i in self.items}
        self.thumbs, self.photos, self.meta = {}, {}, {}
        self.q = queue.Queue()
        self.todo, self.lock, self.closed = [], threading.Lock(), False
        self.selected = current if current in self.by_rel else None
        self.cards = {}
        self.view_tris = None
        self.view_yaw, self.view_pitch = 25.0, 8.0
        self.spin = True
        self._drag = None
        self._build()
        self.layout()
        for _ in range(min(4, max(2, (os.cpu_count() or 4) // 2))):          # several at once: each has its own OpenGL context
            threading.Thread(target=self._worker, daemon=True).start()
        self.after(60, self._poll)
        self.after(150, self._spin_tick)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.bind('<Escape>', lambda _e: self.close())
        if self.selected:
            self.select(self.selected)
        self.search_entry.focus_set()
        self.grab_set()

    # ---------------------------------------------------------------- layout
    def _build(self):
        th = self.th
        bar = ttk.Frame(self, padding=(16, 14, 16, 6)); bar.pack(fill='x')
        ttk.Label(bar, text='Search', style='Muted.TLabel').pack(side='left')
        self.v_search = tk.StringVar()
        self.search_entry = ttk.Entry(bar, textvariable=self.v_search, width=28)
        self.search_entry.pack(side='left', padx=(8, 18))
        self.v_search.trace_add('write', lambda *_: self._later())
        ttk.Label(bar, text='Show', style='Muted.TLabel').pack(side='left')
        self.v_cat = tk.StringVar(value='All')
        cb = ttk.Combobox(bar, textvariable=self.v_cat, values=CATEGORIES, state='readonly', width=12)
        cb.pack(side='left', padx=(8, 18)); cb.bind('<<ComboboxSelected>>', lambda _e: self.layout())
        ttk.Label(bar, text='Sort', style='Muted.TLabel').pack(side='left')
        self.v_sort = tk.StringVar(value=SORTS[0])
        cs = ttk.Combobox(bar, textvariable=self.v_sort, values=SORTS, state='readonly', width=20)
        cs.pack(side='left', padx=8); cs.bind('<<ComboboxSelected>>', lambda _e: self.layout())
        self.v_count = tk.StringVar()
        ttk.Label(bar, textvariable=self.v_count, style='Muted.TLabel').pack(side='right')

        body = ttk.Frame(self, padding=(16, 4, 16, 0)); body.pack(fill='both', expand=True)
        left = ttk.Frame(body); left.pack(side='left', fill='both', expand=True)
        self.canvas = tk.Canvas(left, background=th['PANEL'], highlightthickness=0, borderwidth=0)
        sb = ttk.Scrollbar(left, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y'); self.canvas.pack(side='left', fill='both', expand=True)
        self.inner = tk.Frame(self.canvas, background=th['PANEL'])
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.inner.bind('<Configure>', lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda e: self._resized(e.width))
        self.canvas.bind_all('<MouseWheel>', self._wheel)

        right = ttk.Frame(body, padding=(14, 0, 0, 0)); right.pack(side='right', fill='y')
        self.big = tk.Label(right, background='#0f1115', borderwidth=0, image=self.blank_big)
        self.big.pack()
        for ev, fn in (('<ButtonPress-1>', self._press), ('<B1-Motion>', self._motion), ('<ButtonRelease-1>', self._release)):
            self.big.bind(ev, fn)
        self.v_name = tk.StringVar(value='Pick a model on the left')
        ttk.Label(right, textvariable=self.v_name, style='Title.TLabel', wraplength=400).pack(anchor='w', pady=(10, 0))
        self.v_facts = tk.StringVar()
        ttk.Label(right, textvariable=self.v_facts, style='Muted.TLabel', wraplength=400, justify='left').pack(anchor='w', pady=(4, 0))
        self.v_spin = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text='Turn slowly', variable=self.v_spin, command=lambda: setattr(self, 'spin', self.v_spin.get())).pack(anchor='w', pady=(8, 0))
        ttk.Label(right, text='Drag the picture to turn the model.', style='Muted.TLabel').pack(anchor='w')
        self.btn = ttk.Button(right, text='Use this model', style='Accent.TButton', command=self.choose, state='disabled')
        self.btn.pack(fill='x', pady=(14, 0))
        self.v_prog = tk.StringVar()
        ttk.Label(self, textvariable=self.v_prog, style='Muted.TLabel', padding=(16, 6, 16, 10)).pack(anchor='w')

    def _wheel(self, e):
        if self.winfo_exists():
            self.canvas.yview_scroll(-1 * (e.delta // 120), 'units')

    _later_job = None

    def _later(self):
        if self._later_job:
            self.after_cancel(self._later_job)
        self._later_job = self.after(180, self.layout)

    def _resized(self, w):
        self.canvas.itemconfigure(self.win, width=w)
        cols = max(1, w // (THUMB + 28))
        if cols != getattr(self, '_cols', None):
            self.layout()

    # ---------------------------------------------------------------- the grid
    def visible(self):
        text = self.v_search.get().strip().lower()
        cat = self.v_cat.get()
        out = [i for i in self.items if (cat == 'All' or i['cat'] == cat) and (not text or text in i['name'].lower() or text in i['file'].lower())]
        sort = self.v_sort.get()
        meta = lambda i, k, d=0.0: self.meta.get(i['rel'], {}).get(k, d)
        if sort == SORTS[0]:
            out.sort(key=lambda i: i['name'].lower())
        elif sort == SORTS[1]:
            out.sort(key=lambda i: i['name'].lower(), reverse=True)
        elif sort == SORTS[2]:
            out.sort(key=lambda i: -meta(i, 'height'))
        elif sort == SORTS[3]:
            out.sort(key=lambda i: (i['rel'] not in self.meta, meta(i, 'height')))
        else:
            out.sort(key=lambda i: -meta(i, 'tris'))
        return out

    def layout(self):
        th = self.th
        w = max(self.canvas.winfo_width(), 600)
        self._cols = cols = max(1, w // (THUMB + 28))
        for c in self.inner.winfo_children():
            c.destroy()
        self.cards = {}
        vis = self.visible()
        for n, it in enumerate(vis):
            card = tk.Frame(self.inner, background=th['FIELD'], highlightthickness=2, highlightbackground=th['PANEL'], cursor='hand2')
            card.grid(row=n // cols, column=n % cols, padx=6, pady=6)
            img = tk.Label(card, background='#0f1115', image=self._photo(it['rel']), borderwidth=0)
            img.pack(padx=4, pady=(4, 0))
            nm = tk.Label(card, text=it['name'], background=th['FIELD'], foreground=th['FG'], wraplength=THUMB, font=('Segoe UI', 9))
            nm.pack(padx=4, pady=(4, 0))
            tag = tk.Label(card, text=it['cat'], background=th['FIELD'], foreground=th['MUTED'], font=('Segoe UI', 8))
            tag.pack(pady=(0, 4))
            for wdg in (card, img, nm, tag):
                wdg.bind('<Button-1>', lambda _e, r=it['rel']: self.select(r))
                wdg.bind('<Double-1>', lambda _e, r=it['rel']: (self.select(r), self.choose()))
            self.cards[it['rel']] = (card, img)
        self.v_count.set('%d of %d models' % (len(vis), len(self.items)))
        with self.lock:                                   # draw what is on screen first
            seen = {i['rel'] for i in vis}
            self.todo = [i['rel'] for i in vis if i['rel'] not in self.thumbs] + [i['rel'] for i in self.items if i['rel'] not in seen and i['rel'] not in self.thumbs]
        self._mark()
        self.canvas.yview_moveto(0)

    def _photo(self, rel):
        if rel in self.thumbs:
            if rel not in self.photos:
                self.photos[rel] = ImageTk.PhotoImage(self.thumbs[rel])
            return self.photos[rel]
        return self.blank

    def _mark(self):
        for rel, (card, _img) in self.cards.items():
            card.configure(highlightbackground=self.th['ACCENT'] if rel == self.selected else self.th['PANEL'])

    # ---------------------------------------------------------------- thumbnails (background thread)
    def _cache_key(self, rel):
        p = self.rom.path(rel)
        st = os.stat(p)
        return os.path.join(CACHE, '%s_%d_%d' % (os.path.basename(rel).split('.')[0], st.st_size, int(st.st_mtime)))

    def _worker(self):
        os.makedirs(CACHE, exist_ok=True)
        while not self.closed:
            with self.lock:
                rel = self.todo.pop(0) if self.todo else None
            if rel is None:
                threading.Event().wait(0.25)
                continue
            try:
                key = self._cache_key(rel)
                if os.path.exists(key + '.png') and os.path.exists(key + '.json'):
                    img, facts = Image.open(key + '.png').convert('RGB'), json.load(open(key + '.json'))
                else:
                    img, facts = self._thumb(rel)
                    img.save(key + '.png'); json.dump(facts, open(key + '.json', 'w'))
                self.q.put((rel, img, facts))
            except Exception as e:                       # a model this tool cannot read: a blank tile, not a broken window
                self.q.put((rel, Image.new('RGB', (THUMB, THUMB), (40, 30, 30)), {'tris': 0, 'height': 0.0, 'bones': 0, 'shapes': 0, 'error': str(e)[:80]}))

    def _thumb(self, rel):
        """(image, facts): the model with its own textures through OpenGL; flat grey on the CPU when OpenGL is missing."""
        gl = glview.get()
        if gl is not None:
            parts, facts = gametex.textured_parts(self.rom, rel, 128)
            if not facts['tris']:
                return Image.new('RGB', (THUMB, THUMB), (15, 17, 21)), facts
            img, _cam = gl.render(None, 25.0, 8.0, THUMB, parts=parts)
            bg = Image.new('RGB', (THUMB, THUMB), (15, 17, 21))
            bg.paste(img, ((THUMB - img.width) // 2, (THUMB - img.height) // 2))
            return bg, facts
        tris, facts = model_triangles(self.rom, rel)
        return (self._draw(tris, THUMB) if len(tris) else Image.new('RGB', (THUMB, THUMB), (15, 17, 21))), facts

    @staticmethod
    def _draw(tris, size, yaw=25.0, pitch=8.0):
        img = preview.render_image(tris, yaw, pitch, size)
        bg = Image.new('RGB', (size, size), (15, 17, 21))
        bg.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        return bg

    def _poll(self):
        if not self.winfo_exists():
            return
        n = 0
        try:
            while n < 12:
                rel, img, facts = self.q.get_nowait()
                self.thumbs[rel], self.meta[rel] = img, facts
                if rel in self.cards:
                    self.cards[rel][1].configure(image=self._photo(rel))
                n += 1
        except queue.Empty:
            pass
        done = len(self.thumbs)
        self.v_prog.set('' if done >= len(self.items) else 'Drawing previews: %d of %d' % (done, len(self.items)))
        if done >= len(self.items) and not getattr(self, '_sorted_once', False):
            self._sorted_once = True
            if self.v_sort.get() not in SORTS[:2]:
                self.layout()                              # the size sorts need every model measured
        if self.selected and self.selected in self.meta:
            self._facts(self.selected)
        self.after(80, self._poll)

    # ---------------------------------------------------------------- selection and the big view
    def select(self, rel):
        self.selected = rel
        it = self.by_rel[rel]
        self.v_name.set(it['name'])
        self.btn.configure(state='normal')
        self._facts(rel)
        self._mark()
        self.view_tris = None
        threading.Thread(target=self._load_big, args=(rel,), daemon=True).start()

    def _facts(self, rel):
        m = self.meta.get(rel)
        it = self.by_rel[rel]
        if m:
            self.v_facts.set('%s\nFile: %s\nHeight about %.2f m, %s triangles, %d bones, %d parts'
                             % (it['cat'], it['file'], m['height'], format(m['tris'], ','), m['bones'], m['shapes']))
        else:
            self.v_facts.set('%s\nFile: %s\nMeasuring...' % (it['cat'], it['file']))

    def _load_big(self, rel):
        try:
            if glview.moderngl is not None:
                parts, _ = gametex.textured_parts(self.rom, rel, 512)
                tris = parts
            else:
                tris, _ = model_triangles(self.rom, rel, limit=12000)
        except Exception:
            tris = []
        if rel == self.selected:
            self.view_tris = tris

    def _render_big(self):
        if self.view_tris is None or not len(self.view_tris):
            return
        s = 440
        gl = glview.get() if isinstance(self.view_tris, list) else None
        if gl is not None:
            img, _cam = gl.render(None, self.view_yaw, self.view_pitch, s, parts=self.view_tris)
            bg = Image.new('RGB', (s, s), (15, 17, 21))
            bg.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
            img = bg
        else:
            tris = self.view_tris if not isinstance(self.view_tris, list) else np.concatenate([q['tris'] for q in self.view_tris])
            img = self._draw(tris, s, self.view_yaw, self.view_pitch)
        self._big_photo = ImageTk.PhotoImage(img)
        self.big.configure(image=self._big_photo)

    def _spin_tick(self):
        if not self.winfo_exists():
            return
        if self.view_tris is not None and self._drag is None:
            if self.spin:
                self.view_yaw = (self.view_yaw + 5.0) % 360
            if self.spin or getattr(self, '_fresh', None) is not self.view_tris:
                self._fresh = self.view_tris
                self._render_big()
        self.after(110, self._spin_tick)

    def _press(self, e):
        self._drag = (e.x, e.y, self.view_yaw, self.view_pitch)

    def _motion(self, e):
        if self._drag:
            x0, y0, yaw0, pitch0 = self._drag
            self.view_yaw = (yaw0 + (e.x - x0) * 0.8) % 360
            self.view_pitch = max(-70.0, min(70.0, pitch0 - (e.y - y0) * 0.5))
            self._render_big()

    def _release(self, _e):
        self._drag = None

    # ---------------------------------------------------------------- done
    def choose(self):
        if self.selected and self.on_pick:
            self.on_pick(self.selected)
        self.close()

    def close(self):
        self.closed = True
        try:
            self.canvas.unbind_all('<MouseWheel>')
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
