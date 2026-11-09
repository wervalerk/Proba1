#!/usr/bin/env python3
"""
Telegram-бот: точки из TXT → DXF + PDF.

Поток:
  TXT → подтверждение → размеры креста/текста → превью
      → [создать без подложки] | [подложка: из памяти (5 слотов) / новая]
      → Модуль 0: проверка СК (X↔Y, X0 Y0) → превью с подложкой
      → DXF + PDF → [сохранить подложку в слот] | [новый файл]

Подложка НЕ сохраняется в память автоматически — только по кнопке.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # ОБЯЗАТЕЛЬНО до любых импортов pyplot / ezdxf.addons.drawing

import asyncio
import html
import json
import logging
import math
import os
import re
import shutil
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import ezdxf
import matplotlib.pyplot as plt
from ezdxf import bbox, units
from ezdxf.addons.drawing.matplotlib import qsave
from ezdxf.path import make_path
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes, ConversationHandler, MessageHandler,
                          filters)

try:
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree
    HAS_SHAPELY = True
except Exception:  # pragma: no cover
    HAS_SHAPELY = False

# ================== НАСТРОЙКИ ==================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise SystemExit("Задайте переменную окружения BOT_TOKEN")

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "bot_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_SLOTS = 5
NEAR_M = 5.0        # «рядом»: медиана до ближайшего контура, м
SWAP_RATIO = 3.0    # во сколько раз X↔Y должна улучшить, чтобы предложить
MAX_CS_POINTS = 3000  # для медианы хватит выборки
# ==============================================

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("txt2dxf")

(WAITING_CONFIRM, WAITING_SIZES, WAITING_CUSTOM_SIZE, WAITING_PREVIEW,
 WAITING_UNDERLAY, WAITING_DXF_UPLOAD, WAITING_UNITS, WAITING_CS,
 WAITING_SHIFT, WAITING_NEW_FILE, WAITING_SAVE_SLOT, WAITING_DONE) = range(12)

Point4 = Tuple[Any, float, float, float]


# ═══════════════════════════════════════════════════════════════
#  Файлы пользователя и память подложек (на диске → переживает рестарт)
# ═══════════════════════════════════════════════════════════════

def user_dir(uid: int) -> Path:
    p = WORK_DIR / f"user_{uid}"
    (p / "in").mkdir(parents=True, exist_ok=True)
    (p / "out").mkdir(parents=True, exist_ok=True)
    (p / "underlays").mkdir(parents=True, exist_ok=True)
    return p


def safe_name(name: str, ext: str) -> str:
    stem = Path(name or "file").stem
    stem = re.sub(r"[^A-Za-z0-9А-Яа-я_.-]+", "_", stem).strip("._") or "file"
    return f"{stem[:60]}{ext}"


def load_memory(uid: int) -> Dict[int, dict]:
    p = user_dir(uid) / "memory.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text("utf-8"))
        mem = {int(k): v for k, v in raw.items()}
        # выкидываем слоты, у которых пропал файл
        return {k: v for k, v in mem.items() if Path(v.get("file", "")).exists()}
    except Exception as e:
        logger.warning("memory.json broken: %s", e)
        return {}


def save_memory(uid: int, mem: Dict[int, dict]) -> None:
    p = user_dir(uid) / "memory.json"
    p.write_text(json.dumps({str(k): v for k, v in mem.items()}, ensure_ascii=False, indent=1), "utf-8")


def reset_session(context: ContextTypes.DEFAULT_TYPE, uid: int) -> None:
    context.user_data.clear()
    context.user_data["_uid"] = uid
    context.user_data["swap_xy"] = False
    context.user_data["shift"] = (0.0, 0.0)


# ═══════════════════════════════════════════════════════════════
#  Парсинг TXT
# ═══════════════════════════════════════════════════════════════

def decode_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1251", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_points_from_txt(text: str) -> Tuple[List[Point4], int]:
    """Возвращает (points, skipped_lines). Решение «первый столбец — ID» принимается на уровне файла."""
    rows: List[List[float]] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.replace(";", " ").replace("\t", " ")
        s = re.sub(r"(?<=\d),(?=\d)", ".", s) if re.search(r"\s", s) else s.replace(",", " ")
        tokens = s.split()
        nums = []
        bad = False
        for t in tokens:
            if _NUM.fullmatch(t):
                nums.append(float(t))
            else:
                bad = True
                break
        if bad or len(nums) < 2:
            skipped += 1
            continue
        rows.append(nums)
    if not rows:
        return [], skipped

    first_is_id = (
        all(len(r) >= 3 for r in rows)
        and all(r[0].is_integer() and 0 <= r[0] < 1e6 for r in rows)
        and len({int(r[0]) for r in rows}) == len(rows)
    )
    points: List[Point4] = []
    for i, r in enumerate(rows, 1):
        if first_is_id:
            pid, x, y = int(r[0]), r[1], r[2]
            z = r[3] if len(r) > 3 else 0.0
        else:
            pid, x, y = i, r[0], r[1]
            z = r[2] if len(r) > 2 else 0.0
        points.append((pid, x, y, z))
    return points, skipped


def effective_points(context: ContextTypes.DEFAULT_TYPE) -> List[Point4]:
    """Единственный источник правды: точки с учётом X↔Y — и для проверки, и для превью, и для DXF."""
    pts = context.user_data["points"]
    if context.user_data.get("swap_xy"):
        return [(pid, y, x, z) for pid, x, y, z in pts]
    return pts


# ═══════════════════════════════════════════════════════════════
#  Геометрия DXF
# ═══════════════════════════════════════════════════════════════

@dataclass
class Seg:
    ax: float; ay: float; az: float
    bx: float; by: float; bz: float
    layer: str = "0"


@dataclass
class DxfGeom:
    polylines: List[Tuple[List[Tuple[float, float, float]], bool, str]]
    segments: List[Seg]
    scale_note: str
    has_z: bool


SUPPORTED = {"LWPOLYLINE", "POLYLINE", "LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}
UNIT_TABLE = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 7: 1000.0, 10: 0.9144}
UNIT_OVERRIDE = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def walk(entities, depth: int = 0):
    for e in entities:
        if e.dxftype() == "INSERT" and depth < 3:
            try:
                yield from walk(e.virtual_entities(), depth + 1)
            except Exception as ex:
                logger.warning("INSERT skipped: %s", ex)
        else:
            yield e


def manual_pts(e):
    t = e.dxftype()
    if t == "LWPOLYLINE":
        z = float(e.dxf.get("elevation", 0.0) or 0.0)
        return [(p[0], p[1], z) for p in e.get_points("xy")], bool(e.closed)
    if t == "POLYLINE":
        return [(v.dxf.location.x, v.dxf.location.y, v.dxf.location.z) for v in e.vertices], bool(e.is_closed)
    if t == "LINE":
        s, q = e.dxf.start, e.dxf.end
        return [(s.x, s.y, s.z), (q.x, q.y, q.z)], False
    if t in ("ARC", "CIRCLE"):
        c, r = e.dxf.center, e.dxf.radius
        a0, a1 = (e.dxf.start_angle, e.dxf.end_angle) if t == "ARC" else (0.0, 360.0)
        if a1 <= a0:
            a1 += 360.0
        n = max(16, int((a1 - a0) / 5) + 1)
        return [(c.x + r * math.cos(math.radians(a0 + (a1 - a0) * i / (n - 1))),
                 c.y + r * math.sin(math.radians(a0 + (a1 - a0) * i / (n - 1))), c.z)
                for i in range(n)], t == "CIRCLE"
    return [], False


def raw_path(e) -> Optional[dict]:
    if e.dxftype() not in SUPPORTED:
        return None
    try:
        ext = bbox.extents([e])
        size = max(ext.extmax.x - ext.extmin.x, ext.extmax.y - ext.extmin.y)
    except Exception:
        size = 1.0
    flat = max(size * 5e-4, 1e-6)
    try:
        p = make_path(e)
        pts = [(v.x, v.y, v.z) for v in p.flattening(flat)]
        closed = bool(p.is_closed)
    except Exception:
        pts, closed = manual_pts(e)
    closed = closed or bool(getattr(e, "closed", False)) or bool(getattr(e, "is_closed", False))
    if len(pts) < 2:
        return None
    if closed and math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) > 1e-9:
        pts.append(pts[0])
    return {"pts": pts, "closed": closed, "layer": e.dxf.layer}


def detect_units(path: Path) -> Tuple[Optional[str], float]:
    """(override-код если $INSUNITS известен, иначе None; максимальная |координата| для подсказки)."""
    doc = ezdxf.readfile(str(path))
    u = int(doc.header.get("$INSUNITS", 0) or 0)
    code = {4: "mm", 5: "cm", 6: "m"}.get(u)
    if u in UNIT_TABLE and code is None:
        code = f"insunits{u}"
    maxabs = 0.0
    try:
        ext = bbox.extents(doc.modelspace())
        maxabs = max(abs(ext.extmin.x), abs(ext.extmin.y), abs(ext.extmax.x), abs(ext.extmax.y))
    except Exception:
        pass
    return code, maxabs


def parse_dxf_geom(path: Path, shift: Tuple[float, float] = (0.0, 0.0), unit_code: Optional[str] = None) -> DxfGeom:
    doc = ezdxf.readfile(str(path))
    raw = [r for r in (raw_path(e) for e in walk(doc.modelspace())) if r]
    if not raw:
        raise ValueError("В DXF не найдено поддерживаемой геометрии (полилинии, линии, дуги, окружности, сплайны).")
    if unit_code in UNIT_OVERRIDE:
        scale, note = UNIT_OVERRIDE[unit_code], f"единицы: {unit_code}"
    else:
        u = int(doc.header.get("$INSUNITS", 0) or 0)
        scale, note = UNIT_TABLE.get(u, 1.0), f"$INSUNITS={u}"
    dx, dy = shift
    polylines, segments, zs = [], [], []
    for r in raw:
        pts_m = [(x * scale + dx, y * scale + dy, z * scale) for x, y, z in r["pts"]]
        zs.append(sum(p[2] for p in pts_m) / len(pts_m))
        polylines.append((pts_m, r["closed"], r["layer"]))
        for a, b in zip(pts_m, pts_m[1:]):
            if (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 < 1e-18:
                continue
            segments.append(Seg(*a, *b, r["layer"]))
    if not segments:
        raise ValueError("После приведения единиц не осталось отрезков.")
    has_z = len({round(z, 3) for z in zs}) > 1 or any(abs(z) > 1e-9 for z in zs)
    return DxfGeom(polylines, segments, note, has_z)


# ═══════════════════════════════════════════════════════════════
#  Модуль 0 — проверка СК
# ═══════════════════════════════════════════════════════════════

@dataclass
class CSReport:
    direct: float
    swapped: float
    c_txt: Tuple[float, float]
    c_dxf: Tuple[float, float]
    ok: bool
    suggest_swap: bool


def _seg_dist(px, py, s: Seg) -> float:
    vx, vy = s.bx - s.ax, s.by - s.ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-18:
        return math.hypot(px - s.ax, py - s.ay)
    t = max(0.0, min(1.0, ((px - s.ax) * vx + (py - s.ay) * vy) / L2))
    return math.hypot(px - (s.ax + t * vx), py - (s.ay + t * vy))


def _median_nearest(points: List[Tuple[float, float]], segments: List[Seg]) -> float:
    if not points or not segments:
        return float("inf")
    if len(points) > MAX_CS_POINTS:
        step = len(points) / MAX_CS_POINTS
        points = [points[int(i * step)] for i in range(MAX_CS_POINTS)]
    if HAS_SHAPELY:
        lines = [LineString([(s.ax, s.ay), (s.bx, s.by)]) for s in segments]
        tree = STRtree(lines)
        return statistics.median(lines[tree.nearest(Point(x, y))].distance(Point(x, y)) for x, y in points)
    return statistics.median(min(_seg_dist(x, y, s) for s in segments) for x, y in points)


def _centroid(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not pts:
        return 0.0, 0.0
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def check_cs(points: List[Point4], geom: DxfGeom) -> CSReport:
    """points — уже «эффективные» (с учётом текущего swap). geom — уже со сдвигом."""
    direct = [(x, y) for _, x, y, _ in points]
    swapped = [(y, x) for _, x, y, _ in points]
    d0 = _median_nearest(direct, geom.segments)
    d1 = _median_nearest(swapped, geom.segments)
    ok = d0 <= NEAR_M
    suggest = (not ok) and d1 < d0 and (d1 <= NEAR_M or d1 * SWAP_RATIO <= d0)
    dxf_pts = [(s.ax, s.ay) for s in geom.segments]
    return CSReport(d0, d1, _centroid(direct), _centroid(dxf_pts), ok, suggest)


def fmt(v: float) -> str:
    return "∞" if not math.isfinite(v) else f"{v:,.1f}".replace(",", " ")


def cs_text(rep: CSReport, shift: Tuple[float, float], swapped: bool, note: str) -> str:
    lines = [
        "🔎 <b>Модуль 0 · проверка системы координат</b>",
        f"Подложка: {html.escape(note)}; сдвиг X0 Y0 = <code>{shift[0]:,.3f}; {shift[1]:,.3f}</code> м",
        f"Ориентация TXT: <b>{'X↔Y переставлены' if swapped else 'исходная (X→X, Y→Y)'}</b>",
        f"• медиана до контура как сейчас: <b>{fmt(rep.direct)} м</b>",
        f"• если поменять X↔Y: <b>{fmt(rep.swapped)} м</b>",
        f"Центр точек <code>{fmt(rep.c_txt[0])}; {fmt(rep.c_txt[1])}</code> ↔ "
        f"центр DXF <code>{fmt(rep.c_dxf[0])}; {fmt(rep.c_dxf[1])}</code>",
    ]
    if rep.ok:
        lines.append(f"\n✅ СК совпадает (порог {NEAR_M:.0f} м). Можно создавать файлы.")
    elif rep.suggest_swap:
        g = rep.direct / max(rep.swapped, 1e-6)
        lines.append(f"\n⚠️ Похоже, оси переставлены: инверсия лучше в {g:.0f}×. Нажмите «Поменять X↔Y».")
    else:
        dx, dy = rep.c_dxf[0] - rep.c_txt[0], rep.c_dxf[1] - rep.c_txt[1]
        lines.append(
            f"\n⚠️ Точки далеко от контура. Нужен сдвиг подложки X0 Y0 (в метрах, прибавляется к DXF).\n"
            f"Подсказка по центрам (не применяю автоматически): <code>{-dx:,.3f} {-dy:,.3f}</code>\n"
            f"Можно просто прислать два числа сообщением."
        )
    return "\n".join(lines)


def cs_keyboard(rep: CSReport, swapped: bool) -> InlineKeyboardMarkup:
    create_label = "✅ Создать DXF + PDF" if rep.ok else "⚠️ Создать как есть"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("↩️ Вернуть X/Y" if swapped else "🔄 Поменять X↔Y", callback_data="cs_swap"),
         InlineKeyboardButton("✏️ Ввести X0 Y0", callback_data="cs_shift")],
        [InlineKeyboardButton("🔁 Перепроверить", callback_data="cs_recheck"),
         InlineKeyboardButton("📂 Другая подложка", callback_data="cs_other")],
        [InlineKeyboardButton(create_label, callback_data="cs_create")],
        [InlineKeyboardButton("📄 Новый файл (сброс)", callback_data="new_file")],
    ])


# ═══════════════════════════════════════════════════════════════
#  Размеры, превью, DXF, PDF
# ═══════════════════════════════════════════════════════════════

def calculate_auto_sizes(points: List[Point4]) -> Tuple[float, float]:
    if len(points) < 2:
        return 0.7, 0.9
    xy = sorted((x, y) for _, x, y, _ in points)
    sample = xy[:2000]
    min_d = float("inf")
    for i, (x1, y1) in enumerate(sample):
        for x2, y2 in sample[i + 1:i + 40]:  # после сортировки по X соседи — рядом
            d = math.hypot(x1 - x2, y1 - y2)
            if 0.01 < d < min_d:
                min_d = d
    if not math.isfinite(min_d):
        min_d = 2.0
    cross = max(min(min_d * 0.28, 2.0), 0.35)
    text = max(min(min_d * 0.38, 2.5), 0.5)
    return round(cross, 2), round(text, 2)


def create_preview_image(points: List[Point4], cross: float, text_h: float, out: Path,
                         underlay: Optional[DxfGeom] = None, title_extra: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=130)
    if underlay:
        for pts, _, _ in underlay.polylines:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#7d7d7d", lw=0.7, zorder=1)
    half = cross / 2
    off = text_h * 0.65
    for pid, x, y, _ in points:
        ax.plot([x - half, x + half], [y, y], color="red", lw=1.1, zorder=3)
        ax.plot([x, x], [y - half, y + half], color="red", lw=1.1, zorder=3)
        ax.plot(x, y, "o", color="red", ms=2.5, zorder=4)
        ax.text(x + off, y + off, str(pid), fontsize=7, color="#1a7a1a", fontweight="bold", zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls="--", alpha=0.3)
    t = f"Превью | крест {cross} | текст {text_h} | точек {len(points)}"
    ax.set_title(t + (f" | {title_extra}" if title_extra else ""), fontsize=10)
    ax.margins(0.06)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_dxf(dxf_path: Path, points: List[Point4], cross: float, text_h: float,
              underlay: Optional[DxfGeom]) -> None:
    doc = ezdxf.new(dxfversion="R2010")
    doc.units = units.M
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    for name, color in (("PROJ", 8), ("POINTS", 1), ("TEXT", 3)):
        if name not in doc.layers:
            doc.layers.add(name, color=color)

    if underlay:
        for pts, closed, _ in underlay.polylines:
            if underlay.has_z:
                msp.add_polyline3d(pts, close=closed, dxfattribs={"layer": "PROJ"})
            else:
                xy = [(p[0], p[1]) for p in pts]
                if closed and len(xy) > 2 and math.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1]) < 1e-9:
                    xy = xy[:-1]
                msp.add_lwpolyline(xy, close=closed, dxfattribs={"layer": "PROJ"})

    blk = doc.blocks.new(name="POINT_WITH_ID")
    r = max(cross * 0.13, 0.04)
    off = text_h * 0.65
    blk.add_line((-cross / 2, 0), (cross / 2, 0), dxfattribs={"layer": "POINTS"})
    blk.add_line((0, -cross / 2), (0, cross / 2), dxfattribs={"layer": "POINTS"})
    blk.add_circle((0, 0), radius=r, dxfattribs={"layer": "POINTS"})
    blk.add_attdef(tag="ID", insert=(off, off), dxfattribs={"layer": "TEXT", "height": text_h})

    for pid, x, y, z in points:
        ref = msp.add_blockref("POINT_WITH_ID", (x, y, z), dxfattribs={"layer": "POINTS"})
        ref.add_auto_attribs({"ID": str(pid)})
    doc.saveas(str(dxf_path))


def write_pdf(dxf_path: Path, pdf_path: Path) -> bool:
    try:
        doc = ezdxf.readfile(str(dxf_path))
        qsave(doc.modelspace(), str(pdf_path), dpi=200, bg="#FFFFFF", fg="#000000")
        return True
    except Exception as e:
        logger.error("PDF error: %s", e)
        return False


def build_outputs(uid: int, points, cross, text_h, underlay) -> Tuple[Path, Optional[Path]]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = user_dir(uid) / "out"
    dxf = out / f"points_{stamp}.dxf"
    pdf = out / f"points_{stamp}.pdf"
    write_dxf(dxf, points, cross, text_h, underlay)
    return dxf, (pdf if write_pdf(dxf, pdf) else None)


# ═══════════════════════════════════════════════════════════════
#  Клавиатуры / утилиты
# ═══════════════════════════════════════════════════════════════

def sizes_keyboard(ac: float, at: float) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🤖 Авто ({ac} / {at})", callback_data="size_auto")],
        [InlineKeyboardButton("0.4 / 0.6", callback_data="size_0.4_0.6"),
         InlineKeyboardButton("0.6 / 0.8", callback_data="size_0.6_0.8")],
        [InlineKeyboardButton("0.8 / 1.0", callback_data="size_0.8_1.0"),
         InlineKeyboardButton("1.2 / 1.5", callback_data="size_1.2_1.5")],
        [InlineKeyboardButton("✏️ Свои значения", callback_data="size_custom")],
    ])


def preview_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Создать DXF + PDF (без подложки)", callback_data="create_plain")],
        [InlineKeyboardButton("📂 Добавить подложку DXF", callback_data="add_underlay")],
        [InlineKeyboardButton("🔄 Изменить размеры", callback_data="change_sizes")],
        [InlineKeyboardButton("📄 Новый файл (сброс)", callback_data="new_file")],
    ])


def memory_keyboard(uid: int) -> InlineKeyboardMarkup:
    mem = load_memory(uid)
    rows = [[InlineKeyboardButton(f"💾 {i}: {mem[i]['name']}", callback_data=f"mem_load_{i}")]
            for i in range(1, MEMORY_SLOTS + 1) if i in mem]
    rows.append([InlineKeyboardButton("📤 Загрузить новый DXF", callback_data="mem_upload_new")])
    rows.append([InlineKeyboardButton("⬅️ Назад к превью", callback_data="mem_cancel")])
    return InlineKeyboardMarkup(rows)


def save_slot_keyboard(uid: int) -> InlineKeyboardMarkup:
    mem = load_memory(uid)
    rows = []
    for i in range(1, MEMORY_SLOTS + 1):
        label = f"{i}: {mem[i]['name']} (заменить)" if i in mem else f"{i}: — свободно —"
        rows.append([InlineKeyboardButton(label, callback_data=f"slot_{i}")])
    rows.append([InlineKeyboardButton("❌ Не сохранять", callback_data="slot_cancel")])
    return InlineKeyboardMarkup(rows)


def done_keyboard(has_underlay: bool) -> InlineKeyboardMarkup:
    rows = []
    if has_underlay:
        rows.append([InlineKeyboardButton("💾 Сохранить подложку в память", callback_data="save_underlay")])
    rows.append([InlineKeyboardButton("📄 Новый файл", callback_data="new_file_confirm")])
    return InlineKeyboardMarkup(rows)


def preview_caption(context) -> str:
    return (f"Превью точек: <b>{len(context.user_data['points'])}</b>\n"
            f"Крест: <b>{context.user_data['cross_size']}</b>   Текст: <b>{context.user_data['text_height']}</b>\n\n"
            "Создайте файлы сразу или добавьте DXF-подложку (с проверкой СК).")


async def edit_msg(query, text: str, **kw):
    """Фото → меняем подпись, текст → меняем текст."""
    try:
        if query.message.photo:
            return await query.edit_message_caption(caption=text, **kw)
        return await query.edit_message_text(text=text, **kw)
    except Exception as e:  # например, «message is not modified»
        logger.debug("edit_msg: %s", e)


async def session_ok(update: Update, context) -> bool:
    if "points" in context.user_data:
        return True
    q = update.callback_query
    if q:
        await q.answer("Сессия устарела. Пришлите TXT заново.", show_alert=True)
    elif update.message:
        await update.message.reply_text("Сессия устарела. Пришлите TXT заново.")
    return False


# ═══════════════════════════════════════════════════════════════
#  Хендлеры: старт, TXT, размеры, превью
# ═══════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_session(context, update.effective_user.id)
    await update.message.reply_text(
        "Привет! Я собираю DXF + PDF из точек в TXT.\n\n"
        "1️⃣ Пришлите .txt (строки: «№ X Y [H]» или «X Y [H]»)\n"
        "2️⃣ Выберите размер креста и текста → превью\n"
        "3️⃣ Создайте файлы или добавьте DXF-подложку\n"
        "   (из памяти — 5 слотов — или новую) с проверкой СК\n\n"
        "/list — готовые файлы, /memory — слоты подложек, /cancel — отмена")
    return ConversationHandler.END


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    files = sorted((user_dir(update.effective_user.id) / "out").glob("*.*"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    if not files:
        await update.message.reply_text("Готовых файлов пока нет.")
        return
    await update.message.reply_text("Последние файлы:\n" + "\n".join(f"• <code>{html.escape(f.name)}</code>" for f in files),
                                    parse_mode=ParseMode.HTML)


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mem = load_memory(update.effective_user.id)
    lines = [f"{i}: {html.escape(mem[i]['name'])} ({mem[i].get('saved', '')[:10]})" if i in mem else f"{i}: —"
             for i in range(1, MEMORY_SLOTS + 1)]
    await update.message.reply_text("💾 Память подложек:\n" + "\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_session(context, update.effective_user.id)
    await update.message.reply_text("Отменено. Пришлите новый TXT или /start.")
    return ConversationHandler.END


async def handle_txt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    doc = update.message.document
    msg = await update.message.reply_text("Читаю файл…")
    f = await context.bot.get_file(doc.file_id)
    path = user_dir(uid) / "in" / safe_name(doc.file_name, ".txt")
    await f.download_to_drive(path)
    text = decode_bytes(path.read_bytes())
    points, skipped = parse_points_from_txt(text)
    if not points:
        await msg.edit_text("Не нашёл ни одной точки. Формат: «№ X Y [H]» или «X Y [H]».")
        return ConversationHandler.END

    reset_session(context, uid)
    context.user_data["points"] = points
    context.user_data["source_name"] = path.stem

    head = "\n".join(f"{p[0]}\t{p[1]:.3f}\t{p[2]:.3f}\t{p[3]:.3f}" for p in points[:8])
    note = f"\n⚠️ Пропущено строк: {skipped}" if skipped else ""
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Продолжить", callback_data="confirm_yes")],
                               [InlineKeyboardButton("❌ Отмена", callback_data="confirm_no")]])
    await msg.edit_text(
        f"Найдено точек: <b>{len(points)}</b>{note}\n\n<pre>{html.escape(head)}{'…' if len(points) > 8 else ''}</pre>",
        reply_markup=kb, parse_mode=ParseMode.HTML)
    return WAITING_CONFIRM


async def handle_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not await session_ok(update, context):
        return ConversationHandler.END
    if q.data != "confirm_yes":
        reset_session(context, q.from_user.id)
        await q.edit_message_text("Отменено. Пришлите новый TXT.")
        return ConversationHandler.END
    ac, at = calculate_auto_sizes(context.user_data["points"])
    await q.edit_message_text("Размер креста / высота текста (м):", reply_markup=sizes_keyboard(ac, at))
    return WAITING_SIZES


async def handle_size_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not await session_ok(update, context):
        return ConversationHandler.END
    if q.data == "size_custom":
        await edit_msg(q, "Напишите два числа — крест и текст, например: 0.55 0.75")
        return WAITING_CUSTOM_SIZE
    if q.data == "size_auto":
        cross, th = calculate_auto_sizes(context.user_data["points"])
    else:
        _, c, t = q.data.split("_")
        cross, th = float(c), float(t)
    context.user_data["cross_size"], context.user_data["text_height"] = cross, th
    try:
        await q.delete_message()
    except Exception:
        pass
    return await send_preview(q.message.chat_id, context)


async def handle_custom_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await session_ok(update, context):
        return ConversationHandler.END
    nums = [float(n) for n in _NUM.findall(update.message.text.replace(",", "."))]
    if len(nums) < 2 or nums[0] <= 0 or nums[1] <= 0:
        await update.message.reply_text("Нужно два положительных числа, пример: 0.5 0.7")
        return WAITING_CUSTOM_SIZE
    context.user_data["cross_size"] = round(max(0.15, min(nums[0], 4.0)), 3)
    context.user_data["text_height"] = round(max(0.2, min(nums[1], 5.0)), 3)
    return await send_preview(update.effective_chat.id, context)


async def send_preview(chat_id: int, context, underlay: Optional[DxfGeom] = None,
                       caption: Optional[str] = None, kb: Optional[InlineKeyboardMarkup] = None,
                       extra: str = "") -> int:
    uid = context.user_data["_uid"]
    pts = effective_points(context)
    png = user_dir(uid) / "out" / f"preview_{uuid4().hex[:8]}.png"
    await asyncio.to_thread(create_preview_image, pts, context.user_data["cross_size"],
                            context.user_data["text_height"], png, underlay, extra)
    with open(png, "rb") as f:
        await context.bot.send_photo(chat_id, f, caption=caption or preview_caption(context),
                                     parse_mode=ParseMode.HTML, reply_markup=kb or preview_keyboard())
    png.unlink(missing_ok=True)
    return WAITING_PREVIEW if kb is None else WAITING_CS


async def handle_preview(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not await session_ok(update, context):
        return ConversationHandler.END
    uid = q.from_user.id
    if q.data == "change_sizes":
        ac, at = calculate_auto_sizes(context.user_data["points"])
        await edit_msg(q, "Размер креста / высота текста (м):", reply_markup=sizes_keyboard(ac, at))
        return WAITING_SIZES
    if q.data == "add_underlay":
        await edit_msg(q, "📂 <b>Подложка:</b> выберите из памяти или загрузите новый DXF",
                       reply_markup=memory_keyboard(uid), parse_mode=ParseMode.HTML)
        return WAITING_UNDERLAY
    if q.data == "create_plain":
        context.user_data.pop("geom", None)
        context.user_data.pop("underlay_path", None)
        return await do_create(q.message.chat_id, context, q)
    if q.data == "new_file":
        return await ask_new_file(q)
    return WAITING_PREVIEW


# ═══════════════════════════════════════════════════════════════
#  Новый файл
# ═══════════════════════════════════════════════════════════════

async def ask_new_file(q) -> int:
    await edit_msg(q, "Сбросить текущий набор точек и начать заново?",
                   reply_markup=InlineKeyboardMarkup([
                       [InlineKeyboardButton("✅ Да, новый файл", callback_data="new_file_confirm")],
                       [InlineKeyboardButton("❌ Отмена", callback_data="new_file_cancel")]]))
    return WAITING_NEW_FILE


async def handle_new_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "new_file_confirm":
        reset_session(context, q.from_user.id)
        await edit_msg(q, "🔄 Сброс выполнен. Пришлите новый .txt с точками.", reply_markup=None)
        return ConversationHandler.END
    if not await session_ok(update, context):
        return ConversationHandler.END
    await edit_msg(q, preview_caption(context), reply_markup=preview_keyboard(), parse_mode=ParseMode.HTML)
    return WAITING_PREVIEW


# ═══════════════════════════════════════════════════════════════
#  Подложка: память / загрузка / единицы
# ═══════════════════════════════════════════════════════════════

async def handle_underlay_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not await session_ok(update, context):
        return ConversationHandler.END
    uid = q.from_user.id
    if q.data == "mem_cancel":
        await edit_msg(q, preview_caption(context), reply_markup=preview_keyboard(), parse_mode=ParseMode.HTML)
        return WAITING_PREVIEW
    if q.data == "mem_upload_new":
        await edit_msg(q, "📤 Пришлите DXF-файл подложки документом.", reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton("⬅️ Назад", callback_data="mem_back")]]))
        return WAITING_DXF_UPLOAD
    if q.data.startswith("mem_load_"):
        slot = int(q.data.rsplit("_", 1)[1])
        mem = load_memory(uid)
        if slot not in mem:
            await q.answer("Слот пуст", show_alert=True)
            return WAITING_UNDERLAY
        context.user_data["underlay_path"] = Path(mem[slot]["file"])
        context.user_data["underlay_name"] = mem[slot]["name"]
        context.user_data["unit_code"] = mem[slot].get("units")
        context.user_data["shift"] = tuple(mem[slot].get("shift", (0.0, 0.0)))
        await edit_msg(q, f"💾 Подложка из слота {slot}: {html.escape(mem[slot]['name'])}", parse_mode=ParseMode.HTML)
        return await after_underlay_loaded(q.message.chat_id, context)
    return WAITING_UNDERLAY


async def handle_dxf_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await edit_msg(q, "📂 <b>Подложка:</b>", reply_markup=memory_keyboard(q.from_user.id), parse_mode=ParseMode.HTML)
    return WAITING_UNDERLAY


async def handle_dxf_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await session_ok(update, context):
        return ConversationHandler.END
    uid = update.effective_user.id
    doc = update.message.document
    if not (doc.file_name or "").lower().endswith(".dxf"):
        await update.message.reply_text("Нужен файл .dxf")
        return WAITING_DXF_UPLOAD
    f = await context.bot.get_file(doc.file_id)
    path = user_dir(uid) / "in" / f"{uuid4().hex[:8]}_{safe_name(doc.file_name, '.dxf')}"
    await f.download_to_drive(path)
    context.user_data["underlay_path"] = path
    context.user_data["underlay_name"] = Path(doc.file_name).stem[:40]
    context.user_data["unit_code"] = None
    context.user_data["shift"] = (0.0, 0.0)
    return await after_underlay_loaded(update.effective_chat.id, context)


async def after_underlay_loaded(chat_id: int, context) -> int:
    path = context.user_data["underlay_path"]
    try:
        code, maxabs = await asyncio.to_thread(detect_units, path)
    except Exception as e:
        await context.bot.send_message(chat_id, f"❌ Не удалось прочитать DXF: {html.escape(str(e))}",
                                       parse_mode=ParseMode.HTML, reply_markup=memory_keyboard(context.user_data["_uid"]))
        return WAITING_UNDERLAY
    if context.user_data.get("unit_code") in UNIT_OVERRIDE or code is not None:
        if context.user_data.get("unit_code") not in UNIT_OVERRIDE:
            context.user_data["unit_code"] = code
        return await run_cs(chat_id, context)
    hint = "мм" if maxabs > 1_000_000 else ("м" if maxabs > 1000 else "?")
    await context.bot.send_message(
        chat_id, f"❓ В DXF не указаны единицы ($INSUNITS=0). Макс. |координата| ≈ {maxabs:,.0f}, похоже на: {hint}.\n"
                 "В чём нарисована подложка?",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("метры", callback_data="unit_m"),
                                            InlineKeyboardButton("миллиметры", callback_data="unit_mm"),
                                            InlineKeyboardButton("сантиметры", callback_data="unit_cm")]]))
    return WAITING_UNITS


async def handle_units(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not await session_ok(update, context):
        return ConversationHandler.END
    context.user_data["unit_code"] = q.data.split("_", 1)[1]
    await edit_msg(q, f"Единицы подложки: {context.user_data['unit_code']}", reply_markup=None)
    return await run_cs(q.message.chat_id, context)


# ═══════════════════════════════════════════════════════════════
#  Модуль 0 — хендлеры
# ═══════════════════════════════════════════════════════════════

async def run_cs(chat_id: int, context) -> int:
    wait = await context.bot.send_message(chat_id, "⏳ Считаю геометрию и проверяю СК…")
    shift = tuple(context.user_data.get("shift", (0.0, 0.0)))
    try:
        geom = await asyncio.to_thread(parse_dxf_geom, context.user_data["underlay_path"], shift,
                                       context.user_data.get("unit_code"))
        pts = effective_points(context)
        rep = await asyncio.to_thread(check_cs, pts, geom)
    except Exception as e:
        logger.exception("run_cs")
        await wait.edit_text(f"❌ Ошибка подложки: {html.escape(str(e))}", parse_mode=ParseMode.HTML,
                             reply_markup=memory_keyboard(context.user_data["_uid"]))
        return WAITING_UNDERLAY
    context.user_data["geom"] = geom
    context.user_data["rep"] = rep
    try:
        await wait.delete()
    except Exception:
        pass
    swapped = context.user_data.get("swap_xy", False)
    caption = cs_text(rep, shift, swapped, geom.scale_note)
    return await send_preview(chat_id, context, underlay=geom, caption=caption,
                              kb=cs_keyboard(rep, swapped), extra="с подложкой")


async def handle_cs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not await session_ok(update, context):
        return ConversationHandler.END
    chat = q.message.chat_id
    d = q.data
    if d == "cs_swap":
        context.user_data["swap_xy"] = not context.user_data.get("swap_xy", False)
        await edit_msg(q, "🔄 Меняю X↔Y и перепроверяю…", reply_markup=None)
        return await run_cs(chat, context)
    if d == "cs_recheck":
        await edit_msg(q, "🔁 Перепроверяю…", reply_markup=None)
        return await run_cs(chat, context)
    if d == "cs_shift":
        await context.bot.send_message(chat, "Введите X0 Y0 (м) — сдвиг, прибавляемый к координатам DXF. "
                                             "Например: <code>615000 1234000</code> или <code>0 0</code>",
                                       parse_mode=ParseMode.HTML)
        return WAITING_SHIFT
    if d == "cs_other":
        await edit_msg(q, "📂 <b>Подложка:</b>", reply_markup=memory_keyboard(q.from_user.id), parse_mode=ParseMode.HTML)
        return WAITING_UNDERLAY
    if d == "cs_create":
        return await do_create(chat, context, q)
    if d == "new_file":
        return await ask_new_file(q)
    return WAITING_CS


async def handle_shift_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await session_ok(update, context):
        return ConversationHandler.END
    nums = [float(n) for n in _NUM.findall(update.message.text.replace(",", "."))]
    if len(nums) < 2:
        await update.message.reply_text("Нужно два числа: X0 Y0")
        return WAITING_SHIFT
    context.user_data["shift"] = (nums[0], nums[1])
    return await run_cs(update.effective_chat.id, context)


# ═══════════════════════════════════════════════════════════════
#  Создание файлов, сохранение подложки в память
# ═══════════════════════════════════════════════════════════════

async def do_create(chat_id: int, context, q=None) -> int:
    uid = context.user_data["_uid"]
    geom: Optional[DxfGeom] = context.user_data.get("geom")
    if q:
        await edit_msg(q, "⏳ Создаю DXF и PDF…", reply_markup=None)
    try:
        dxf, pdf = await asyncio.to_thread(build_outputs, uid, effective_points(context),
                                           context.user_data["cross_size"], context.user_data["text_height"], geom)
    except Exception as e:
        logger.exception("build_outputs")
        await context.bot.send_message(chat_id, f"❌ Ошибка создания: {html.escape(str(e))}", parse_mode=ParseMode.HTML)
        return WAITING_PREVIEW
    rep: Optional[CSReport] = context.user_data.get("rep")
    warn = ""
    if geom and rep and not rep.ok:
        warn = f"\n⚠️ Создано с предупреждением: медиана до контура {fmt(rep.direct)} м."
    info = (f"✅ Готово: точек <b>{len(context.user_data['points'])}</b>"
            f"{', подложка: ' + html.escape(context.user_data.get('underlay_name', '')) if geom else ''}"
            f"{', X↔Y переставлены' if context.user_data.get('swap_xy') else ''}{warn}")
    with open(dxf, "rb") as f:
        await context.bot.send_document(chat_id, f, filename=dxf.name, caption=info, parse_mode=ParseMode.HTML)
    if pdf:
        with open(pdf, "rb") as f:
            await context.bot.send_document(chat_id, f, filename=pdf.name)
    else:
        await context.bot.send_message(chat_id, "PDF создать не удалось (см. лог), DXF отправлен.")
    await context.bot.send_message(chat_id, "Что дальше?", reply_markup=done_keyboard(bool(geom)))
    return WAITING_DONE


async def handle_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "new_file_confirm":
        reset_session(context, q.from_user.id)
        await edit_msg(q, "🔄 Готов к новому файлу. Пришлите .txt с точками.", reply_markup=None)
        return ConversationHandler.END
    if q.data == "save_underlay":
        if not context.user_data.get("underlay_path"):
            await q.answer("Подложка не использовалась", show_alert=True)
            return WAITING_DONE
        await edit_msg(q, "💾 В какой слот сохранить подложку?", reply_markup=save_slot_keyboard(q.from_user.id))
        return WAITING_SAVE_SLOT
    return WAITING_DONE


async def handle_save_slot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    if q.data == "slot_cancel":
        await edit_msg(q, "Не сохранено.", reply_markup=done_keyboard(True))
        return WAITING_DONE
    slot = int(q.data.split("_")[1])
    src: Path = context.user_data["underlay_path"]
    dst = user_dir(uid) / "underlays" / f"slot{slot}.dxf"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    mem = load_memory(uid)
    mem[slot] = {
        "name": context.user_data.get("underlay_name", src.stem)[:40],
        "file": str(dst),
        "units": context.user_data.get("unit_code"),
        "shift": list(context.user_data.get("shift", (0.0, 0.0))),
        "saved": datetime.now().isoformat(timespec="seconds"),
    }
    save_memory(uid, mem)
    await edit_msg(q, f"💾 Сохранено в слот {slot}: {html.escape(mem[slot]['name'])}\n"
                      f"(единицы: {mem[slot]['units'] or 'из файла'}, сдвиг {mem[slot]['shift']})",
                   parse_mode=ParseMode.HTML, reply_markup=done_keyboard(False))
    return WAITING_DONE


# ═══════════════════════════════════════════════════════════════
#  Сборка приложения
# ═══════════════════════════════════════════════════════════════

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled error", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(update.effective_chat.id, "⚠️ Что-то пошло не так. Попробуйте ещё раз или /start.")
        except Exception:
            pass


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    txt_filter = filters.Document.FileExtension("txt") | filters.Document.MimeType("text/plain")

    conv = ConversationHandler(
        entry_points=[MessageHandler(txt_filter, handle_txt)],
        states={
            WAITING_CONFIRM: [CallbackQueryHandler(handle_confirm, pattern=r"^confirm_")],
            WAITING_SIZES: [CallbackQueryHandler(handle_size_choice, pattern=r"^size_")],
            WAITING_CUSTOM_SIZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_size)],
            WAITING_PREVIEW: [CallbackQueryHandler(handle_preview,
                                                   pattern=r"^(change_sizes|add_underlay|create_plain|new_file)$")],
            WAITING_UNDERLAY: [CallbackQueryHandler(handle_underlay_choice, pattern=r"^mem_")],
            WAITING_DXF_UPLOAD: [MessageHandler(filters.Document.ALL, handle_dxf_upload),
                                 CallbackQueryHandler(handle_dxf_back, pattern=r"^mem_back$")],
            WAITING_UNITS: [CallbackQueryHandler(handle_units, pattern=r"^unit_")],
            WAITING_CS: [CallbackQueryHandler(handle_cs, pattern=r"^(cs_|new_file$)"),
                         MessageHandler(filters.TEXT & ~filters.COMMAND, handle_shift_text)],
            WAITING_SHIFT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_shift_text)],
            WAITING_NEW_FILE: [CallbackQueryHandler(handle_new_file, pattern=r"^new_file_")],
            WAITING_DONE: [CallbackQueryHandler(handle_done, pattern=r"^(save_underlay|new_file_confirm)$")],
            WAITING_SAVE_SLOT: [CallbackQueryHandler(handle_save_slot, pattern=r"^slot_")],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel), CommandHandler("start", cmd_start)],
        allow_reentry=True,
        name="txt2dxf",
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(conv)
    app.add_error_handler(on_error)
    logger.info("Bot started (shapely=%s)", HAS_SHAPELY)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
