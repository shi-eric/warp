# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the figures of the Distributed Computing documentation section.

Each figure is a declarative layout in ``viewBox`` units that renders to a light and a dark SVG file next to this
script, ``<stem>-light.svg`` and ``<stem>-dark.svg``. The page embeds both with the theme's ``only-light`` and
``only-dark`` classes. The rules the layouts follow, and the reasons for them, are in
``design/docs-figure-style.md``.

Run ``uv run docs/img/distributed/generate.py`` to rewrite every figure, or add ``--check`` to compare the files on
disk with a fresh rendering without writing anything.
"""

from __future__ import annotations

import argparse
import html
import itertools
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------------
# Palette, type and geometry
# ---------------------------------------------------------------------------------------------------------------------

MODES = ("light", "dark")

PALETTE: dict[str, dict[str, str]] = {
    "light": {
        "text": "#1a1a1a",
        "muted": "#666666",
        "surface": "#f7f7f7",
        "border": "#d1d5da",
        "primary": "#004831",
        "highlight": "#76b900",
        "wait": "#b35c00",
        "barrier": "#999999",
    },
    "dark": {
        "text": "#eeeeee",
        "muted": "#999999",
        "surface": "#1a1a1a",
        "border": "#48566b",
        "primary": "#76b900",
        "highlight": "#76b900",
        "wait": "#f0a742",
        "barrier": "#777777",
    },
}
"""Color of each role in each theme mode, from the palette table of the style guide."""

TINT_OPACITY = {"light": 0.22, "dark": 0.25}
"""Fill opacity of tinted shapes (highlight bands and colored operations) in each theme mode."""

SANS = "Arial, Helvetica, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, monospace"

# Font sizes; SMALL_SIZE is for captions and secondary labels.
TITLE_SIZE = 13
LABEL_SIZE = 12
CODE_SIZE = 10.5
SMALL_SIZE = 10.5

LABEL_MARGIN = 12
"""Minimum space between a label and each side of its box."""

# Extent of a text above and below its baseline, as fractions of the font size.
ASCENT = 0.75
DESCENT = 0.25

CENTER_SHIFT = 0.36
"""Offset from a box's vertical center to the baseline of a label centered in it: half the cap height."""

# Stroke widths; borders use the SVG default of 1.
GUIDE_WIDTH = 0.8
SECONDARY_WIDTH = 1.2
PRIMARY_WIDTH = 1.6

ARROWHEAD = 3.75
"""Arrowhead length and width in stroke widths: 6 units on a 1.6 data-movement arrow."""

OP_HEIGHT = 24

EPS = 1e-6

Point = tuple[float, float]


def text_width(s: str, size: float, mono: bool) -> float:
    """Estimate the rendered width of a string in the fallback fonts.

    An SVG shown through an ``<img>`` tag does not get the page's web fonts, so the browser sets its text in whichever
    fallback font is installed. The per-character table errs wide for the sans-serif fallbacks, so that a box sized
    from it leaves room for the wider ones.

    Args:
        s: The text.
        size: Font size in ``viewBox`` units.
        mono: Whether the text is set in the monospace stack.

    Returns:
        The width in ``viewBox`` units.
    """
    if mono:
        return 0.60 * size * len(s)
    wide = sum(ch in "mwMW" for ch in s)
    narrow = sum(ch in "iljtf.,:;|'" for ch in s)
    capitals = sum(ch.isupper() and ch not in "MW" for ch in s)
    other = len(s) - wide - narrow - capitals
    return (0.90 * wide + 0.30 * narrow + 0.72 * capitals + 0.56 * other) * size


def _num(v: float) -> str:
    """Format a coordinate with at most two decimals, so that output is stable and compact."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _pt(p: Point) -> str:
    return f"({_num(p[0])}, {_num(p[1])})"


def _check_role(role: str) -> None:
    if role not in PALETTE["light"]:
        raise ValueError(f"unknown color role {role!r}; expected one of {', '.join(PALETTE['light'])}")


# ---------------------------------------------------------------------------------------------------------------------
# Layout engine
# ---------------------------------------------------------------------------------------------------------------------


class LayoutError(Exception):
    """A figure breaks a layout rule of the style guide and is not rendered."""


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in ``viewBox`` units."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    def contains(self, other: Rect) -> bool:
        """Return whether ``other`` lies inside this rectangle; shared edges count as inside."""
        return (
            self.x0 - EPS <= other.x0
            and other.x1 <= self.x1 + EPS
            and self.y0 - EPS <= other.y0
            and other.y1 <= self.y1 + EPS
        )

    def contains_point(self, p: Point) -> bool:
        return self.x0 - EPS <= p[0] <= self.x1 + EPS and self.y0 - EPS <= p[1] <= self.y1 + EPS

    def overlaps(self, other: Rect) -> bool:
        """Return whether the two rectangles share an area; touching edges do not count."""
        return (
            min(self.x1, other.x1) - max(self.x0, other.x0) > EPS
            and min(self.y1, other.y1) - max(self.y0, other.y0) > EPS
        )

    def clip(self, a: Point, b: Point) -> tuple[float, float] | None:
        """Return the interval of the segment ``a + t * (b - a)`` with ``t`` in [0, 1] that lies inside the rectangle.

        Edges count as inside. Returns ``None`` when the segment misses the rectangle.
        """
        t0, t1 = 0.0, 1.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        for p, q in ((-dx, a[0] - self.x0), (dx, self.x1 - a[0]), (-dy, a[1] - self.y0), (dy, self.y1 - a[1])):
            if p == 0:
                if q < -EPS:
                    return None
            elif p < 0:
                t0 = max(t0, q / p)
            else:
                t1 = min(t1, q / p)
        return (t0, t1) if t0 <= t1 else None

    def crossed_by(self, a: Point, b: Point) -> bool:
        """Return whether the segment from ``a`` to ``b`` passes through the inside of the rectangle."""
        interval = self.clip(a, b)
        if interval is None:
            return False
        t = (interval[0] + interval[1]) / 2
        x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        return self.x0 + EPS < x < self.x1 - EPS and self.y0 + EPS < y < self.y1 - EPS


def _covered(a: Point, b: Point, rects: list[Rect]) -> bool:
    """Return whether every point of the segment from ``a`` to ``b`` lies in at least one of ``rects``."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length == 0:
        return any(r.contains_point(a) for r in rects)
    tolerance = EPS / length
    reach = 0.0
    for t0, t1 in sorted(iv for r in rects if (iv := r.clip(a, b)) is not None):
        if t0 > reach + tolerance:
            break
        reach = max(reach, t1)
    return reach >= 1 - tolerance


@dataclass(frozen=True)
class _Shape:
    rect: Rect
    fill: str | None
    stroke: str | None
    rx: float = 0
    tinted: bool = False
    dashed: bool = False

    def svg(self, mode: str) -> str:
        colors = PALETTE[mode]
        r = self.rect
        attrs = [f'x="{_num(r.x0)}" y="{_num(r.y0)}" width="{_num(r.x1 - r.x0)}" height="{_num(r.y1 - r.y0)}"']
        if self.rx:
            attrs.append(f'rx="{_num(self.rx)}"')
        attrs.append(f'fill="{colors[self.fill]}"' if self.fill else 'fill="none"')
        if self.tinted:
            attrs.append(f'fill-opacity="{_num(TINT_OPACITY[mode])}"')
        if self.stroke:
            attrs.append(f'stroke="{colors[self.stroke]}"')
        if self.dashed:
            attrs.append('stroke-dasharray="4 3"')
        return f"<rect {' '.join(attrs)}/>"


@dataclass(frozen=True)
class _Text:
    x: float
    y: float
    s: str
    size: float
    anchor: str
    mono: bool
    role: str

    @property
    def bounds(self) -> Rect:
        w = text_width(self.s, self.size, self.mono)
        x0 = {"start": self.x, "middle": self.x - w / 2, "end": self.x - w}[self.anchor]
        return Rect(x0, self.y - ASCENT * self.size, x0 + w, self.y + DESCENT * self.size)

    def svg(self, mode: str) -> str:
        attrs = [f'x="{_num(self.x)}" y="{_num(self.y)}"']
        if self.anchor != "start":
            attrs.append(f'text-anchor="{self.anchor}"')
        if self.mono:
            attrs.append(f'font-family="{MONO}"')
        attrs.append(f'font-size="{_num(self.size)}" fill="{PALETTE[mode][self.role]}"')
        return f"<text {' '.join(attrs)}>{html.escape(self.s, quote=False)}</text>"


@dataclass(frozen=True)
class _Path:
    points: tuple[Point, ...]
    role: str
    width: float
    dash: str | None
    marker: bool

    def svg(self, mode: str) -> str:
        d = "M " + " L ".join(f"{_num(x)} {_num(y)}" for x, y in self.points)
        attrs = [f'd="{d}" fill="none" stroke="{PALETTE[mode][self.role]}" stroke-width="{_num(self.width)}"']
        if self.dash:
            attrs.append(f'stroke-dasharray="{self.dash}"')
        if self.marker:
            attrs.append(f'marker-end="url(#ah-{self.role})"')
        return f"<path {' '.join(attrs)}/>"


def _marker(role: str, color: str) -> str:
    """Return an arrowhead marker sized in stroke widths, so that heads stay proportional to their arrows.

    The head's tip sits one stroke width beyond the end of the line, which :meth:`Figure.arrow` shortens by that much,
    so the square end of the line stays under the head and the tip lands on the arrow's last point.
    """
    # The marker's own coordinates are quarter stroke widths, which keeps its numbers exact.
    size, s, half, ref = _num(ARROWHEAD), _num(4 * ARROWHEAD), _num(2 * ARROWHEAD), _num(4 * ARROWHEAD - 4)
    return (
        f'<marker id="ah-{role}" viewBox="0 0 {s} {s}" refX="{ref}" refY="{half}" markerWidth="{size}" '
        f'markerHeight="{size}" orient="auto"><path d="M 0 0 L {s} {half} L 0 {s} z" fill="{color}"/></marker>'
    )


class Figure:
    """A figure layout that renders to a light and a dark SVG.

    Coordinates are ``viewBox`` units with the origin at the top left. Elements are drawn in the order they are added,
    so guides added before the panels run behind them and show only in the gaps.

    :meth:`render` refuses to emit a figure that breaks a layout rule: a label wider than its box minus
    ``LABEL_MARGIN`` on each side, two texts or boxes that overlap (a text or box nested inside a box is fine), a text
    or box outside the ``viewBox``, or an arrow segment that leaves the declared channels, crosses a text, or passes
    through a box that holds neither end of the arrow.

    Args:
        stem: File name stem; the figure renders to ``<stem>-light.svg`` and ``<stem>-dark.svg``.
        width: ``viewBox`` width: 640 for standard figures, 800 for wide timelines.
        height: ``viewBox`` height.
    """

    def __init__(self, stem: str, width: int, height: int):
        self.stem = stem
        self.width = width
        self.height = height
        self._items: list[_Shape | _Text | _Path] = []
        self._boxes: list[tuple[Rect, str]] = []
        self._texts: list[_Text] = []
        self._labels: list[tuple[_Text, Rect]] = []
        self._channels: list[Rect] = []
        self._arrows: list[tuple[tuple[Point, ...], str]] = []

    # -- primitives ---------------------------------------------------------------------------------------------------

    def panel(self, x: float, y: float, w: float, h: float, title: str | None = None) -> None:
        """Add a panel: a bordered surface that holds the elements of one process or one view.

        Args:
            x: Left edge.
            y: Top edge.
            w: Width.
            h: Height.
            title: Optional title, centered at the top of the panel.
        """
        rect = Rect(x, y, x + w, y + h)
        self._items.append(_Shape(rect, fill="surface", stroke="border", rx=4))
        self._boxes.append((rect, f"panel {title!r} at {_pt((x, y))}" if title else f"panel at {_pt((x, y))}"))
        if title:
            self._label(x + w / 2, y + 20, title, LABEL_SIZE, mono=False, role="text", box=rect)

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        mono: bool = False,
        dashed: bool = False,
        role: str = "text",
        row: bool = False,
    ) -> None:
        """Add a box with a label.

        A solid box is an object, such as a kernel or a step, and its label is centered in it. A dashed box marks a
        region that is not an object of its own, such as a heap inside a process; its label sits at the top in the
        muted color, leaving the inside free for the elements it holds.

        Args:
            x: Left edge.
            y: Top edge.
            w: Width.
            h: Height.
            label: Label text; it must fit with ``LABEL_MARGIN`` to spare on each side. An empty label draws a bare box.
            mono: Set the label in the monospace stack at code size.
            dashed: Draw a dashed region instead of a solid box.
            role: Palette role of the label color of a solid box, for example ``"muted"`` for a library-level name.
            row: Draw one row of a stack, such as a slab of a grid, instead of an object: square corners and a
                ``surface`` fill. A solid row has a ``border`` outline; a dashed row, such as a ghost row, a dashed
                ``muted`` one.
        """
        _check_role(role)
        rect = Rect(x, y, x + w, y + h)
        self._boxes.append((rect, f"box {label!r} at {_pt((x, y))}"))
        if row:
            self._items.append(_Shape(rect, fill="surface", stroke="muted" if dashed else "border", dashed=dashed))
        elif dashed:
            self._items.append(_Shape(rect, fill=None, stroke="muted", dashed=True))
        else:
            self._items.append(_Shape(rect, fill="surface", stroke="muted", rx=3))
        if not label:
            return
        if dashed:
            self._label(x + w / 2, y + 16, label, SMALL_SIZE, mono=mono, role="muted", box=rect)
        else:
            size = CODE_SIZE if mono else LABEL_SIZE
            self._label(x + w / 2, y + h / 2 + CENTER_SHIFT * size, label, size, mono=mono, role=role, box=rect)

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float = LABEL_SIZE,
        anchor: str = "middle",
        mono: bool = False,
        role: str = "text",
    ) -> None:
        """Add a free text.

        Args:
            x: Anchor position.
            y: Baseline position.
            s: The text.
            size: Font size: 13 for titles, 12 for labels, 10.5 for code and captions.
            anchor: ``"start"``, ``"middle"`` or ``"end"``.
            mono: Set the text in the monospace stack, as for code.
            role: Palette role of the text color; code labels use ``"primary"``.
        """
        if anchor not in ("start", "middle", "end"):
            raise ValueError(f"anchor must be 'start', 'middle' or 'end', not {anchor!r}")
        _check_role(role)
        t = _Text(x, y, s, size, anchor, mono, role)
        self._items.append(t)
        self._texts.append(t)

    def band(self, x: float, y: float, w: float, h: float, label: str) -> None:
        """Add a highlighted band, such as a symmetric array, with its name in monospace at its center.

        Args:
            x: Left edge.
            y: Top edge.
            w: Width.
            h: Height.
            label: Name shown in the band; an empty label draws a bare band.
        """
        rect = Rect(x, y, x + w, y + h)
        self._items.append(_Shape(rect, fill="highlight", stroke="highlight", tinted=True))
        self._boxes.append((rect, f"band {label!r} at {_pt((x, y))}"))
        if label:
            self._label(
                x + w / 2, y + h / 2 + CENTER_SHIFT * LABEL_SIZE, label, LABEL_SIZE, mono=True, role="text", box=rect
            )

    def lane(self, y: float, x0: float, x1: float, label: str) -> None:
        """Add a horizontal lane of a lane diagram, a muted line labelled at its left end.

        Args:
            y: Height of the lane line; operations on the lane are centered on it.
            x0: Left end of the lane line. The label ends 8 units to its left.
            x1: Right end of the lane line.
            label: Lane name, such as a PE or a stream.
        """
        self._items.append(_Path(((x0, y), (x1, y)), "muted", 1, None, marker=False))
        self.text(x0 - 8, y + CENTER_SHIFT * LABEL_SIZE, label, size=LABEL_SIZE, anchor="end")

    def op(self, x: float, y: float, w: float, label: str, role: str = "surface") -> None:
        """Add an operation: a rounded box centered on a lane, with a monospace label.

        Args:
            x: Left edge.
            y: Height of the lane the operation sits on.
            w: Width.
            label: Label text; an empty label draws a bare bar, such as a barrier.
            role: ``"surface"`` for a plain operation; any other palette role tints the box in that color, such as
                ``"primary"`` for data movement, ``"wait"`` for a blocked operation and ``"barrier"`` for a barrier.
        """
        _check_role(role)
        rect = Rect(x, y - OP_HEIGHT / 2, x + w, y + OP_HEIGHT / 2)
        if role == "surface":
            self._items.append(_Shape(rect, fill="surface", stroke="muted", rx=3))
        else:
            # The opaque base keeps the lane line from showing through the tint.
            self._items.append(_Shape(rect, fill="surface", stroke=None, rx=3))
            self._items.append(_Shape(rect, fill=role, stroke=role, rx=3, tinted=True))
        self._boxes.append((rect, f"op {label!r} at {_pt((x, y))}"))
        if label:
            self._label(x + w / 2, y + CENTER_SHIFT * CODE_SIZE, label, CODE_SIZE, mono=True, role="text", box=rect)

    def arrow(
        self, points: list[tuple[float, float]], role: str = "primary", dashed: bool = False, head: bool = True
    ) -> None:
        """Add an arrow along a polyline; the tip of its head lands on the last point.

        A ``"primary"`` arrow is a data-movement arrow, stroke 1.6; any other role draws a secondary arrow, stroke 1.2,
        such as a dim ``"muted"`` arrow. When the figure renders, every point of every segment must lie in a declared
        :meth:`channel`, and no segment may cross a text or pass through a box, except the boxes that hold the arrow's
        first or last point (its source, its target, and the panels or regions around them).

        Args:
            points: Start point, bends and tip. Two points give a straight arrow in any direction; more points give
                an orthogonal arrow with at most two bends. With a head, the last segment must be longer than it.
            role: Palette role of the arrow color.
            dashed: Dash the line, for dependencies that move no data.
            head: Draw the arrowhead. Without one, the line ends on the last point; use it for a tie that runs both
                ways, such as the one between the bars of a barrier.

        Raises:
            ValueError: The arrow is not straight or orthogonal, has more than two bends, or has a segment too short
                for its geometry.
        """
        _check_role(role)
        pts = tuple((float(x), float(y)) for x, y in points)
        if len(pts) < 2:
            raise ValueError("an arrow needs at least two points")
        directions = []
        for a, b in itertools.pairwise(pts):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length == 0:
                raise ValueError(f"arrow segment {_pt(a)}-{_pt(b)} has zero length")
            if len(pts) > 2 and a[0] != b[0] and a[1] != b[1]:
                raise ValueError(f"arrow segment {_pt(a)}-{_pt(b)} is diagonal; a bent arrow must be orthogonal")
            directions.append(((b[0] - a[0]) / length, (b[1] - a[1]) / length))
        bends = sum(d0 != d1 for d0, d1 in itertools.pairwise(directions))
        if bends > 2:
            raise ValueError(f"arrow from {_pt(pts[0])} has {bends} bends; at most two are allowed")

        width = PRIMARY_WIDTH if role == "primary" else SECONDARY_WIDTH
        end = pts[-1]
        if head:
            (ax, ay), (bx, by) = pts[-2], pts[-1]
            length = math.hypot(bx - ax, by - ay)
            # A shorter last segment would put the head over the bend, or flip it once the line is shortened under it.
            if length <= ARROWHEAD * width:
                raise ValueError(
                    f"the last segment of the arrow from {_pt(pts[0])} is {length:g} long; "
                    f"its arrowhead needs more than {ARROWHEAD * width:g}"
                )
            end = (bx - (bx - ax) / length * width, by - (by - ay) / length * width)
        self._items.append(_Path((*pts[:-1], end), role, width, "4 3" if dashed else None, marker=head))
        self._arrows.append((pts, role))

    def guide(self, points: list[tuple[float, float]], role: str = "primary", dashed: bool = True) -> None:
        """Add a thin guide line: a dashed guide connects related elements, a solid one brackets a group.

        A dashed guide connects elements that share a position across panels or a leader from an element to its
        bracket; add it before the panels it crosses so that it runs behind them. A solid guide, with
        ``dashed=False``, draws a bracket that marks the extent of a group. Guides are not checked against channels or
        text.

        Args:
            points: Points of the guide polyline.
            role: Palette role of the guide color.
            dashed: Draw the line dashed instead of solid.
        """
        _check_role(role)
        pts = tuple((float(x), float(y)) for x, y in points)
        self._items.append(_Path(pts, role, GUIDE_WIDTH, "3 3" if dashed else None, marker=False))

    def channel(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Declare a rectangle that arrows may run through, such as a gap between panels or a margin.

        Channels are not drawn.

        Args:
            x0: One horizontal edge.
            y0: One vertical edge.
            x1: The other horizontal edge.
            y1: The other vertical edge.
        """
        self._channels.append(Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    def caption(self, *lines: str) -> None:
        """Add a caption of one or two centered lines in the muted color at the bottom of the figure.

        Args:
            lines: Caption lines, top to bottom; the last baseline sits 17 units above the bottom edge.
        """
        for i, line in enumerate(lines):
            y = self.height - 17 - (len(lines) - 1 - i) * 18
            self.text(self.width / 2, y, line, size=SMALL_SIZE, role="muted")

    # -- checks and output --------------------------------------------------------------------------------------------

    def _label(self, x: float, y: float, s: str, size: float, mono: bool, role: str, box: Rect) -> None:
        self.text(x, y, s, size=size, mono=mono, role=role)
        self._labels.append((self._texts[-1], box))

    def _layout_errors(self) -> list[str]:
        """Return a description of every layout rule the figure breaks, in a stable order."""
        errors = []

        for label, box in self._labels:
            width = text_width(label.s, label.size, label.mono)
            room = box.width - 2 * LABEL_MARGIN
            if width > room + EPS:
                errors.append(
                    f"label {label.s!r} is {width:.1f} wide; its box leaves {room:.1f} inside the "
                    f"{LABEL_MARGIN}-unit margins"
                )

        canvas = Rect(0, 0, self.width, self.height)
        shapes = [(t.bounds, f"text {t.s!r}", True) for t in self._texts]
        shapes += [(rect, name, False) for rect, name in self._boxes]
        for rect, name, _ in shapes:
            if not canvas.contains(rect):
                errors.append(f"{name} extends beyond the {self.width}x{self.height} viewBox")
        for i, (a, name_a, a_is_text) in enumerate(shapes):
            for b, name_b, b_is_text in shapes[i + 1 :]:
                if not a.overlaps(b):
                    continue
                if a_is_text and b_is_text:
                    nested = False
                elif a_is_text:
                    nested = b.contains(a)
                elif b_is_text:
                    nested = a.contains(b)
                else:
                    nested = a.contains(b) or b.contains(a)
                if not nested:
                    errors.append(f"{name_a} overlaps {name_b}")

        for points, _role in self._arrows:
            ends = (points[0], points[-1])
            obstacles = [(rect, name) for rect, name in self._boxes if not any(rect.contains_point(p) for p in ends)]
            for a, b in itertools.pairwise(points):
                segment = f"arrow segment {_pt(a)}-{_pt(b)}"
                if not _covered(a, b, self._channels):
                    errors.append(f"{segment} leaves every declared channel")
                errors.extend(f"{segment} crosses text {t.s!r}" for t in self._texts if t.bounds.crossed_by(a, b))
                errors.extend(f"{segment} crosses {name}" for rect, name in obstacles if rect.crossed_by(a, b))

        return errors

    def render(self, mode: str) -> str:
        """Render the figure as SVG text for one theme mode.

        Args:
            mode: ``"light"`` or ``"dark"``.

        Returns:
            The SVG document, ending in a newline.

        Raises:
            LayoutError: The figure breaks a layout rule; the message lists every violation.
        """
        if mode not in PALETTE:
            raise ValueError(f"mode must be one of {', '.join(PALETTE)}, not {mode!r}")
        errors = self._layout_errors()
        if errors:
            raise LayoutError(f"{self.stem}: {len(errors)} layout error(s)\n  " + "\n  ".join(errors))

        colors = PALETTE[mode]
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" font-family="{SANS}">'
        ]
        roles = sorted({item.role for item in self._items if isinstance(item, _Path) and item.marker})
        if roles:
            lines.append("  <defs>")
            lines.extend(f"    {_marker(role, colors[role])}" for role in roles)
            lines.append("  </defs>")
        lines.extend(f"  {item.svg(mode)}" for item in self._items)
        lines.append("</svg>")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------------------------


def figure_symmetric_heap() -> Figure:
    """Build Figure 1: three PEs with ``u`` at the same offset of each symmetric heap, and a put from PE 0 to PE 1."""
    fig = Figure("symmetric_heap", 640, 345)
    fig.text(
        320, 24, "One collective allocation: u lives at the same offset in every PE's symmetric heap", size=TITLE_SIZE
    )

    # The shared offset of u, behind the panels.
    fig.guide([(12, 118), (628, 118)])
    fig.guide([(12, 154), (628, 154)])

    for pe, x in enumerate((25, 235, 445)):
        fig.panel(x, 40, 180, 242, title=f"PE {pe} · GPU {pe}")
        fig.box(x + 15, 72, 150, 128, "symmetric heap", dashed=True)
        fig.band(x + 15, 118, 150, 36, "u")

    fig.box(70, 216, 90, 26, "kernel", mono=True)
    fig.text(115, 260, "wp.nvshmem_put(", size=CODE_SIZE, mono=True, role="primary")
    fig.text(115, 274, "u, u[i:j], pe=1)", size=CODE_SIZE, mono=True, role="primary")
    fig.arrow([(160, 229), (220, 229), (220, 136), (248, 136)])

    fig.arrow([(535, 210), (535, 156)], role="muted")
    fig.text(535, 228, "wp.array(ptr=buf.handle)", size=CODE_SIZE, mono=True, role="primary")
    fig.text(535, 244, "this PE's copy of u", size=SMALL_SIZE, role="muted")

    # Gaps between the panels and the left and right margins.
    fig.channel(205, 40, 235, 282)
    fig.channel(415, 40, 445, 282)
    fig.channel(0, 40, 25, 282)
    fig.channel(625, 40, 640, 282)
    # Inside the panels: the row right of the kernel, PE 1's band row outside its heap, and the lane under PE 2's band.
    fig.channel(160, 216, 205, 242)
    fig.channel(235, 118, 250, 154)
    fig.channel(525, 154, 545, 210)

    fig.caption(
        "Every PE allocates u with the same collective call, and Warp wraps the local copy as a non-owning array.",
        "A kernel names the copy on another PE by its PE index; the dashed lines mark the shared offset.",
    )
    return fig


def _fit(label: str, size: float = LABEL_SIZE, mono: bool = False) -> int:
    """Return the width of a box that holds ``label`` with ``LABEL_MARGIN`` to spare on each side."""
    return math.ceil(text_width(label, size, mono)) + 2 * LABEL_MARGIN


def _code_w(label: str) -> int:
    """Return the width of an operation that holds ``label``, which :meth:`Figure.op` sets in code type."""
    return _fit(label, CODE_SIZE, mono=True)


def figure_halo_slabs() -> Figure:
    """Build Figure 2: two row slabs with ghost rows, and the two puts that fill the neighbors' ghost rows."""
    fig = Figure("halo_slabs", 640, 404)
    fig.text(320, 24, "Row slabs: each PE stores its owned rows plus two ghost rows", size=TITLE_SIZE)

    row, left, width = 22, 158, 150
    right = left + width
    # Per slab: the top edge, the kind of each row from top to bottom, and the label of the boundary row.
    slabs = [
        (46, ["ghost", "owned", "owned", "owned", "boundary", "ghost"], "last owned row"),
        (214, ["ghost", "boundary", "owned", "owned", "owned", "ghost"], "first owned row"),
    ]
    # Owned rows first, then the boundary rows, then the ghost rows, so that the dashed edges stay on top.
    for kind in ("owned", "boundary", "ghost"):
        for top, rows, _ in slabs:
            for i in (i for i, name in enumerate(rows) if name == kind):
                if kind == "boundary":
                    fig.band(left, top + i * row, width, row, "")
                else:
                    fig.box(left, top + i * row, width, row, "", dashed=kind == "ghost", row=True)
    for pe, (top, _, _) in enumerate(slabs):
        fig.text(left - 12, top + 3 * row + CENTER_SHIFT * LABEL_SIZE, f"PE {pe}", anchor="end")

    def center(slab: int, i: int) -> float:
        """Return the height of the center of row ``i`` of slab ``slab``."""
        return slabs[slab][0] + (i + 0.5) * row

    # The labels sit at the right end of their rows, because the strip right of the slabs belongs to the arrows. A
    # boundary row lies on a highlight band, which the muted color does not contrast with enough in dark mode.
    for k, (_, rows, boundary) in enumerate(slabs):
        for i, name in enumerate(rows):
            if name != "owned":
                y = center(k, i) + CENTER_SHIFT * SMALL_SIZE
                text, role = ("ghost", "muted") if name == "ghost" else (boundary, "text")
                fig.text(right - 8, y, text, SMALL_SIZE, "end", role=role)

    # The two arrows run through that strip, one vertical each.
    inner, outer = right + 190, right + 212

    # PE 0's last owned row to PE 1's top ghost row, and PE 1's first owned row to PE 0's bottom ghost row.
    last_owned, bottom_ghost = center(0, slabs[0][1].index("boundary")), center(0, len(slabs[0][1]) - 1)
    top_ghost, first_owned = center(1, 0), center(1, slabs[1][1].index("boundary"))
    fig.arrow([(right, last_owned), (inner, last_owned), (inner, top_ghost), (right + 2, top_ghost)])
    fig.arrow([(right, first_owned), (outer, first_owned), (outer, bottom_ghost), (right + 2, bottom_ghost)])
    put = "wp.nvshmem_put(..., pe {} 1)"
    fig.text((right + inner) / 2, last_owned - 7, put.format("+"), CODE_SIZE, mono=True, role="primary")
    fig.text((right + outer) / 2, first_owned + 16, put.format("-"), CODE_SIZE, mono=True, role="primary")
    fig.channel(right, last_owned - 6, outer + 6, first_owned + 6)

    fig.caption(
        "A put copies a boundary row into the neighbor's ghost row.",
        "That ghost row is a slice of the same symmetric array, at an offset known to both PEs.",
    )
    return fig


def figure_warp_nvshmem_stack() -> Figure:
    """Build Figure 3: the host steps and the first-launch steps of one PE, and how they connect."""
    fig = Figure("warp_nvshmem_stack", 640, 384)
    fig.text(320, 24, "How a Warp kernel reaches NVSHMEM", size=TITLE_SIZE)
    fig.box(8, 40, 624, 290, "one PE (process)", dashed=True)

    host_x, warp_x, lane_x = 26, 350, 326
    height, pitch, top = 28, 48, 94
    # The label, whether it is code, and the label role of each box, top to bottom.
    host = [
        ("mpirun / srun", True, "text"),
        ("wp.set_device(...)", True, "text"),
        ("nvshmem.init(...)", True, "text"),
        ("nvshmem.buffer(...) \u2192 wp.array(ptr=...)", True, "text"),
        ("wp.launch(...)", True, "text"),
    ]
    warp = [
        ("code generation \u2192 NVRTC (LTO)", False, "text"),
        ("link libnvshmem_device (nvJitLink)", False, "text"),
        ("nvshmemx_cumodule_init", True, "muted"),
        ("kernel runs; built-ins call NVSHMEM", False, "text"),
    ]
    fig.text(host_x, 80, "Host (Python)", anchor="start")
    fig.text(warp_x, 80, "Warp, on first launch", anchor="start")

    right: dict[str, float] = {}
    for x, column in ((host_x, host), (warp_x, warp)):
        column_width = max(_fit(label, CODE_SIZE if mono else LABEL_SIZE, mono) for label, mono, _ in column)
        for i, (label, mono, role) in enumerate(column):
            y = top + i * pitch
            w = _fit(label, CODE_SIZE if mono else LABEL_SIZE, mono)
            right[label] = x + w
            fig.box(x, y, w, height, label, mono=mono, role=role)
            if i:
                # A dim arrow from the box above, through the gap between the two.
                fig.arrow([(x + 40, y - pitch + height), (x + 40, y - 2)], role="muted")
                fig.channel(x, y - pitch + height, x + column_width, y)

    def center(i: int) -> float:
        return top + i * pitch + height / 2

    # wp.launch to the first Warp box: right, up through the strip between the columns, right again.
    launch_right = right["wp.launch(...)"]
    fig.arrow([(launch_right, center(4)), (lane_x, center(4)), (lane_x, center(0)), (warp_x - 2, center(0))])
    fig.channel(launch_right, center(4) - height / 2, lane_x + 6, center(4) + height / 2)
    fig.channel(lane_x - 6, center(0) - height / 2, lane_x + 6, center(4) + height / 2)
    fig.channel(lane_x - 6, center(0) - height / 2, warp_x, center(0) + height / 2)

    # nvshmem.init to nvshmemx_cumodule_init: the host library that init loaded is the one the module registers with.
    init_right = right["nvshmem.init(...)"]
    fig.arrow([(init_right, center(2)), (warp_x - 2, center(2))], role="muted")
    fig.channel(init_right, center(2) - height / 2, warp_x, center(2) + height / 2)
    fig.text((init_right + lane_x) / 2, center(2) - 9, "host library already loaded", SMALL_SIZE, role="muted")

    fig.caption(
        "Warp reuses the host library that the application loaded; the device library comes from",
        "the installed NVSHMEM wheel (3.9 or newer) or from WARP_NVSHMEM_DEVICE_LIBRARY.",
    )
    return fig


def figure_ordering_completion() -> Figure:
    """Build Figure 6: six calls of one thread in program order, and what each guarantees about the earlier ones."""
    fig = Figure("ordering_completion", 800, 300)
    fig.text(400, 24, "What each call guarantees about the operations this thread has issued", size=TITLE_SIZE)

    lane_y, start, step = 84, 84, 104
    fig.lane(lane_y, 70, 788, "thread")
    # The six labels need more width than one row of boxes has, so the calls alternate above and below the lane, each
    # resting against it and a step after the previous one in program order: data movement above, ordering and
    # completion below.
    offset = OP_HEIGHT / 2 + 1
    calls = [
        ("wp.nvshmem_put(A)", "primary"),
        ("wp.nvshmem_fence()", "surface"),
        ("wp.nvshmem_put(B)", "primary"),
        ("wp.nvshmem_quiet()", "surface"),
        ("wp.nvshmem_signal_op(...)", "primary"),
        ("wp.nvshmem_barrier_all()", "barrier"),
    ]
    centers, bottoms = [], []
    for i, (label, role) in enumerate(calls):
        x, w = start + i * step, _code_w(label)
        y = lane_y + (offset if i % 2 else -offset)
        fig.op(x, y, w, label, role)
        centers.append(x + w / 2)
        bottoms.append(y + OP_HEIGHT / 2)

    # Each bracket spans from the first call to the center of the last call it covers; a dashed leader joins that end
    # to the call, and a line of text sits under the bracket.
    brackets = [
        (2, "wp.nvshmem_fence() orders A before B at the same PE"),
        (3, "wp.nvshmem_quiet(): A and B are complete and visible at their targets"),
        (4, "Writing a flag after wp.nvshmem_quiet(), or with wp.nvshmem_put_signal(...), publishes the data"),
        (5, "wp.nvshmem_barrier_all(): every PE arrived and all prior operations are complete"),
    ]
    for k, (last, line) in enumerate(brackets):
        y, x = 134 + 36 * k, centers[last]
        fig.guide([(start, y - 4), (start, y), (x, y), (x, y - 4)], role="muted", dashed=False)
        fig.guide([(x, bottoms[last]), (x, y - 4)], role="muted")
        fig.text(start, y + 15, line, SMALL_SIZE, anchor="start", role="muted")

    fig.caption("Kernel completion does not imply remote completion.")
    return fig


def _code_op(fig: Figure, x: float, y: float, label: str, role: str = "surface", extra: float = 0) -> float:
    """Add an operation sized from its label, plus ``extra`` units of duration, and return its right edge."""
    w = _code_w(label) + extra
    fig.op(x, y, w, label, role)
    return x + w


def figure_two_sided_one_sided() -> Figure:
    """Build Figure 4: one row moved from PE 0 to PE 1 with MPI's two-sided calls and with NVSHMEM's one-sided ones."""
    fig = Figure("two_sided_one_sided", 640, 268)
    fig.text(320, 24, "Moving one row from PE 0 to PE 1, two-sided and one-sided", size=TITLE_SIZE)

    top, height, lanes, lead = 40, 190, (96, 160), 52
    # Each half: the panel's left edge and width, its title, the two lines of its note.
    halves = [
        (8, 292, "Two-sided (MPI)", ("Both sides post an operation;", "the receive gives the buffer.")),
        (312, 320, "One-sided (NVSHMEM)", ("Only the origin moves data;", "the target waits on a flag.")),
    ]
    for x, w, title, note in halves:
        fig.panel(x, top, w, height, title=title)
        for pe, y in enumerate(lanes):
            fig.lane(y, x + lead, x + w - 12, f"PE {pe}")
        for i, line in enumerate(note):
            fig.text(x + w / 2, 202 + 16 * i, line, SMALL_SIZE, role="muted")
    # The data moves down the gap between the lanes; the arrows stop short of their targets as in the other figures.
    tip = lanes[1] - OP_HEIGHT / 2 - 2
    fig.channel(8, lanes[0] + OP_HEIGHT / 2, 632, lanes[1] - OP_HEIGHT / 2)

    # Two-sided: both PEs post, then both wait for completion; the send is matched with the receive.
    start = 8 + lead + 16
    post_ends = []
    for y, post in zip(lanes, ("Isend", "Irecv"), strict=True):
        post_ends.append(_code_op(fig, start, y, post, "primary"))
        _code_op(fig, post_ends[-1] + 64, y, "Wait", "wait")
    send_x = (start + post_ends[0]) / 2
    fig.arrow([(send_x, lanes[0] + OP_HEIGHT / 2), (send_x, tip)])

    # One-sided: PE 1 is already waiting on its flag when PE 0's put arrives; the signal follows the data, so the wait
    # ends just after the put.
    start = 312 + lead + 8
    _code_op(fig, start, lanes[1], "wp.nvshmem_signal_wait_until(...)", "wait")
    put_end = _code_op(fig, start + 24, lanes[0], "wp.nvshmem_put_signal(...)", "primary")
    signal_x = put_end - 22
    fig.arrow([(signal_x, lanes[0] + OP_HEIGHT / 2), (signal_x, tip)])

    fig.caption("The signal arrives after the data, so a target that sees it reads the complete copy.")
    return fig


def figure_jacobi_step_timeline() -> Figure:
    """Build Figure 5: a Jacobi step on each PE's streams, pipelined on two streams and with a barrier per step."""
    fig = Figure("jacobi_step_timeline", 800, 556)
    fig.text(400, 24, "One Jacobi step as launches on each PE's streams", size=TITLE_SIZE)

    # The two panels share one time axis, and in both, each PE's heading sits above its lanes.
    lane_x0, lane_x1, start = 164, 784, 172
    half = OP_HEIGHT / 2
    # Space between two kernels on a stream; an event tie runs through its middle.
    gap = 12

    # Panel A. Each PE has a compute and a communication stream; the two communication lanes are in the middle, so
    # that each arrow between the PEs passes only a lane that is idle at its x.
    fig.panel(8, 40, 784, 276, title="pipelined, two streams")
    # Per PE: the heading's baseline, the compute lane and the communication lane.
    pes = [(80, 104, 152), (192, 264, 216)]
    wait, boundary, interior, push, quiet = (
        "wait flags",
        "boundary rows",
        "interior rows",
        "push rows + signal",
        "quiet",
    )
    boundary_x = start + _code_w(wait)
    push_x = boundary_x + _code_w(boundary) + gap
    quiet_x = push_x + _code_w(push)
    next_x = quiet_x + _code_w(quiet) + gap
    end_x = next_x + _code_w(wait) + _code_w(boundary)

    for pe, (heading, compute, comm) in enumerate(pes):
        fig.text(20, heading, f"PE {pe}", anchor="start")
        fig.lane(compute, lane_x0, lane_x1, "compute stream")
        fig.lane(comm, lane_x0, lane_x1, "communication stream")
        for x in (start, next_x):
            # The boundary kernel begins with its threads that read a ghost row waiting on the flag of that row.
            _code_op(fig, _code_op(fig, x, compute, wait, "wait"), compute, boundary)
        # The interior stencil runs on the compute stream while the push and the quiet run on the communication stream.
        fig.op(push_x, compute, next_x - gap - push_x, interior)
        _code_op(fig, _code_op(fig, push_x, comm, push, "primary"), comm, quiet)
        # Events: the communication stream waits for the boundary rows; the next boundary kernel waits for the quiet.
        label_y = (compute + comm) / 2 + CENTER_SHIFT * SMALL_SIZE
        for x, source, target in ((push_x - gap / 2, compute, comm), (next_x - gap / 2, comm, compute)):
            fig.arrow([(x, source), (x, target)], role="muted", dashed=True)
            fig.channel(x - 1, source, x + 1, target)
            fig.text(x - 6, label_y, "event", SMALL_SIZE, "end", role="muted")

    # Each push lands in the other PE's ghost row, which the other PE's next boundary kernel waits for. The arrows
    # leave the pushes 20 units apart, run through the gap between the communication lanes 16 units apart, and cross
    # once there; past the quiet, the lanes they pass are idle.
    (_, compute0, comm0), (_, compute1, comm1) = pes
    wait_center = next_x + _code_w(wait) / 2
    down_from, up_from, down, up = quiet_x - 36, quiet_x - 16, wait_center - 14, wait_center + 14
    fig.arrow([(down_from, comm0 + half), (down_from, comm0 + 24), (down, comm0 + 24), (down, compute1 - half - 2)])
    fig.arrow([(up_from, comm1 - half), (up_from, comm1 - 24), (up, comm1 - 24), (up, compute0 + half + 2)])
    fig.channel(push_x, comm0 + half, up + 1, comm1 - half)
    fig.channel(down - 1, comm0 + half, down + 1, compute1 - half)
    fig.channel(up - 1, compute0 + half, up + 1, comm1 - half)

    bracket_y, quiet_end = compute1 + half + 10, next_x - gap
    fig.guide(
        [(push_x, bracket_y - 4), (push_x, bracket_y), (quiet_end, bracket_y), (quiet_end, bracket_y - 4)],
        role="muted",
        dashed=False,
    )
    fig.text(
        (push_x + quiet_end) / 2,
        bracket_y + 16,
        "communication overlaps the interior stencil",
        SMALL_SIZE,
        role="muted",
    )

    # Panel B. One lane per PE. The stencil does the work of panel A's boundary and interior kernels, so it spans them;
    # PE 1's takes longer, and PE 0's barrier waits for it. Like panel A, the panel ends early in the next step, which
    # starts later than panel A's.
    fig.panel(8, 328, 784, 174, title="one stream, barrier per step")
    stencil, barrier = "stencil + element puts", "barrier_all"
    stencil_w, slower = next_x - boundary_x - 2 * gap, 20
    barrier_end = start + stencil_w + slower + gap + _code_w(barrier)
    next_step = barrier_end + gap
    if next_step <= next_x:
        raise LayoutError(f"{fig.stem}: the barrier form's step must not look shorter than the pipelined one")
    lanes = [(368, 392), (432, 456)]
    for pe, (heading, y) in enumerate(lanes):
        fig.text(20, heading, f"PE {pe}", anchor="start")
        fig.lane(y, lane_x0, lane_x1, "stream")
        stencil_end = start + stencil_w + (slower if pe == 1 else 0)
        fig.op(start, y, stencil_end - start, stencil)
        fig.op(stencil_end + gap, y, barrier_end - stencil_end - gap, barrier, "barrier")
        fig.op(next_step, y, end_x - next_step, stencil)
    (_, lane0), (_, lane1) = lanes
    tie_x = barrier_end - 12
    fig.arrow([(tie_x, lane0 + half), (tie_x, lane1 - half)], role="muted", dashed=True, head=False)
    fig.channel(tie_x - 1, lane0 + half, tie_x + 1, lane1 - half)
    fig.text(
        start,
        490,
        "Each barrier_all is a one-thread kernel; every PE waits in it for the slowest.",
        SMALL_SIZE,
        anchor="start",
        role="muted",
    )

    fig.caption(
        "Box widths show order and overlap, not measured durations. The two-stream form needs a second stream and two "
        "events per step.",
        "The barrier loop captures after an eager warm-up; the two-stream loop also needs the step number on the device "
        "to replay.",
    )
    return fig


def figure_backward_pass() -> Figure:
    """Build Figure 7: forward pushes, and the backward pass with its pulls and the barriers Warp inserts."""
    fig = Figure("backward_pass", 640, 244)
    fig.text(320, 24, "Forward pushes become backward pulls; Warp inserts the barriers", size=TITLE_SIZE)

    top, height, lanes, lead, gap, bar = 40, 150, (96, 160), 52, 10, 10
    half = OP_HEIGHT / 2
    fig.channel(8, lanes[0] + half, 632, lanes[1] - half)

    # Forward: a step, the push of its boundary row into the neighbor's ghost row, the next step.
    fig.panel(8, top, 287, height, title="forward")
    step, push = "step k", "push"
    step_x = 8 + lead + 8
    push_x = step_x + _code_w(step) + gap
    for pe, y in enumerate(lanes):
        fig.lane(y, 8 + lead, 8 + 287 - 12, f"PE {pe}")
        _code_op(fig, step_x, y, step)
        _code_op(fig, _code_op(fig, push_x, y, push, "primary") + gap, y, "step k+1")
    push_center = push_x + _code_w(push) / 2
    fig.arrow([(push_center, lanes[0] + half), (push_center, lanes[1] - half - 2)])

    # Backward: the adjoint launches in reverse order. The adjoint of the push pulls the gradient of the neighbor's
    # ghost row; Warp enqueues a barrier before it, because it reads remote memory, and one after the last adjoint
    # launch. The adjoint of a step is local and gets none.
    left = 8 + 287 + 10
    fig.panel(left, top, 640 - 8 - left, height, title="backward")
    adj_next, pull, adj = "adj k+1", "pull", "adj k"
    adj_x = left + lead + 8
    bar1 = adj_x + _code_w(adj_next) + gap
    pull_x = bar1 + bar + gap
    bar2 = pull_x + _code_w(pull) + gap + _code_w(adj) + gap
    for pe, y in enumerate(lanes):
        fig.lane(y, left + lead, 640 - 8 - 12, f"PE {pe}")
        _code_op(fig, adj_x, y, adj_next)
        fig.op(bar1, y, bar, "", "barrier")
        _code_op(fig, _code_op(fig, pull_x, y, pull, "primary") + gap, y, adj)
        fig.op(bar2, y, bar, "", "barrier")
    pull_center = pull_x + _code_w(pull) / 2
    fig.arrow([(pull_center, lanes[1] - half), (pull_center, lanes[0] + half + 2)])
    # A barrier joins the PEs both ways, so its tie has no head.
    for bar_x in (bar1, bar2):
        center = bar_x + bar / 2
        fig.arrow([(center, lanes[0] + half), (center, lanes[1] - half)], role="muted", dashed=True, head=False)

    fig.caption(
        "Grey bars are the one-thread barrier kernels that Warp enqueues before each adjoint launch",
        "that touches remote memory, plus one after the last adjoint launch.",
    )
    return fig


FIGURES: list[Callable[[], Figure]] = [
    figure_symmetric_heap,
    figure_halo_slabs,
    figure_warp_nvshmem_stack,
    figure_two_sided_one_sided,
    figure_jacobi_step_timeline,
    figure_ordering_completion,
    figure_backward_pass,
]


# ---------------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the light and dark SVG figures of this section.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the files on disk with a fresh rendering, print each one that differs, and exit with status 1",
    )
    args = parser.parse_args(argv)

    out_dir = Path(__file__).resolve().parent
    rendered: dict[Path, bytes] = {}
    try:
        for build in FIGURES:
            fig = build()
            for mode in MODES:
                rendered[out_dir / f"{fig.stem}-{mode}.svg"] = fig.render(mode).encode("utf-8")
    except LayoutError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    if args.check:
        # A Windows checkout may convert the committed files to CRLF line endings.
        stale = [
            path
            for path, data in rendered.items()
            if not path.is_file() or path.read_bytes().replace(b"\r\n", b"\n") != data
        ]
        # Figures whose stem is no longer registered.
        stale += sorted(path for mode in MODES for path in out_dir.glob(f"*-{mode}.svg") if path not in rendered)
        for path in stale:
            print(path)
        return 1 if stale else 0

    for path, data in rendered.items():
        if not path.is_file() or path.read_bytes() != data:
            path.write_bytes(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
