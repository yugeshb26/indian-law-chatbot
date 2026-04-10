"""
Chart extraction and rendering for Indian Law Chatbot.

Gemini is instructed (via SYSTEM_PROMPT) to append a <chart>…</chart>
block when its response contains data suitable for visualisation.
This module:
  1. parse_chart_from_response() — strips that block from the text and
     returns a (clean_text, chart_spec | None) tuple.
  2. render_chart()             — renders the spec as a Plotly figure
     styled to match the Aether Neon dark theme.
"""

import re
import json
import streamlit as st


# ── Palette — Aether Neon colours ────────────────────────────────────────────
_GOLD    = "#FFB84D"
_GOLD2   = "#FFD580"
_CYAN    = "#00E5FF"
_VIOLET  = "#B14AED"
_MAGENTA = "#FF3DA1"
_CYAN2   = "#80F5FF"
_VIO2    = "#D4A0FF"
_MAG2    = "#FF9CB5"

_PALETTE = [_GOLD, _CYAN, _VIOLET, _MAGENTA, _GOLD2, _CYAN2, _VIO2, _MAG2]


def _dark_layout(height: int = 320) -> dict:
    """Shared Plotly layout overrides for the dark glass theme."""
    return dict(
        paper_bgcolor="rgba(10,14,39,0.0)",
        plot_bgcolor="rgba(10,14,39,0.0)",
        font=dict(color="#E8EBF7", family="Inter, sans-serif", size=12),
        title_font=dict(color=_GOLD, size=14, family="Space Grotesk, Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.10)",
            borderwidth=1,
            font=dict(color="#E8EBF7", size=11),
        ),
        margin=dict(l=20, r=20, t=44, b=20),
        height=height,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.12)",
            tickfont=dict(color="#B0B8D8", size=11),
            title_font=dict(color="#E8EBF7"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.12)",
            tickfont=dict(color="#B0B8D8", size=11),
            title_font=dict(color="#E8EBF7"),
        ),
    )


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_chart_from_response(text: str) -> tuple[str, "dict | None"]:
    """Extract the first <chart>…</chart> block from *text*.

    Returns ``(clean_text, chart_spec)`` where *clean_text* has the block
    removed and *chart_spec* is the parsed JSON dict (or ``None`` if the
    block is absent or malformed).
    """
    match = re.search(r"<chart>\s*(.*?)\s*</chart>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return text, None

    clean_text = text[: match.start()].rstrip()
    try:
        spec = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return text, None   # malformed JSON → show everything as-is

    return clean_text, spec


def truncate_at_chart_tag(text: str) -> str:
    """During streaming, hide partial <chart> JSON from the live display."""
    idx = text.find("<chart>")
    return text[:idx].rstrip() if idx != -1 else text


# ── Renderer ──────────────────────────────────────────────────────────────────

def render_chart(spec: dict) -> None:
    """Render *spec* as a Plotly chart inside Streamlit.

    Supported types: bar, horizontal_bar, pie, donut, line, timeline.
    Silently skips on import error or bad data so it never breaks the chat.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.caption("_Install `plotly` for chart support._")
        return

    chart_type = str(spec.get("type", "bar")).lower()
    title      = spec.get("title", "")
    data       = spec.get("data", [])
    if not data:
        return

    try:
        fig = _build_figure(go, chart_type, title, data, spec)
        if fig is None:
            return
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )
    except Exception:
        pass  # charts are an enhancement — never crash the chat


def _build_figure(go, chart_type, title, data, spec):
    """Dispatch to the appropriate figure builder."""
    if chart_type in ("bar", "horizontal_bar"):
        return _bar(go, chart_type, title, data, spec)
    if chart_type in ("pie", "donut"):
        return _pie(go, chart_type, title, data)
    if chart_type == "line":
        return _line(go, title, data, spec)
    if chart_type == "timeline":
        return _timeline(go, title, data)
    return None


# ── Figure builders ───────────────────────────────────────────────────────────

def _bar(go, chart_type, title, data, spec):
    labels = [str(d.get("label", "")) for d in data]
    values = [float(d.get("value", 0))  for d in data]
    notes  = [str(d.get("note", ""))    for d in data]

    # Colour each bar by position cycling through palette
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]

    text_labels = [
        (n if n else str(int(v)) if v == int(v) else str(round(v, 1)))
        for v, n in zip(values, notes)
    ]

    if chart_type == "horizontal_bar":
        bar = go.Bar(
            y=labels, x=values,
            orientation="h",
            marker=dict(color=colours, opacity=0.88,
                        line=dict(color="rgba(255,255,255,0.12)", width=1)),
            text=text_labels, textposition="outside",
            textfont=dict(color="#E8EBF7", size=11),
        )
        layout = _dark_layout(max(280, len(labels) * 38))
        layout["yaxis"]["autorange"] = "reversed"
        layout["xaxis"]["title"]     = spec.get("ylabel", "")
        layout["yaxis"]["title"]     = spec.get("xlabel", "")
    else:
        bar = go.Bar(
            x=labels, y=values,
            marker=dict(color=colours, opacity=0.88,
                        line=dict(color="rgba(255,255,255,0.12)", width=1)),
            text=text_labels, textposition="outside",
            textfont=dict(color="#E8EBF7", size=11),
        )
        layout = _dark_layout()
        layout["xaxis"]["title"] = spec.get("xlabel", "")
        layout["yaxis"]["title"] = spec.get("ylabel", "")

    fig = go.Figure(bar)
    fig.update_layout(title=title, **{k: v for k, v in layout.items()})
    return fig


def _pie(go, chart_type, title, data):
    labels = [str(d.get("label", "")) for d in data]
    values = [float(d.get("value", 0)) for d in data]
    hole   = 0.38 if chart_type == "donut" else 0.0

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=hole,
        marker=dict(
            colors=_PALETTE[: len(labels)],
            line=dict(color="rgba(7,9,18,0.6)", width=2),
        ),
        textfont=dict(color="#E8EBF7", size=12),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    layout = _dark_layout(340)
    fig.update_layout(title=title, **{k: v for k, v in layout.items()})
    return fig


def _line(go, title, data, spec):
    x_vals  = [d.get("x", d.get("year", i))  for i, d in enumerate(data)]
    y_vals  = [float(d.get("y", d.get("value", 0))) for d in data]
    labels  = [str(d.get("label", "")) for d in data]

    fig = go.Figure()
    # Gradient-ish: line in cyan, markers in gold
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines+markers",
        line=dict(color=_CYAN, width=2.5,
                  shape="spline", smoothing=0.8),
        marker=dict(color=_GOLD, size=9,
                    line=dict(color=_CYAN, width=1.5)),
        text=labels,
        hovertemplate="%{x}: %{y}<br>%{text}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.07)",
    ))
    layout = _dark_layout()
    layout["xaxis"]["title"] = spec.get("xlabel", "")
    layout["yaxis"]["title"] = spec.get("ylabel", "")
    fig.update_layout(title=title, **{k: v for k, v in layout.items()})
    return fig


def _timeline(go, title, data):
    years  = [int(d.get("year", 0)) for d in data]
    labels = [str(d.get("label", "")) for d in data]

    # Alternate labels above/below the line so they don't overlap
    y_pos  = [0.12 if i % 2 == 0 else -0.12 for i in range(len(years))]
    anchor = ["bottom" if i % 2 == 0 else "top" for i in range(len(years))]

    fig = go.Figure()

    # Spine line
    fig.add_shape(
        type="line",
        x0=min(years), x1=max(years), y0=0, y1=0,
        line=dict(color=_CYAN, width=2),
    )

    # Vertical tick marks
    for yr in years:
        fig.add_shape(
            type="line",
            x0=yr, x1=yr, y0=-0.06, y1=0.06,
            line=dict(color=_GOLD, width=1.5),
        )

    fig.add_trace(go.Scatter(
        x=years,
        y=[0] * len(years),
        mode="markers+text",
        marker=dict(
            color=_GOLD, size=12,
            line=dict(color=_VIOLET, width=2),
            symbol="circle",
        ),
        text=labels,
        textposition=["top center" if i % 2 == 0 else "bottom center"
                      for i in range(len(years))],
        textfont=dict(color="#E8EBF7", size=10),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))

    layout = _dark_layout(260)
    layout["yaxis"] = dict(
        showticklabels=False, showgrid=False, zeroline=False,
        range=[-0.35, 0.35],
    )
    layout["xaxis"]["title"] = "Year"
    layout["showlegend"]     = False
    fig.update_layout(title=title, **{k: v for k, v in layout.items()})
    return fig
