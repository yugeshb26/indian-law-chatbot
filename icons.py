"""
Icon library for Indian Law Chatbot — Aether Neon design system.

All icons are inline SVGs, hand-crafted in a consistent line style with
1.75px stroke, rounded line caps and joins, 24x24 viewBox. They inherit
`currentColor` so they tint cleanly with CSS, and they share the .iconv
class so animations / hover effects apply uniformly across the app.

Usage:
    from icons import icon
    st.markdown(icon("scale", size=20, cls="icon-pulse"), unsafe_allow_html=True)
"""

# 24x24 path data only — wrapping happens in `icon()` so we can stamp
# consistent attributes (stroke, viewBox, classes) in one place.
_PATHS: dict[str, str] = {
    # ── Brand / domain ──────────────────────────────────────────
    # Scales of justice — used for the AI avatar + sidebar title
    "scale": (
        '<path d="M12 3v18"/>'
        '<path d="M5 7h14"/>'
        '<path d="M12 3a2 2 0 0 1 2 2"/>'
        '<path d="M5 7l-3 7c0 1.66 1.34 3 3 3s3-1.34 3-3l-3-7z"/>'
        '<path d="M19 7l-3 7c0 1.66 1.34 3 3 3s3-1.34 3-3l-3-7z"/>'
        '<path d="M8 21h8"/>'
    ),
    # Gavel — for legal CTAs
    "gavel": (
        '<path d="M14.5 4.5l5 5"/>'
        '<path d="M17 7l-9 9"/>'
        '<path d="M3 21l6-6"/>'
        '<path d="M12.5 2.5l9 9"/>'
        '<path d="M16 11l-5 5"/>'
    ),
    # Scroll / parchment — for laws & sections
    "scroll": (
        '<path d="M19 3H6a2 2 0 0 0-2 2v3h2"/>'
        '<path d="M19 3a2 2 0 0 1 2 2v3h-9"/>'
        '<path d="M6 8v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V8"/>'
        '<path d="M9 13h6"/>'
        '<path d="M9 17h4"/>'
    ),

    # ── UI controls ────────────────────────────────────────────
    "plus": (
        '<path d="M12 5v14"/>'
        '<path d="M5 12h14"/>'
    ),
    "send": (
        '<path d="M22 2L11 13"/>'
        '<path d="M22 2l-7 20-4-9-9-4 20-7z"/>'
    ),
    "trash": (
        '<path d="M3 6h18"/>'
        '<path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
        '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>'
        '<path d="M10 11v6"/>'
        '<path d="M14 11v6"/>'
    ),
    "refresh": (
        '<path d="M21 12a9 9 0 0 1-15.5 6.36L3 16"/>'
        '<path d="M3 12a9 9 0 0 1 15.5-6.36L21 8"/>'
        '<path d="M21 3v5h-5"/>'
        '<path d="M3 21v-5h5"/>'
    ),
    "arrow-right": (
        '<path d="M5 12h14"/>'
        '<path d="M13 5l7 7-7 7"/>'
    ),
    "copy": (
        '<rect x="9" y="9" width="13" height="13" rx="2"/>'
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>'
    ),
    "check": (
        '<path d="M20 6L9 17l-5-5"/>'
    ),
    "spark": (
        '<path d="M12 3v3"/>'
        '<path d="M12 18v3"/>'
        '<path d="M5.6 5.6l2.1 2.1"/>'
        '<path d="M16.3 16.3l2.1 2.1"/>'
        '<path d="M3 12h3"/>'
        '<path d="M18 12h3"/>'
        '<path d="M5.6 18.4l2.1-2.1"/>'
        '<path d="M16.3 7.7l2.1-2.1"/>'
        '<circle cx="12" cy="12" r="2"/>'
    ),
    "menu": (
        '<path d="M3 12h18"/>'
        '<path d="M3 6h18"/>'
        '<path d="M3 18h18"/>'
    ),
    "close": (
        '<path d="M18 6L6 18"/>'
        '<path d="M6 6l12 12"/>'
    ),
    "sidebar": (
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<path d="M9 3v18"/>'
    ),

    # ── Status / state ─────────────────────────────────────────
    "clock": (
        '<circle cx="12" cy="12" r="9"/>'
        '<path d="M12 7v5l3 2"/>'
    ),
    "alert": (
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<path d="M12 9v4"/>'
        '<circle cx="12" cy="17" r="0.5" fill="currentColor"/>'
    ),
    "user": (
        '<circle cx="12" cy="8" r="4"/>'
        '<path d="M4 21a8 8 0 0 1 16 0"/>'
    ),
    "bot": (
        '<rect x="3" y="8" width="18" height="12" rx="3"/>'
        '<path d="M12 4v4"/>'
        '<circle cx="12" cy="4" r="1.2" fill="currentColor"/>'
        '<circle cx="9" cy="14" r="0.9" fill="currentColor"/>'
        '<circle cx="15" cy="14" r="0.9" fill="currentColor"/>'
        '<path d="M9 18h6"/>'
    ),
    "lightbulb": (
        '<path d="M9 18h6"/>'
        '<path d="M10 21h4"/>'
        '<path d="M12 3a6 6 0 0 0-4 10.5c.7.7 1.5 1.5 1.5 2.5h5c0-1 .8-1.8 1.5-2.5A6 6 0 0 0 12 3z"/>'
    ),

    # ── Quick topics — domain icons ────────────────────────────
    # Constitution / fundamental rights — pillars of liberty
    "shield-check": (
        '<path d="M12 2l8 4v6c0 5-3.5 9.5-8 11-4.5-1.5-8-6-8-11V6l8-4z"/>'
        '<path d="M9 12l2 2 4-4"/>'
    ),
    # IPC sections — book/law book
    "book": (
        '<path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5z"/>'
        '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>'
    ),
    # RTI / search — magnifier
    "search": (
        '<circle cx="11" cy="11" r="7"/>'
        '<path d="M21 21l-5-5"/>'
    ),
    # Property — house
    "home": (
        '<path d="M3 11l9-8 9 8"/>'
        '<path d="M5 10v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V10"/>'
        '<path d="M10 21v-6h4v6"/>'
    ),
    # Consumer rights — shopping bag
    "bag": (
        '<path d="M6 7h12l-1 13a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L6 7z"/>'
        '<path d="M9 7V5a3 3 0 0 1 6 0v2"/>'
    ),
    # Labour law — briefcase
    "briefcase": (
        '<rect x="2" y="7" width="20" height="14" rx="2"/>'
        '<path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>'
    ),
    # FIR / siren
    "siren": (
        '<path d="M7 18v-6a5 5 0 0 1 10 0v6"/>'
        '<path d="M5 21h14"/>'
        '<path d="M12 4V2"/>'
        '<path d="M19 12h2"/>'
        '<path d="M3 12h2"/>'
        '<path d="M17.6 5.4l1.4-1.4"/>'
        '<path d="M5 5.4L6.4 4"/>'
    ),
    # Divorce / heart broken
    "heart-crack": (
        '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>'
        '<path d="M12 6l-2 4 4 2-2 4"/>'
    ),
    # Workplace — building
    "building": (
        '<rect x="4" y="3" width="16" height="18" rx="2"/>'
        '<path d="M9 7h.01"/>'
        '<path d="M15 7h.01"/>'
        '<path d="M9 11h.01"/>'
        '<path d="M15 11h.01"/>'
        '<path d="M9 15h6"/>'
    ),

    # ── Sidebar / chat history ─────────────────────────────────
    "message": (
        '<path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5z"/>'
    ),
    "messages": (
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
    ),
    "history": (
        '<path d="M3 12a9 9 0 1 0 9-9 9.7 9.7 0 0 0-6.4 2.6L3 8"/>'
        '<path d="M3 3v5h5"/>'
        '<path d="M12 7v5l3 2"/>'
    ),
    "bolt": (
        '<path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/>'
    ),
}


def icon(name: str, size: int = 20, cls: str = "", stroke_width: float = 1.75) -> str:
    """Return inline SVG markup for the named icon.

    Args:
        name: key in _PATHS (e.g. "scale", "send")
        size: pixel size for both width and height
        cls:  extra class names to append (e.g. "icon-pulse icon-gold")
        stroke_width: stroke thickness; the default 1.75 keeps lines crisp at 20px
    """
    paths = _PATHS.get(name)
    if paths is None:
        return ""  # silently no-op for unknown icons
    classes = f"iconv {cls}".strip()
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'class="{classes}" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f'{paths}'
        f'</svg>'
    )


def avatar_svg(role: str, size: int = 38) -> str:
    """Build a polished avatar block (gradient-filled SVG inside a glass ring).

    For the bot we use a stylised scale-of-justice mark; for the user a clean
    profile silhouette. The wrapper div carries the gradient ring + glow.
    """
    if role == "user":
        return (
            f'<div class="avatar user-avatar" style="width:{size}px;height:{size}px;">'
            f'{icon("user", size=int(size * 0.55))}'
            f'</div>'
        )
    # bot — scale icon, slightly larger because the geometry is busier
    return (
        f'<div class="avatar bot-avatar" style="width:{size}px;height:{size}px;">'
        f'{icon("scale", size=int(size * 0.6))}'
        f'</div>'
    )
