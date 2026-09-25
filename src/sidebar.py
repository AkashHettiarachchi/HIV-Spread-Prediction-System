import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


_LOGO_PATH = Path(__file__).parent / "assets" / "LOGO1.png"
_LOGO_URI = "data:image/png;base64," + base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
_PCA_PATH = Path(__file__).parent / "assets" / "PCA.png"
_PCA_URI = "data:image/png;base64," + base64.b64encode(_PCA_PATH.read_bytes()).decode("ascii")

_NAV_ITEMS = [
    {"key": "overview", "page": "", "label": "Analytics overview"},
    {"key": "forecast", "page": "pages/2_Future_Forecast.py", "label": "Forecast horizon"},
    {"key": "benchmark", "page": "pages/4_Model_Benchmark.py", "label": "Model benchmark"},
    {"key": "scenario", "page": "pages/5_Scenario_Simulation.py", "label": "Scenario simulation"},
    {"key": "assistant", "page": "pages/3_AI_Assistant.py", "label": "AI knowledge assistant"},
]

_ACTIVE_ICON_SVG = {
    "overview": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#72FF98" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="4" y1="20" x2="4" y2="12"></line>'
        '<line x1="10" y1="20" x2="10" y2="6"></line>'
        '<line x1="16" y1="20" x2="16" y2="10"></line>'
        '<line x1="20" y1="20" x2="20" y2="4"></line>'
        '</svg>'
    ),
    "forecast": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#72FF98" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="5" y1="12" x2="19" y2="12"></line>'
        '<polyline points="12 5 19 12 12 19"></polyline>'
        '</svg>'
    ),
    "benchmark": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#72FF98" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M4 18V8M10 18V4M16 18v-8M22 18V6"></path>'
        '</svg>'
    ),
    "scenario": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#72FF98" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M6 3h12l-1 5-4 4v5l3 3H8l3-3v-5L7 8 6 3z"></path>'
        '<path d="M8 8h8"></path>'
        '</svg>'
    ),
    "assistant": (
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#72FF98" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 11.5a8.4 8.4 0 0 1-9 8.5 9.5 9.5 0 0 1-4-.9L3 21l1.9-4.7A8.2 8.2 0 0 1 3 11.5a8.4 8.4 0 0 1 9-8.5 8.4 8.4 0 0 1 9 8.5z"></path>'
        '<path d="M8 12h.01M12 12h.01M16 12h.01"></path>'
        '</svg>'
    ),
}

_ENABLED_ICON = {
    "overview": ":material/analytics:",
    "forecast": ":material/arrow_forward:",
    "benchmark": ":material/bar_chart:",
    "scenario": ":material/science:",
    "assistant": ":material/chat:",
}


def render_sidebar(active_page: str) -> None:
    """Render the shared Preditca sidebar without changing page behavior.

    The currently active page is rendered as a plain, non-interactive styled
    block (not a disabled st.page_link) so its appearance is fully controlled
    here rather than depending on Streamlit's internal disabled-link markup,
    which does not reliably expose a stable selector to target with CSS.
    Every other page is still rendered as a normal, fully working st.page_link
    -- navigation destinations and behavior are unchanged.
    """
    st.markdown(
        """<style>
        section[data-testid="stSidebar"] { background: #080D0A !important; border-right: 1px solid #1D3025; }
        div[data-testid="stSidebarNav"], section[data-testid="stSidebarNav"] { display: none !important; }
        section[data-testid="stSidebar"] > div:first-child { padding: 1.35rem 1.25rem 2rem; }
        section[data-testid="stSidebar"] section { justify-content: flex-start !important; }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 1.2rem; }
        .pca-brand { display: flex; align-items: center; gap: 14px; margin: .2rem 0 3rem; }
        .pca-mark { display: inline-flex; width: 52px; height: 52px; align-items: center; justify-content: center; flex: 0 0 52px; overflow: hidden; border-radius: 15px; background: #9AF5B5; box-shadow: 0 0 0 1px rgba(154,245,181,.12), 0 8px 24px rgba(50,255,120,.08); }
        .pca-mark img { display: block; width: 100%; height: 100%; object-fit: contain; }
        .pca-brand-copy { min-width: 0; }
        .pca-name { color: #F2F6F3; font-size: 1.7rem; font-weight: 750; line-height: 1.05; letter-spacing: -.04em; }
        .pca-subtitle { margin-top: .45rem; color: #8A9990; font-size: .64rem; font-weight: 600; letter-spacing: .16em; text-transform: uppercase; white-space: nowrap; }
        .pca-section { margin: 0 0 .8rem; color: #68766E; font-size: .72rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; }

        /* Real, clickable (non-active) nav links */
        section[data-testid="stSidebar"] [data-testid="stPageLink"] { display: block !important; width: 100% !important; height: auto !important; min-height: 34px !important; margin: 0 0 .25rem !important; padding: 0 !important; overflow: visible !important; }
        section[data-testid="stSidebar"] [data-testid="stPageLink"] > a, section[data-testid="stSidebar"] [data-testid="stPageLink"] a { height: 34px !important; min-height: 34px !important; box-sizing: border-box; display: flex !important; align-items: center; padding: 0 .85rem !important; border: 1px solid transparent; border-radius: 12px; color: #D8E1DC !important; font-size: 1.05rem !important; font-weight: 500 !important; text-decoration: none !important; transition: background .18s ease, color .18s ease, border-color .18s ease; overflow: visible !important; }
        section[data-testid="stSidebar"] [data-testid="stPageLink"] > a:hover {
            background: transparent !important;
            color: #9AF5B5 !important;
            border-color: transparent !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] [data-testid="stPageLink"] span[data-testid="stIconMaterial"] { color: #68766E !important; font-size: 19px !important; margin-right: .7rem !important; transform: none !important; transition: none !important; }

        /* Active (current) page -- plain styled block, not a real link */
        .pca-active { display: flex; align-items: center; width: 100%; height: 38px; min-height: 38px; margin: 0 0 .28rem; padding: 0 .85rem; box-sizing: border-box; border: 1px solid #294534; border-radius: 13px; background: #121C16; color: #F2F6F3; font-size: 1.05rem; font-weight: 500; box-shadow: 0 0 14px rgba(114,255,152,.10), inset 0 0 12px rgba(114,255,152,.04); }
        .pca-active-icon { display: inline-flex; align-items: center; justify-content: center; width: 20px; margin-right: .7rem; flex: 0 0 20px; }

        .pca-divider { width: 100%; height: 1px; margin: 20px 0 1.1rem; background: #1D3025; }
        .pca-status { width: 100%; padding: 0 .1rem 0; color: #F2F6F3; }
        .pca-status-title { display: flex; align-items: center; gap: .55rem; margin-bottom: .45rem; font-size: .92rem; font-weight: 650; }
        .pca-live-dot { width: 7px; height: 7px; border-radius: 50%; background: #72FF98; box-shadow: 0 0 0 4px rgba(114,255,152,.08); }
        .pca-live-label { color: #72FF98; font-size: .58rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
        .pca-status-copy { color: #8A9990; font-size: .76rem; line-height: 1.6; }
        .pca-chat-widget { position: fixed; right: 1.35rem; bottom: 1.15rem; z-index: 9999; display: flex; flex-direction: column; align-items: flex-end; }
        .pca-chat-image { cursor: grab; user-select: none; -webkit-user-drag: none; touch-action: none; }
        .pca-chat-image:active { cursor: grabbing; }
        .pca-chat-widget.pca-is-dragging { user-select: none; }
        .pca-chat-toggle { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
        .pca-chat-launcher { display: flex; flex-direction: column; align-items: center; gap: .35rem; cursor: pointer; }
        .pca-chat-launcher img { display: block; width: 172px; height: 172px; object-fit: contain; object-position: center bottom; filter: drop-shadow(0 10px 18px rgba(0,0,0,.42)); transition: transform .2s ease, filter .2s ease; }
        .pca-chat-launcher:hover img, .pca-chat-toggle:checked ~ .pca-chat-launcher img { transform: translateY(-3px) scale(1.03); filter: drop-shadow(0 12px 22px rgba(0,230,118,.22)); }
        .pca-chat-label { padding: .3rem .55rem; border: 1px solid #315541; border-radius: 8px; background: #101A14; color: #9AF5B5; font-size: .72rem; font-weight: 700; white-space: nowrap; }
        .pca-chat-panel { display: none; position: absolute; right: 0; bottom: 10.75rem; width: min(360px, calc(100vw - 2rem)); padding: 1.05rem; border: 1px solid #2B493A; border-radius: 16px; background: linear-gradient(145deg, #141A21, #0C1210); box-shadow: 0 18px 50px rgba(0,0,0,.46), 0 0 0 1px rgba(0,230,118,.04); animation: pca-chat-rise .2s ease-out; }
        .pca-chat-toggle:checked ~ .pca-chat-panel { display: block; }
        .pca-chat-panel::after { content: ""; position: absolute; right: 1.65rem; bottom: -7px; width: 13px; height: 13px; background: #0D1511; border-right: 1px solid #2B493A; border-bottom: 1px solid #2B493A; transform: rotate(45deg); }
        .pca-chat-close { position: absolute; top: .55rem; right: .65rem; z-index: 1; display: inline-flex; width: 28px; height: 28px; align-items: center; justify-content: center; border: 1px solid #385246; border-radius: 50%; background: #101914; color: #B5C8BD; cursor: pointer; font-size: 1.3rem; line-height: 1; transition: background .18s ease, color .18s ease, border-color .18s ease; }
        .pca-chat-close:hover { border-color: #00E676; background: #173021; color: #00E676; }
        .pca-chat-panel-title { color: #F1F7F3; font-family: 'Space Grotesk', sans-serif; font-size: 1.08rem; font-weight: 600; }
        .pca-chat-panel-copy { margin: .35rem 0 .9rem; color: #91A39A; font-size: .8rem; line-height: 1.5; }
        .pca-chat-panel a { display: block; position: relative; z-index: 1; padding: .7rem .85rem; border-radius: 9px; background: #00E676; color: #06100A !important; font-size: .82rem; font-weight: 700; text-align: center; text-decoration: none !important; transition: transform .18s ease, box-shadow .18s ease; }
        .pca-chat-panel a:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(0,230,118,.2); }
        @keyframes pca-chat-rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 600px) { .pca-chat-widget { right: .7rem; bottom: .7rem; } .pca-chat-launcher img { width: 144px; height: 144px; } .pca-chat-label { display: none; } .pca-chat-panel { bottom: 9.1rem; } }
        </style>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="pca-chat-widget"><input class="pca-chat-toggle" id="pca-chat-toggle" type="checkbox"><label class="pca-chat-launcher" for="pca-chat-toggle"><img class="pca-chat-image" src="{_PCA_URI}" alt="Open Preditca AI assistant"><span class="pca-chat-label">Ask Preditca</span></label><div class="pca-chat-panel"><label class="pca-chat-close" for="pca-chat-toggle" aria-label="Close Preditca chatbot">&times;</label><div class="pca-chat-panel-title">Ask Preditca</div><div class="pca-chat-panel-copy">Explore HIV health knowledge, Sri Lanka surveillance context, and the forecasting model.</div><a href="/AI_Assistant" target="_self">Open Preditca</a></div></div>',
        unsafe_allow_html=True,
    )
    components.html(


        """
        <script>
        (() => {
            const attachDrag = () => {
                const host = window.parent.document;
                const widget = host.querySelector('.pca-chat-widget');
                const image = widget && widget.querySelector('.pca-chat-image');
                const toggle = widget && widget.querySelector('.pca-chat-toggle');
                if (!widget || !image || !toggle || image.dataset.dragReady === 'true') return Boolean(widget && image);
                image.dataset.dragReady = 'true';
                let dragging = false;
                let moved = false;
                let suppressClick = false;
                let offsetX = 0;
                let offsetY = 0;

                image.addEventListener('pointerdown', (event) => {
                    event.preventDefault();
                    const rect = widget.getBoundingClientRect();
                    offsetX = event.clientX - rect.left;
                    offsetY = event.clientY - rect.top;
                    dragging = true;
                    moved = false;
                    suppressClick = false;
                    image.setPointerCapture(event.pointerId);
                });
                image.addEventListener('pointermove', (event) => {
                    if (!dragging) return;
                    if (Math.hypot(event.clientX - (offsetX + widget.getBoundingClientRect().left), event.clientY - (offsetY + widget.getBoundingClientRect().top)) < 4) return;
                    moved = true;
                    const rect = widget.getBoundingClientRect();
                    if (!widget.classList.contains('pca-is-dragging')) {
                        widget.style.left = `${rect.left}px`;
                        widget.style.top = `${rect.top}px`;
                        widget.style.right = 'auto';
                        widget.style.bottom = 'auto';
                        widget.classList.add('pca-is-dragging');
                    }
                    const maxLeft = Math.max(0, host.documentElement.clientWidth - widget.offsetWidth);
                    const maxTop = Math.max(0, host.documentElement.clientHeight - widget.offsetHeight);
                    const left = Math.min(maxLeft, Math.max(0, event.clientX - offsetX));
                    const top = Math.min(maxTop, Math.max(0, event.clientY - offsetY));
                    widget.style.left = `${left}px`;
                    widget.style.top = `${top}px`;
                });
                const stopDragging = (event) => {
                    if (!dragging) return;
                    dragging = false;
                    widget.classList.remove('pca-is-dragging');
                    suppressClick = true;
                    if (!moved) toggle.checked = !toggle.checked;
                    if (image.hasPointerCapture(event.pointerId)) image.releasePointerCapture(event.pointerId);
                };
                image.addEventListener('pointerup', stopDragging);
                image.addEventListener('pointercancel', stopDragging);
                image.addEventListener('click', (event) => {
                    if (!suppressClick) return;
                    event.preventDefault();
                    event.stopPropagation();
                    suppressClick = false;
                });
                return true;
            };
            const timer = setInterval(() => {
                if (attachDrag()) clearInterval(timer);
            }, 100);
            attachDrag();
        })();
        </script>
        """,
        height=0,
        width=0,
    )

    with st.sidebar:
        st.markdown(
            f'<div class="pca-brand"><div class="pca-mark"><img src="{_LOGO_URI}" alt="Preditca logo"></div><div class="pca-brand-copy">'
            '<div class="pca-name">P R E D I T C A</div><div class="pca-subtitle">HIV Epidemic AI Platform</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="pca-section">Workspace</div>', unsafe_allow_html=True)

        for item in _NAV_ITEMS:
            if active_page == item["key"]:
                st.markdown(
                    f'<div class="pca-active">'
                    f'<span class="pca-active-icon">{_ACTIVE_ICON_SVG[item["key"]]}</span>'
                    f'{item["label"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                page_target = item["page"]
                if item["key"] == "overview":
                    page_target = f"http://localhost:{st.get_option('server.port')}/"
                st.page_link(
                    page_target,
                    label=item["label"],
                    icon=_ENABLED_ICON[item["key"]],
                )

        st.markdown(
            '<div class="pca-divider"></div><div class="pca-status">'
            '<div class="pca-status-title"><span class="pca-live-dot"></span><span>Live model workspace</span>'
            '<span class="pca-live-label">Live</span></div><div class="pca-status-copy">'
            'Real NSACP surveillance data with a calibrated<br>SICA baseline and Bi-LSTM residual correction.</div></div>',
            unsafe_allow_html=True,
        )
