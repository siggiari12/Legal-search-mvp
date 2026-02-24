"""
app_ui.py — Streamlit frontend for the Legal Search MVP.

Connects to the FastAPI backend at http://localhost:8000/api/chat.
Run with:  streamlit run app_ui.py
"""

import json
import requests
import streamlit as st

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000/api/chat"
PAGE_TITLE  = "Lagaleit — Íslenskur lögfræðiaðstoðarmaður"

CONFIDENCE_BADGE = {
    "high":   ("🟢", "Há öryggi"),
    "medium": ("🟡", "Miðlungs öryggi"),
    "low":    ("🔴", "Lágt öryggi"),
    "none":   ("⚪", "Ekkert öryggi"),
}

FAILURE_LABELS = {
    "ambiguous_query":   "Spurningin er of almenn",
    "no_relevant_data":  "Engar heimildir fundust",
    "validation_failed": "Ekki tókst að staðfesta svar",
    "rate_limited":      "Of margar fyrirspurnir",
    "internal_error":    "Kerfisvilla",
}

# ─────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="⚖️",
    layout="centered",
)

# Minimal CSS: tighten citation blocks, style the badge
st.markdown("""
<style>
    /* Make the chat input sticky at the bottom */
    .stChatFloatingInputContainer { bottom: 1rem; }

    /* Citation expander */
    .citation-block {
        border-left: 3px solid #4a7cdc;
        padding: 0.4rem 0.8rem;
        margin: 0.4rem 0;
        background: #f7f9ff;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
    }
    .citation-locator {
        font-weight: 600;
        color: #1a3a6b;
        margin-bottom: 0.2rem;
    }
    .citation-quote {
        font-style: italic;
        color: #333;
    }

    /* Confidence badge */
    .conf-badge {
        display: inline-block;
        font-size: 0.78rem;
        padding: 2px 8px;
        border-radius: 10px;
        background: #eef2ff;
        color: #444;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Lagaleit")
    st.caption("Íslenskur lögfræðiaðstoðarmaður")
    st.divider()

    if st.button("🗑️ Hreinsa samtal", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    debug_mode = st.toggle("🔧 Debug JSON", value=False)

    st.divider()
    st.markdown("""
**Um tólið**

Þetta tól leitar í íslenska lagasafninu og svarar spurningum þínum með nákvæmum tilvísunum.

Dæmi um spurningar:
- *Hvenær fyrnist krafa?*
- *Hvernig er hlutafélag stofnað?*
- *Getur leigjandi rifið leigusamning?*
""")

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("⚖️ Lagaleit")
st.caption("Spurðu um íslensk lög — svör koma með nákvæmum heimildum.")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def query_backend(query: str) -> dict:
    """POST query to backend; return parsed JSON or error dict."""
    try:
        resp = requests.post(
            BACKEND_URL,
            json={"query": query},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": {
                "type": "connection_error",
                "message": "Ekki náðist samband við þjóninn. Er FastAPI keyrandi?",
            },
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": {
                "type": "timeout",
                "message": "Fyrirspurnin tók of langan tíma. Reyndu aftur.",
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": {"type": "unknown", "message": str(e)},
        }


def render_citations(citations: list):
    """Render citation list inside an expander."""
    if not citations:
        return
    label = f"📚 Sjá heimildir ({len(citations)})"
    with st.expander(label, expanded=False):
        for i, cit in enumerate(citations, 1):
            locator = cit.get("locator") or "—"
            quote   = cit.get("quote", "").strip()
            url     = cit.get("url")

            st.markdown(
                f'<div class="citation-block">'
                f'<div class="citation-locator">{i}. {locator}</div>'
                + (f'<div class="citation-quote">„{quote}"</div>' if quote else "")
                + (f'<div style="margin-top:4px"><a href="{url}" target="_blank">🔗 Opna heimild</a></div>' if url else "")
                + "</div>",
                unsafe_allow_html=True,
            )


def render_assistant_message(data: dict, raw_json: dict, show_debug: bool):
    """Render a successful assistant turn."""
    answer     = data.get("answer", "")
    citations  = data.get("citations", [])
    confidence = data.get("confidence", "none")

    # Answer
    st.markdown(answer)

    # Citations expander
    render_citations(citations)

    # Confidence badge
    icon, label = CONFIDENCE_BADGE.get(confidence, ("⚪", confidence))
    st.markdown(
        f'<span class="conf-badge">{icon} {label}</span>',
        unsafe_allow_html=True,
    )

    # Debug raw JSON
    if show_debug:
        with st.expander("🔧 Raw JSON"):
            st.json(raw_json)


def render_failure_message(data: dict, raw_json: dict, show_debug: bool):
    """Render a failure/refusal turn."""
    title      = data.get("title", "Villa")
    message    = data.get("message", "Óþekkt villa.")
    suggestion = data.get("suggestion")
    clarq      = data.get("clarification_question")

    st.warning(f"**{title}**\n\n{message}" + (f"\n\n💡 {suggestion}" if suggestion else ""))

    if clarq:
        st.info(f"❓ {clarq}")

    if show_debug:
        with st.expander("🔧 Raw JSON"):
            st.json(raw_json)


# ─────────────────────────────────────────────
# Replay existing messages
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            payload = msg["payload"]
            raw     = msg["raw"]
            if payload.get("success") and payload.get("data"):
                render_assistant_message(payload["data"], raw, debug_mode)
            elif payload.get("error"):
                err = payload["error"]
                st.error(f"**{err.get('type','Villa')}:** {err.get('message','')}")
                if debug_mode:
                    with st.expander("🔧 Raw JSON"):
                        st.json(raw)
            else:
                data = payload.get("data") or {}
                render_failure_message(data, raw, debug_mode)

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Spurðu um íslensk lög..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query backend with spinner
    with st.chat_message("assistant"):
        with st.spinner("Leita í lagasafninu…"):
            result = query_backend(prompt)

        # Render response
        if result.get("error"):
            err = result["error"]
            st.error(f"**{err.get('type', 'Villa')}:** {err.get('message', '')}")
        elif result.get("success") and result.get("data"):
            render_assistant_message(result["data"], result, debug_mode)
        else:
            data = result.get("data") or {}
            render_failure_message(data, result, debug_mode)

    # Persist to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "payload": result,
        "raw":     result,
    })
