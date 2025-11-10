# --- bootstrap sys.path so 'src' is importable ---
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------

import os
import time
import re
import html
import queue
import concurrent.futures

import streamlit as st
from dotenv import load_dotenv
from src.common.config import *
# Core agent module
from src.agents.db_operator_core import (
    AgentRuntime,
    ConnectionConfig,
    start_agent_thread,
    start_log_stream,
)

# ─────────────────────────────────────────────
# SIMPLE TOKEN AUTH (hard-coded with env override)
# ─────────────────────────────────────────────
# Change the default token to whatever you want, or set env DBOP_UI_TOKEN
HARD_TOKEN = os.getenv("DBOP_UI_TOKEN", "dbop-ui-secret-123")

def _auth_section() -> bool:
    """Render auth UI and return True if user is authenticated this session."""
    ss = st.session_state
    ss.setdefault("authed", False)
    ss.setdefault("auth_err", "")

    st.sidebar.markdown("### 🔐 Authentication")
    if ss.get("authed", False):
        st.sidebar.success("Authenticated")
        if st.sidebar.button("Lock"):
            ss.authed = False
            ss.auth_err = ""
            # Optionally clear runtime on logout
            if "runtime" in ss:
                try:
                    ss.runtime.stop_flag["stop"] = True
                except Exception:
                    pass
                ss.pop("runtime", None)
            st.rerun()
        return True

    token = st.sidebar.text_input("Access token", type="password", placeholder="Enter token")
    if st.sidebar.button("Unlock"):
        if (token or "").strip() == HARD_TOKEN:
            ss.authed = True
            ss.auth_err = ""
            st.rerun()
        else:
            ss.auth_err = "Invalid token."

    if ss.get("auth_err"):
        st.sidebar.error(ss["auth_err"])
    return False


# ─────────────────────────────────────────────
# SAFE RENDER + STABILITY HELPERS (UI-only)
# ─────────────────────────────────────────────
def _safe_str(x) -> str:
    try:
        import json
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        s = str(x)
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', s)

def _safe_html_text(text: str) -> str:
    s = _safe_str(text)
    return html.escape(s).replace("\n", "<br>")

def _render_html(container, html_str: str, fallback_text: str = ""):
    try:
        container.markdown(html_str, unsafe_allow_html=True)
    except Exception as e:
        container.warning(f"Render fallback (invalid HTML): {e}")
        container.code(_safe_str(fallback_text or html_str), language="text")

def _hash_chat(chat_history: list[tuple[str, str]]) -> str:
    import hashlib
    h = hashlib.sha256()
    for role, content in chat_history:
        h.update(role.encode("utf-8", errors="ignore")); h.update(b"\x00")
        h.update(str(content).encode("utf-8", errors="ignore")); h.update(b"\x00")
    return h.hexdigest()

def rerun_throttled(min_interval_sec: float = 1.0, key: str = "_last_rerun_ts"):
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if now - last >= min_interval_sec:
        st.session_state[key] = now
        st.rerun()

def _clean_ai_text(text: str) -> str:
    if not text: return ""
    keep = []
    for line in text.splitlines():
        if re.match(r'^\s*(Thought|Action|Observation|Agent Action|Tool|Reasoning)\b', line, re.I):
            continue
        keep.append(line)
    out = "\n".join(keep).strip()
    m = re.search(r'(?is)Final Answer:\s*(.+)$', out)
    return m.group(1).strip() if m else out

def _md_to_html(md_text: str) -> str:
    if not md_text:
        return ""
    esc = html.escape(md_text)
    def repl(m):
        code = m.group(2)
        return (
            "<pre style='margin:8px 0;padding:10px;background:#0d0d0d;border-radius:6px;overflow:auto;'>"
            "<code>" + html.escape(code) + "</code></pre>"
        )
    esc = re.sub(r"```(\w+)?\n(.*?)```", repl, esc, flags=re.S)
    esc = re.sub(r"`([^`]+)`", r"<code style='background:#0d0d0d;padding:2px 4px;border-radius:4px;'>\1</code>", esc)
    esc = re.sub(r"(?m)^- (.+)$", r"• \1", esc)
    esc = esc.replace("\n", "<br>")
    return esc


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Oracle DB Operator", layout="wide")
    load_dotenv()

    st.title("Oracle DB Operator 💬")
    st.caption("Conversational Agent for Oracle MCP Servers")

    # ---------- Authentication gate ----------
    authed = _auth_section()
    if not authed:
        st.info("Enter a valid token in the left panel to unlock the chatbot.")
        # Show minimal footer and return early; nothing else is initialized
        st.stop()

    # ---------- UI-local state (only after auth) ----------
    ss = st.session_state
    ss.setdefault("busy", False)
    ss.setdefault("suppress_refresh", False)
    ss.setdefault("chat_history", [])
    ss.setdefault("chat_input", "")
    ss.setdefault("pending_response", False)
    ss.setdefault("_chat_hash", None)
    ss.setdefault("_chat_rendered_once", False)

    # ---------- Core runtime (kept inside Streamlit session) ----------
    if "runtime" not in ss:
        ss.runtime = AgentRuntime()
    runtime = ss.runtime

    # ---------- Sidebar Controls ----------
    st.sidebar.markdown("### ⚙️ Agent Settings")
    auto_approve = st.sidebar.checkbox("Auto-approve SQL executions", value=False)

    agent_running = ("agent" in runtime.threads) and runtime.threads["agent"].is_alive()
    st.sidebar.markdown(f"**Status:** {'🟢 Running' if agent_running else '🔴 Stopped'}")

    with st.sidebar.expander("MCP Connections", expanded=True):
        def _to_int(v, fallback: int):
            try:
                return int(v)
            except Exception:
                return int(fallback)

        # Defaults from your config/env
        sqlcl_default = (os.getenv("SQLCLI_MCP_PROFILE") or str(SQLCLI_MCP_PROFILE)).strip()
        tavily_default = (os.getenv("TAVILY_MCP_SERVER") or str(TAVILY_MCP_SERVER)).strip()
        fs_key_default = (os.getenv("FILE_SYSTEM_ACCESS_KEY") or str(FILE_SYSTEM_ACCESS_KEY)).strip()

        redis_host_default = (os.getenv("MCP_SSE_HOST") or str(MCP_SSE_HOST)).strip()
        redis_port_default = _to_int(os.getenv("MCP_SSE_PORT") or MCP_SSE_PORT, MCP_SSE_PORT)

        dbtools_host_default = (os.getenv("MCP_SSE_HOST_DBTOOLS") or str(MCP_SSE_HOST_DBTOOLS)).strip()
        dbtools_port_default = _to_int(os.getenv("MCP_SSE_PORT_DBTOOLS") or MCP_SSE_PORT_DBTOOLS, MCP_SSE_PORT_DBTOOLS)

        sqlcl_command = st.text_input("SQLcl MCP command/profile", value=sqlcl_default, help="E.g., 'sql' or profile script")
        tavily_remote  = st.text_input("Tavily MCP Remote", value=tavily_default, help="E.g., https://…")
        filesystem_key = st.text_input("Filesystem server arg", value=fs_key_default)

        st.markdown("**Redis SSE (Streamable HTTP MCP)**")
        redis_host = st.text_input("Redis SSE Host (with scheme)", value=redis_host_default, help="E.g., https://localhost")
        redis_port = st.number_input("Redis SSE Port", value=redis_port_default, min_value=1, max_value=65535, step=1)

        st.markdown("**DBTools SSE (Streamable HTTP MCP)**")
        dbtools_host = st.text_input("DBTools SSE Host (with scheme)", value=dbtools_host_default, help="E.g., https://localhost")
        dbtools_port = st.number_input("DBTools SSE Port", value=dbtools_port_default, min_value=1, max_value=65535, step=1)

        conn_cfg = ConnectionConfig(
            sqlcl_command=sqlcl_command.strip(),
            tavily_remote=tavily_remote.strip(),
            filesystem_key=filesystem_key.strip(),
            redis_host=redis_host.strip(),
            redis_port=int(redis_port),
            dbtools_host=dbtools_host.strip(),
            dbtools_port=int(dbtools_port),
        )

    live_refresh = st.sidebar.toggle(
        "Live refresh logs",
        value=False,
        help="When ON, page re-runs every 4 seconds to pull new logs."
    )

    if st.sidebar.button("🧪 Test Add Pending SQL"):
        from langchain_core.tools import Tool
        fake_tool = Tool(name="test_sql", description="Fake SQL tool", func=lambda x: "Fake result")
        future = concurrent.futures.Future()
        runtime.pending_approvals["SELECT 1 FROM DUAL;"] = {
            "original_tool": fake_tool,
            "payload": {"sql": "SELECT 1 FROM DUAL;"},
            "future": future
        }
        st.info("Added test SQL for approval.")
        rerun_throttled(0.5)

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ Start Agent"):
        runtime.trace_logs.append(f"[{time.strftime('%H:%M:%S')}] 🚀 Starting MCP Agent…")
        with st.spinner("Starting MCP Agent..."):
            start_agent_thread(auto_approve, conn_cfg, runtime)
            start_log_stream(runtime)
            st.success("✅ Agent started")
        # drain a bit
        t_end = time.time() + 1.0
        while time.time() < t_end:
            drained = False
            try:
                while not runtime.log_q.empty():
                    runtime.trace_logs.append(_safe_str(runtime.log_q.get_nowait()))
                    drained = True
            except queue.Empty:
                pass
            if not drained:
                time.sleep(0.05)
        rerun_throttled(0.5)

    if c2.button("🛑 Stop Agent"):
        runtime.stop_flag["stop"] = True
        st.warning("Stopping agent...")

    # ---------- Logs ----------
    st.subheader("📜 Agent Logs (Live)")
    if st.button("🧹 Clear Logs"):
        runtime.trace_logs.clear()

    try:
        while not runtime.log_q.empty():
            runtime.trace_logs.append(_safe_str(runtime.log_q.get_nowait()))
    except queue.Empty:
        pass

    logs_box = st.empty()
    log_text_raw = "\n".join(runtime.trace_logs[-1500:]) or "No logs yet..."
    log_html = (
        "<div id='logsbox' style='background-color:#111;color:#EEE;padding:10px;"
        "border-radius:8px;height:300px;overflow-y:auto;"
        "font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Courier New\", monospace;"
        "font-size:13px;line-height:1.35;'>"
        + _safe_html_text(log_text_raw) +
        f"</div><span id='logsmarker' data-n='{len(runtime.trace_logs)}' style='display:none;'></span>"
        "<script>(function(){"
        "  var el=document.getElementById('logsbox'); if(!el) return;"
        "  function tail(){"
        "    el.scrollTop = el.scrollHeight;"
        "    el.scrollTop = el.scrollHeight - el.clientHeight + 1;"
        "  }"
        "  tail();"
        "  requestAnimationFrame(tail);"
        "  setTimeout(tail,0);"
        "})();</script>"
    )
    _render_html(logs_box, log_html, fallback_text=log_text_raw)

    # ---------- Conversation ----------
    st.markdown("#### 💬 Conversation")
    chat_ph = st.empty()

    def render_chat(always: bool = False):
        try:
            new_hash = _hash_chat(ss.chat_history)
        except Exception:
            new_hash = str(time.time())

        if not (always or not ss._chat_rendered_once or new_hash != ss._chat_hash):
            return

        ss._chat_hash = new_hash
        ss._chat_rendered_once = True

        bubbles = []
        for role, content in ss.chat_history:
            if role == "user":
                safe = _md_to_html(str(content))
                bubbles.append(
                    "<div style='background:#2a2a2a;color:#fff;padding:10px;border-radius:8px;margin-bottom:8px;'>"
                    "<b>👤 You:</b><br>" + safe + "</div>"
                )
            else:
                cleaned = _clean_ai_text(str(content))
                safe = _md_to_html(cleaned)
                bubbles.append(
                    "<div style='background:#1e1e1e;color:#9CDCFE;padding:10px;border-radius:8px;margin-bottom:8px;'>"
                    "<b>🤖 AI:</b><br>" + safe + "</div>"
                )

        html_block = (
            "<div id='chatbox' style='background:#111;color:#EEE;padding:10px;border-radius:8px;"
            "height:300px;overflow-y:auto;'>"
            + ("".join(bubbles) if bubbles else "<i>No conversation yet...</i>")
            + f"</div><span id='chatmarker' data-n='{len(ss.chat_history)}' style='display:none;'></span>"
            "<script>(function(){"
            "  var cb=document.getElementById('chatbox'); if(!cb) return;"
            "  function tail(){"
            "    cb.scrollTop = cb.scrollHeight;"
            "    cb.scrollTop = cb.scrollHeight - cb.clientHeight + 1;"
            "  }"
            "  tail();"
            "  requestAnimationFrame(tail);"
            "  setTimeout(tail,0);"
            "})();</script>"
        )
        _render_html(chat_ph, html_block, fallback_text="(chat render fallback)")

    render_chat(always=True)

    # ---------- Pending SQL Approvals ----------
    pending_sqls = list(runtime.pending_approvals.keys())
    if pending_sqls:
        st.markdown("### 📝 Pending SQL Approvals")
        for sql in pending_sqls:
            item = runtime.pending_approvals[sql]
            sql_card_html = (
                "<div style='border:1px solid #333; background:#0d0d0d; border-radius:10px; padding:12px; margin:8px 0;'>"
                "<div style='color:#bbb; font-size:12px; margin-bottom:6px;'>SQL Query</div>"
                "<div style='font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Courier New\", monospace;"
                "            font-size:13px; line-height:1.45; color:#E8E8E8; white-space:pre-wrap; word-break:break-word;'>"
                + _safe_html_text(sql) +
                "</div>"
                "</div>"
            )
            _render_html(st, sql_card_html, fallback_text=sql)

            col1, col2 = st.columns(2)
            if col1.button("✅ Approve and Execute", key=f"approve_{hash(sql)}"):
                try:
                    item['future'].set_result({"approved": True})
                    st.success("✅ Approved. Execution will proceed.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                finally:
                    runtime.pending_approvals.pop(sql, None)
                rerun_throttled(0.5)

            if col2.button("❌ Deny", key=f"deny_{hash(sql)}"):
                try:
                    item['future'].set_result({"approved": False})
                except Exception:
                    try:
                        item['future'].cancel()
                    except Exception:
                        pass
                finally:
                    runtime.pending_approvals.pop(sql, None)
                rerun_throttled(0.5)
    else:
        st.info(f"No pending SQL approvals. Pending count: {len(runtime.pending_approvals)}")

    # ---------- Input ----------
    status_ph = st.empty()
    if ss.pending_response:
        _render_html(
            status_ph,
            "<div style='margin-bottom:6px;color:#9CDCFE;'>🤖 <em>Thinking…</em></div>",
            fallback_text="Thinking…"
        )

    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area(
            "💬 Ask your question:",
            placeholder="Type your SQL or question here...",
            key="chat_input",
            height=80,
        )
        submitted = st.form_submit_button("🚀 Run", use_container_width=True)

    if submitted:
        q = (user_query or "").strip()
        agent_running_now = ("agent" in runtime.threads) and runtime.threads["agent"].is_alive()
        if not q:
            st.warning("Please enter a question.")
        elif not agent_running_now:
            st.error("Agent not running. Please start it first.")
        else:
            runtime.prompt_q.put(q)
            ss.chat_history.append(("user", q))
            ss.pending_response = True
            _render_html(
                status_ph,
                "<div style='margin-bottom:6px;color:#9CDCFE;'>🤖 <em>Thinking…</em></div>",
                fallback_text="Thinking…"
            )
            rerun_throttled(0.5)

    # ---------- Pull AI response (no extra rerun) ----------
    if ss.pending_response:
        try:
            reply = runtime.response_q.get_nowait()
            ss.chat_history.append(("ai", reply))
            ss.pending_response = False
            status_ph.empty()
            render_chat(always=True)
        except queue.Empty:
            pass

    # ---------- Optional auto-refresh logs only ----------
    try:
        if live_refresh:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=4000, key="live_log_refresh")
    except Exception:
        pass


if __name__ == "__main__":
    main()