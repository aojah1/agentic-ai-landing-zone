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
import json
import queue
import shutil
import tempfile
from datetime import datetime
from src.common.config import (
    HARD_TOKEN,
    COMPARTMENT_ID,
    ENDPOINT,
    MODEL_ID,
)
import streamlit as st
from dotenv import load_dotenv

from src.agents.db_operator_core import (
    AgentRuntime,
    ConnectionConfig,
    start_agent_thread,
    start_log_stream,
    set_agent_history,
    clear_agent_history,
    set_force_auto_approve,
)

def _auth_section() -> bool:
    ss = st.session_state
    ss.setdefault("authed", False)
    ss.setdefault("auth_err", "")
    st.sidebar.markdown("### 🔐 Authentication")
    if ss.get("authed", False):
        st.sidebar.success("Authenticated")
        if st.sidebar.button("Lock"):
            ss.authed = False
            ss.auth_err = ""
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
# SAFE RENDER HELPERS
# ─────────────────────────────────────────────
def _safe_str(x) -> str:
    try:
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

def _render_tail_box(container, box_id: str, inner_html: str, height_px: int = 300, bg="#111", fg="#EEE"):
    """
    Scrollable box that always tails to bottom on load and on DOM mutations.
    """
    box_style = (
        f"background-color:{bg};color:{fg};padding:10px;border-radius:8px;"
        f"height:{height_px}px;overflow-y:auto;"
        'font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;'
        "font-size:13px;line-height:1.35;"
    )
    html_block = (
        f'<div id="{box_id}" style="{box_style}">{inner_html}</div>'
        "<script>(function(){"
        f'  var el=document.getElementById("{box_id}"); if(!el) return;'
        "  function tail(){ try{ el.scrollTop = el.scrollHeight; }catch(e){} }"
        "  tail(); requestAnimationFrame(tail); setTimeout(tail,0); setTimeout(tail,120);"
        "  try { var obs=new MutationObserver(function(){ tail(); });"
        "        obs.observe(el,{childList:true,subtree:true,characterData:true}); } catch(e) {}"
        "})();</script>"
    )
    _render_html(container, html_block, fallback_text=_safe_str(inner_html))

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

def rerun_throttled(min_interval_sec: float = 1.0, key: str = "_last_rerun_ts"):
    now = time.time()
    last = st.session_state.get(key, 0.0)
    if now - last >= min_interval_sec:
        st.session_state[key] = now
        st.rerun()


# ─────────────────────────────────────────────
# PERSISTENCE (FILE_SYSTEM_ACCESS_KEY/.dbop_checkpoints/checkpoints.json)
# ─────────────────────────────────────────────
def _fs_root_and_path():
    from src.common.config import FILE_SYSTEM_ACCESS_KEY
    root = os.path.abspath(os.path.expanduser(str(os.getenv("FILE_SYSTEM_ACCESS_KEY", FILE_SYSTEM_ACCESS_KEY))))
    root = os.path.realpath(root)
    folder = os.path.join(root, ".dbop_checkpoints")
    path = os.path.join(folder, "checkpoints.json")
    return root, folder, path

def _ensure_folder(folder: str):
    os.makedirs(folder, exist_ok=True)

def _atomic_write_json(path: str, data: dict):
    folder = os.path.dirname(path)
    _ensure_folder(folder)
    fd, tmp = tempfile.mkstemp(prefix=".cp_", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        shutil.move(tmp, path)
    finally:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception:
            pass

def _load_checkpoints_from_disk() -> dict:
    _, _, path = _fs_root_and_path()
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception as e:
        st.sidebar.warning(f"Could not load checkpoints: {e}")
        return {}

def _save_checkpoints_to_disk(checkpoints: dict):
    _, _, path = _fs_root_and_path()
    try:
        _atomic_write_json(path, checkpoints)
    except Exception as e:
        st.sidebar.error(f"Failed to save checkpoints: {e}")


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
        st.stop()

    # ---------- UI-local session ----------
    ss = st.session_state
    ss.setdefault("busy", False)
    ss.setdefault("chat_history", [])         # list[("user"|"ai", content)]
    ss.setdefault("chat_input", "")
    ss.setdefault("pending_response", False)

    # Replay state machine (non-blocking)
    ss.setdefault("replay", {
        "active": False,      # currently replaying?
        "prompts": [],        # list of user prompts to replay
        "i": 0,               # next prompt index
        "autoapprove": True,  # force auto-approve during replay
        "name": None,         # checkpoint name
        "clear_before": False # clear chat before starting replay
    })

    # checkpoints: name -> {created_at, notes, script: [user prompts], snapshot: [{"role","content"}]}
    ss.setdefault("checkpoints_loaded", False)
    ss.setdefault("checkpoints", {})
    ss.setdefault("last_checkpoint", None)

    if not ss.checkpoints_loaded:
        ss.checkpoints = _load_checkpoints_from_disk() or {}
        ss.checkpoints_loaded = True

    # ---------- Core runtime ----------
    # Do NOT eagerly initialize the runtime with stale env values.
    # Just ensure a placeholder for later.
    if "runtime" not in ss:
        ss.runtime = AgentRuntime()
    runtime = ss.runtime

    # ---------- Sidebar: Agent settings ----------
    st.sidebar.markdown("### ⚙️ Agent Settings")
    auto_approve = st.sidebar.checkbox("Auto-approve SQL executions", value=False)

    agent_running = ("agent" in runtime.threads) and runtime.threads["agent"].is_alive()
    st.sidebar.markdown(f"**Status:** {'🟢 Running' if agent_running else '🔴 Stopped'}")

    # ---------- Agent LLM / OCI GenAI Settings ----------
    with st.sidebar.expander("🤖 Agent LLM Settings (OCI GenAI)", expanded=False):
        # Initialize in session_state once, using env or config defaults
        if "oci_compartment" not in ss:
            ss.oci_compartment = COMPARTMENT_ID
        if "oci_endpoint" not in ss:
            ss.oci_endpoint = ENDPOINT
        if "oci_model_id" not in ss:
            ss.oci_model_id = MODEL_ID

        ss.oci_compartment = st.text_input(
            "OCI Compartment OCID",
            value=ss.oci_compartment,
            help="Compartment where the GenAI endpoint is authorized."
        )
        ss.oci_endpoint = st.text_input(
            "GenAI Endpoint",
            value=ss.oci_endpoint,
            help="e.g. https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        )
        ss.oci_model_id = st.text_input(
            "GenAI Model ID",
            value=ss.oci_model_id,
            help="e.g. xai.grok-3"
        )

        st.caption("These values will be applied when you start the MCP Agent.")


    with st.sidebar.expander("MCP Connections", expanded=False):
        def _to_int(v, fallback: int):
            try: return int(v)
            except Exception: return int(fallback)

        from src.common.config import (
            SQLCLI_MCP_PROFILE, TAVILY_MCP_SERVER, FILE_SYSTEM_ACCESS_KEY,
            MCP_SSE_HOST, MCP_SSE_PORT, MCP_SSE_HOST_DBTOOLS, MCP_SSE_PORT_DBTOOLS
        )

        sqlcl_default = (os.getenv("SQLCLI_MCP_PROFILE") or str(SQLCLI_MCP_PROFILE)).strip()
        tavily_default = (os.getenv("TAVILY_MCP_SERVER") or str(TAVILY_MCP_SERVER)).strip()
        fs_key_default = (os.getenv("FILE_SYSTEM_ACCESS_KEY") or str(FILE_SYSTEM_ACCESS_KEY)).strip()
        redis_host_default = (os.getenv("MCP_SSE_HOST") or str(MCP_SSE_HOST)).strip()
        redis_port_default = _to_int(os.getenv("MCP_SSE_PORT") or MCP_SSE_PORT, MCP_SSE_PORT)
        dbtools_host_default = (os.getenv("MCP_SSE_HOST_DBTOOLS") or str(MCP_SSE_HOST_DBTOOLS)).strip()
        dbtools_port_default = _to_int(os.getenv("MCP_SSE_PORT_DBTOOLS") or MCP_SSE_PORT_DBTOOLS, MCP_SSE_PORT_DBTOOLS)

        sqlcl_command = st.text_input("SQLcl MCP command/profile", value=sqlcl_default)
        tavily_remote  = st.text_input("Tavily MCP Remote", value=tavily_default)
        filesystem_key = st.text_input("Filesystem server arg", value=fs_key_default)
        st.markdown("**Redis SSE (Streamable HTTP MCP)**")
        redis_host = st.text_input("Redis SSE Host (with scheme)", value=redis_host_default)
        redis_port = st.number_input("Redis SSE Port", value=redis_port_default, min_value=1, max_value=65535, step=1)
        st.markdown("**DBTools SSE (Streamable HTTP MCP)**")
        dbtools_host = st.text_input("DBTools SSE Host (with scheme)", value=dbtools_host_default)
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

    # ─────────────────────────────────────────
    # 🧷 Checkpoints — Separate RESTORE and REPLAY
    # ─────────────────────────────────────────
    with st.sidebar.expander("🧷 Checkpoints", expanded=False):
        default_name = f"cp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        cp_name = st.text_input("Checkpoint name", value=default_name, help="Unique name")
        cp_notes = st.text_area("Notes (optional)", value="", height=60)

        col_cp1, col_cp2 = st.columns(2)
        if col_cp1.button("Save checkpoint", use_container_width=True):
            script = [c for (r, c) in ss.chat_history if r == "user"]
            snapshot = [{"role": r, "content": c} for (r, c) in ss.chat_history]
            ss.checkpoints[cp_name] = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "notes": cp_notes.strip(),
                "script": script,
                "snapshot": snapshot,
            }
            _save_checkpoints_to_disk(ss.checkpoints)
            ss.last_checkpoint = cp_name
            st.sidebar.success(f"Saved '{cp_name}'")
            rerun_throttled(0.2)

        if col_cp2.button("Clear all", use_container_width=True):
            ss.checkpoints.clear()
            ss.last_checkpoint = None
            _save_checkpoints_to_disk(ss.checkpoints)
            st.sidebar.warning("All checkpoints cleared.")
            rerun_throttled(0.2)

        if ss.checkpoints:
            names = sorted(ss.checkpoints.keys(), key=lambda n: ss.checkpoints[n]["created_at"], reverse=True)
            selected = st.selectbox("Select checkpoint", names, index=0 if names else None)

            with st.container():
                st.markdown("**Restore Options**")
                seed_memory = st.checkbox(
                    "Seed agent memory from snapshot (preload Q/A into agent state)",
                    value=True,
                    help="Restores the conversation in the panel and seeds the agent's memory. No automatic execution."
                )
                if st.button("Restore (show prior Q/A only)", use_container_width=True):
                    meta = ss.checkpoints[selected]
                    snapshot = meta.get("snapshot", [])
                    # Show prior Q/A in the conversation panel
                    ss.chat_history = [(d["role"], d["content"]) for d in snapshot if "role" in d and "content" in d]
                    # Seed agent memory (optional)
                    clear_agent_history(runtime)
                    if seed_memory and snapshot:
                        set_agent_history(runtime, snapshot)
                    ss.last_checkpoint = selected
                    st.sidebar.success(f"Restored '{selected}' (no execution).")
                    rerun_throttled(0.2)

            st.markdown("---")

            st.markdown("**Replay Options**")
            approve_replay = st.checkbox(
                "Auto-approve during replay",
                value=False,
                help="If OFF, approvals will be required and buttons will be interactive during replay."
            )
            clear_before_replay = st.checkbox(
                "Clear chat before replay",
                value=True,
                help="If ON, removes the current chat panel before re-running the saved script."
            )
            ensure_running = st.checkbox(
                "Ensure agent is started",
                value=True,
                help="Start the agent automatically if not already running."
            )

            cpr1, cpr2, cpr3 = st.columns([1,1,1])

            if cpr1.button("Preview", use_container_width=True):
                meta = ss.checkpoints[selected]
                st.sidebar.info(
                    f"Name: {selected}\n\n"
                    f"Created: {meta['created_at']}\n\n"
                    f"Notes: {meta.get('notes','')}\n\n"
                    f"Script length: {len(meta.get('script', []))} prompt(s)\n"
                    f"Snapshot turns: {len(meta.get('snapshot', []))}"
                )

            if cpr2.button("Start Replay", use_container_width=True):
                # Start the agent if needed
                if ensure_running and not (("agent" in runtime.threads) and runtime.threads["agent"].is_alive()):
                    start_agent_thread(auto_approve, conn_cfg, runtime)
                    start_log_stream(runtime)
                    time.sleep(0.5)

                meta = ss.checkpoints[selected]
                script = list(meta.get("script", []))

                # Clear chat panel if requested
                if clear_before_replay:
                    ss.chat_history = []

                # Initialize replay state (non-blocking). No seeding: Replay re-executes prompts.
                ss.replay = {
                    "active": True,
                    "prompts": script,
                    "i": 0,
                    "autoapprove": bool(approve_replay),
                    "name": selected,
                    "clear_before": bool(clear_before_replay),
                }
                set_force_auto_approve(runtime, bool(approve_replay))
                ss.last_checkpoint = selected
                st.sidebar.success(f"Replay started: '{selected}'")
                rerun_throttled(0.2)

            if cpr3.button("Delete", use_container_width=True):
                ss.checkpoints.pop(selected, None)
                if ss.last_checkpoint == selected:
                    ss.last_checkpoint = None
                _save_checkpoints_to_disk(ss.checkpoints)
                st.sidebar.warning(f"Deleted '{selected}'")
                rerun_throttled(0.2)
        else:
            st.info("No checkpoints yet.")

        # Show disk location
        _, folder, path = _fs_root_and_path()
        st.caption(f"Disk location: `{path}`")

        # ---------- Live auto-refresh (manual only) ----------
    try:
        # Default OFF; user can opt in from the sidebar if they really want polling
        live_refresh = st.sidebar.toggle(
            "Live refresh logs",
            value=False,
            help="When ON, page re-runs every ~4 seconds to pull new logs/approvals."
        )
        if live_refresh:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=4000, key="live_log_refresh")
    except Exception:
        pass


    # ---------- Start/Stop ----------
    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ Start Agent"):
        # 1) Apply latest OCI LLM settings to environment
        if "oci_compartment" in ss:
            os.environ["OCI_COMPARTMENT_ID"] = ss.oci_compartment.strip()
        if "oci_endpoint" in ss:
            os.environ["OCI_GENAI_ENDPOINT"] = ss.oci_endpoint.strip()
        if "oci_model_id" in ss:
            os.environ["OCI_GENAI_MODEL_ID"] = ss.oci_model_id.strip()

        # 2) Kill old runtime/threads if any
        try:
            if hasattr(ss.runtime, "stop_flag"):
                ss.runtime.stop_flag["stop"] = True
        except Exception:
            pass

        # 3) Recreate runtime so it picks up new env when it calls initialize_llm()
        ss.runtime = AgentRuntime()
        runtime = ss.runtime

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

    # ---------- Logs (auto-tail) ----------
    st.subheader("📜 Agent Logs (Live)")
    if st.button("🧹 Clear Logs"):
        runtime.trace_logs.clear()

    try:
        while not runtime.log_q.empty():
            runtime.trace_logs.append(_safe_str(runtime.log_q.get_nowait()))
    except queue.Empty:
        pass

    logs_box = st.empty()
    log_text_raw = "\n".join(runtime.trace_logs[-1500:]) if runtime.trace_logs else "No logs yet..."
    _render_tail_box(
        container=logs_box,
        box_id="logsbox",
        inner_html=_safe_html_text(log_text_raw),
        height_px=300,
        bg="#111",
        fg="#EEE",
    )

    # ---------- Conversation (auto-expanding, ChatGPT-style) ----------
    st.markdown("#### 💬 Conversation")

    # Use a normal container so the chat can grow with the page instead of a fixed-height box
    chat_container = st.container()

    bubbles = []
    for role, content in ss.chat_history:
        if role == "user":
            safe_user = _md_to_html(str(content))
            bubbles.append(
                "<div style='background:#2a2a2a;color:#fff;padding:10px;border-radius:8px;margin-bottom:8px;'>"
                "<b>👤 You:</b><br>" + safe_user + "</div>"
            )
        else:
            cleaned = _clean_ai_text(str(content))
            safe_ai = _md_to_html(cleaned)
            bubbles.append(
                "<div style='background:#1e1e1e;color:#9CDCFE;padding:10px;border-radius:8px;margin-bottom:8px;'>"
                "<b>🤖 AI:</b><br>" + safe_ai + "</div>"
            )

    inner_chat = "".join(bubbles) if bubbles else "<i>No conversation yet...</i>"

    # Wrap in a flex column so it feels like a chat thread, but no fixed height / scroll
    html_chat = (
        "<div style='display:flex;flex-direction:column;gap:8px;max-width:900px;'>"
        f"{inner_chat}"
        "</div>"
    )

    _render_html(chat_container, html_chat, fallback_text=_safe_str(inner_chat))


    # ---------- Pending SQL Approvals (approval_id keyed) ----------
    pending_items = list(runtime.pending_approvals.values())
    if pending_items:
        st.markdown("### 📝 Pending SQL Approvals")
        for item in pending_items:
            approval_id = item.get("id")
            sql = item.get("sql", "")
            sql_card_html = (
                "<div style='border:1px solid #333; background:#0d0d0d; border-radius:10px; padding:12px; margin:8px 0;'>"
                "<div style='color:#bbb; font-size:12px; margin-bottom:6px;'>Approval ID</div>"
                f"<div style='font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Courier New\", monospace;"
                f"            font-size:12px; color:#B0B0B0; white-space:pre-wrap; word-break:break-word;'>{html.escape(str(approval_id))}</div>"
                "<div style='color:#bbb; font-size:12px; margin:8px 0 6px;'>SQL Query</div>"
                "<div style='font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Courier New\", monospace;"
                "            font-size:13px; line-height:1.45; color:#E8E8E8; white-space:pre-wrap; word-break:break-word;'>"
                + _safe_html_text(sql) +
                "</div>"
                "</div>"
            )
            _render_html(st, sql_card_html, fallback_text=f"{approval_id}\n\n{sql}")

            col1, col2 = st.columns(2)
            if col1.button("✅ Approve and Execute", key=f"approve_{approval_id}"):
                try:
                    item_future = item.get("future")
                    if item_future:
                        item_future.set_result({"approved": True})
                    st.success(f"✅ Approved (id={approval_id}). Execution will proceed.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                finally:
                    runtime.pending_approvals.pop(approval_id, None)
                rerun_throttled(0.2)

            if col2.button("❌ Deny", key=f"deny_{approval_id}"):
                try:
                    item_future = item.get("future")
                    if item_future:
                        item_future.set_result({"approved": False})
                except Exception:
                    try:
                        if item_future:
                            item_future.cancel()
                    except Exception:
                        pass
                finally:
                    runtime.pending_approvals.pop(approval_id, None)
                rerun_throttled(0.2)
    else:
        st.info(f"No pending SQL approvals. Pending count: {len(runtime.pending_approvals)}")

    # ---------- Input ----------
    status_ph = st.empty()
    if ss.pending_response:
        _render_html(status_ph, "<div style='margin-bottom:6px;color:#9CDCFE;'>🤖 <em>Thinking…</em></div>", "Thinking…")

    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area("💬 Ask your question:", placeholder="Type your SQL or question here...", key="chat_input", height=80)
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
            _render_html(status_ph, "<div style='margin-bottom:6px;color:#9CDCFE;'>🤖 <em>Thinking…</em></div>", "Thinking…")
            rerun_throttled(0.2)

    # Pull AI response without blocking
    if ss.pending_response:
        try:
            reply = runtime.response_q.get_nowait()
            ss.chat_history.append(("ai", reply))
            ss.pending_response = False
            status_ph.empty()
        except queue.Empty:
            pass

    # ---------- 🧠 REPLAY DRIVER (non-blocking) ----------
    # Advances one step per rerun; pauses on approvals or while waiting for model
    if ss.replay.get("active"):
        if not ss.pending_response and not runtime.pending_approvals:
            prompts = ss.replay["prompts"]
            i = ss.replay["i"]
            total = len(prompts)

            if i >= total:
                set_force_auto_approve(runtime, False)  # back to normal
                st.sidebar.success(f"Replay complete: '{ss.replay.get('name')}'")
                ss.replay["active"] = False
                rerun_throttled(0.2)
            else:
                next_prompt = prompts[i]
                ss.chat_history.append(("user", next_prompt))
                runtime.prompt_q.put(next_prompt)
                ss.pending_response = True
                ss.replay["i"] = i + 1
                pct = (i + 1) / max(1, total)
                st.sidebar.progress(pct, text=f"Replaying {i+1}/{total}: {ss.replay.get('name')}")
                rerun_throttled(0.2)
        else:
            total = len(ss.replay["prompts"])
            i = ss.replay["i"]
            if runtime.pending_approvals:
                st.sidebar.info(f"Replay paused for approval ({i}/{total}). Use buttons below to continue.")
            elif ss.pending_response:
                st.sidebar.info(f"Waiting for model response ({i}/{total})…")


if __name__ == "__main__":
    main()
