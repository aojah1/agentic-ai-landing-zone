import os
import asyncio
import threading
import queue
import time
from contextlib import AsyncExitStack
import concurrent.futures

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentFinish
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import StructuredTool
from mcp import StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools

from src.llm.oci_genai import initialize_llm
from src.prompt_engineering.topics.db_operator import promt_oracle_db_operator
from src.tools.rag_agent import _rag_agent_service
from src.tools.python_scratchpad import run_python
from src.common.config import *
from src.utils.mcp_helper_connection import safe_connect

from langchain.callbacks.base import BaseCallbackHandler
import time, html, json


class ReactLogHandler(BaseCallbackHandler):
    """Stream real-time logs to a queue; resilient to None/variant payloads."""
    def __init__(self, log_q):
        self.log_q = log_q

    # -------- helpers --------
    def _ts(self):
        return time.strftime('%H:%M:%S')

    def _emit(self, emoji: str, prefix: str, msg: str):
        try:
            safe = msg if isinstance(msg, str) else json.dumps(msg, ensure_ascii=False)
        except Exception:
            safe = str(msg)
        self.log_q.put(f"[{self._ts()}] {emoji} {prefix}: {safe}")

    def _name_from_serialized(self, serialized):
        # serialized can be None, str, dict, tuple...
        try:
            if isinstance(serialized, dict):
                # langchain often passes {"name": "..."} OR {"id": ["chain", "AgentExecutor"]}
                if "name" in serialized and isinstance(serialized["name"], str):
                    return serialized["name"]
                if "id" in serialized:
                    sid = serialized["id"]
                    if isinstance(sid, (list, tuple)) and sid:
                        return ".".join(map(str, sid))
                    return str(sid)
                return json.dumps(serialized, ensure_ascii=False)[:120]
            if isinstance(serialized, (list, tuple)):
                return ".".join(map(str, serialized))
            if serialized is None:
                return "Unknown"
            return str(serialized)
        except Exception:
            return "Unknown"

    # -------- chain lifecycle --------
    def on_chain_start(self, serialized, inputs, **kwargs):
        name = self._name_from_serialized(serialized)
        # inputs can be dict/str/None
        self._emit("🔗", "Chain Start", {"name": name, "inputs": inputs})

    def on_chain_end(self, outputs, **kwargs):
        self._emit("✅", "Chain End", outputs)

    def on_error(self, error, **kwargs):
        self._emit("💥", "Error", str(error))

    # -------- llm --------
    def on_llm_start(self, serialized, prompts, **kwargs):
        name = self._name_from_serialized(serialized)
        if isinstance(prompts, (list, tuple)):
            for p in prompts:
                self._emit("🧠", f"LLM Prompt ({name})", p)
        else:
            self._emit("🧠", f"LLM Prompt ({name})", prompts)

    def on_llm_end(self, response, **kwargs):
        # response.generations can be missing/empty
        text = None
        try:
            gens = getattr(response, "generations", None)
            if gens and len(gens) and len(gens[0]) and getattr(gens[0][0], "text", None):
                text = gens[0][0].text
        except Exception:
            pass
        self._emit("💡", "LLM Response", text or "<no generations>")

    # -------- tools --------
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = self._name_from_serialized(serialized)
        # input_str may be dict/str/None
        self._emit("🧩", f"Action: {name}", input_str)

    def on_tool_end(self, output, **kwargs):
        self._emit("👁️", "Observation", output)

    # -------- agent --------
    def on_agent_action(self, action, **kwargs):
        # action.log may be None
        log_text = getattr(action, "log", None)
        self._emit("⚙️", "Agent Action", log_text or str(action))

    def on_agent_finish(self, finish, **kwargs):
        try:
            val = getattr(finish, "return_values", {}) or {}
        except Exception:
            val = {}
        self._emit("🎯", "Final Answer", val.get("output") or val)


# ─────────────────────────────────────────────
# GLOBAL STATE (persistent across reruns)
# ─────────────────────────────────────────────
class GlobalState:
    def __init__(self):
        self.prompt_q = queue.Queue()
        self.response_q = queue.Queue()
        self.log_q = queue.Queue()
        self.stop_flag = {"stop": False}
        self.trace_logs = []
        self.threads = {}
        self.pending_approvals = {}  # sql -> {"original_tool": tool, "payload": payload, "future": future}

if "global_state" not in st.session_state:
    st.session_state.global_state = GlobalState()
_global_state = st.session_state.global_state

st.set_page_config(page_title="Oracle DB Operator", layout="wide")
load_dotenv()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def is_sql_tool(tool) -> bool:
    nm = getattr(tool, "name", "") or ""
    return any(kw in nm.lower() for kw in ("adb", "sql", "oracle"))

class RunSQLInput(BaseModel):
    sql: str = Field(default="select sysdate from dual")
    model: str = Field(default="oci/generativeai-chat:2024-05-01")
    sqlcl: str = Field(default="sqlcl")

# ====== CHANGED: approval happens in UI; tool execution stays in agent loop ======
async def user_confirmed_tool(original_tool, auto_approve):
    async def wrapper(*args, **kwargs):
        sql_query = kwargs.get("sql", args[0] if args else "select sysdate from dual")
        payload = {"sql": sql_query}
        if not auto_approve:
            _global_state.log_q.put(f"[DEBUG] Queuing SQL for approval: {sql_query}")
            future = concurrent.futures.Future()
            _global_state.pending_approvals[sql_query] = {
                "original_tool": original_tool,
                "payload": payload,
                "future": future,
            }
            try:
                decision = await asyncio.wait_for(asyncio.wrap_future(future), timeout=300.0)
            except asyncio.TimeoutError:
                if sql_query in _global_state.pending_approvals:
                    del _global_state.pending_approvals[sql_query]
                return "⚠️ SQL approval timed out after 5 minutes. Execution denied."
            except concurrent.futures.CancelledError:
                return "⚠️ SQL execution denied by user."

            approved = bool(decision is True or (isinstance(decision, dict) and decision.get("approved")))
            if not approved:
                return "❌ SQL execution denied."

            # Execute tool HERE (inside the same event loop as the agent)
            if hasattr(original_tool, "ainvoke"):
                return await original_tool.ainvoke(payload)
            elif hasattr(original_tool, "invoke"):
                return original_tool.invoke(payload)
            return original_tool.run(**payload)
        else:
            if hasattr(original_tool, "ainvoke"):
                return await original_tool.ainvoke(payload)
            elif hasattr(original_tool, "invoke"):
                return original_tool.invoke(payload)
            return original_tool.run(**payload)

    return StructuredTool(
        name=original_tool.name,
        description=original_tool.description or "SQL tool",
        args_schema=RunSQLInput,
        coroutine=wrapper,
    )

def normalize_output(ai_response, history):
    if isinstance(ai_response, AgentFinish):
        msg = ai_response.return_values.get("output")
    elif isinstance(ai_response, dict):
        msg = ai_response.get("output")
    else:
        msg = ai_response
    if isinstance(msg, AIMessage):
        history.append(msg)
        return msg.content
    elif isinstance(msg, str):
        history.append(AIMessage(content=msg))
        return msg
    elif isinstance(msg, dict) and "content" in msg:
        history.append(AIMessage(content=msg["content"]))
        return msg["content"]
    return "<<no response>>"


import re, html

# Hide internal traces (Thought/Action/Observation/etc.) from the AI panel
def _clean_ai_text(text: str) -> str:
    if not text:
        return ""
    keep = []
    for line in text.splitlines():
        if re.match(r'^\s*(Thought|Action|Observation|Agent Action|Tool|Reasoning)\b', line, re.I):
            continue
        keep.append(line)
    out = "\n".join(keep).strip()
    # If there's an explicit "Final Answer:" label, surface just that portion
    m = re.search(r'(?is)Final Answer:\s*(.+)$', out)
    return m.group(1).strip() if m else out

# Minimal markdown→HTML safe-ish rendering:
# - escape HTML
# - support ``` code fences
# - preserve newlines
def _md_to_html(md_text: str) -> str:
    if not md_text:
        return ""
    # Escape first
    esc = html.escape(md_text)

    # Restore fenced code blocks to <pre><code>...</code></pre>
    def repl(m):
        code = m.group(2)
        return f"<pre style='margin:8px 0;padding:10px;background:#0d0d0d;border-radius:6px;overflow:auto;'><code>{html.escape(code)}</code></pre>"

    esc = re.sub(r"```(\w+)?\n(.*?)```", repl, esc, flags=re.S)

    # Inline code `...`
    esc = re.sub(r"`([^`]+)`", r"<code style='background:#0d0d0d;padding:2px 4px;border-radius:4px;'>\1</code>", esc)

    # Basic bullets
    esc = re.sub(r"(?m)^- (.+)$", r"• \1", esc)

    # Newlines → <br>
    esc = esc.replace("\n", "<br>")
    return esc


# ─────────────────────────────────────────────
# ASYNC AGENT LOOP
# ─────────────────────────────────────────────
async def agent_loop(auto_approve):
    log = lambda m: _global_state.log_q.put(f"[{time.strftime('%H:%M:%S')}] {m}")
    try:
        log("🟡 Initializing LLM + MCP servers...")
        model = initialize_llm()

        adb_server = StdioServerParameters(command=SQLCLI_MCP_PROFILE, args=["-mcp"])
        tavily_server = StdioServerParameters(command="npx", args=["-y", "mcp-remote", TAVILY_MCP_SERVER])
        file_server = StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", FILE_SYSTEM_ACCESS_KEY])
        url_redis = f"{MCP_SSE_HOST}:{MCP_SSE_PORT}/mcp"
        url_dbtools = f"{MCP_SSE_HOST_DBTOOLS}:{MCP_SSE_PORT_DBTOOLS}/mcp"

        async with AsyncExitStack() as stack:
            start_t = time.time()
            adb_sess = await safe_connect("Oracle SQLcl", adb_server, stack)
            tav_sess = await safe_connect("Tavily", tavily_server, stack)
            file_sess = await safe_connect("File Server", file_server, stack)
            red_sess = await safe_connect("Redis", url_redis, stack, http_streamable=True)
            dbt_sess = await safe_connect("DBTools", url_dbtools, stack, http_streamable=True)

            all_tools = []
            for name, sess in [
                ("SQLcl", adb_sess),
                ("Tavily", tav_sess),
                ("File Server", file_sess),
                ("Redis", red_sess),
                ("DBTools", dbt_sess),
            ]:
                t0 = time.time()
                if sess:
                    try:
                        tools = await load_mcp_tools(sess)
                        latency = int((time.time() - t0) * 1000)
                        if tools:
                            log(f"🟢 [{name}] Connected in {latency} ms ({len(tools)} tools):")
                            for t in tools:
                                t_name = getattr(t, 'name', '<unknown>')
                                t_desc = getattr(t, 'description', '') or ''
                                short_desc = (t_desc[:60] + '...') if len(t_desc) > 60 else t_desc
                                log(f"    • 🧩 {t_name} — {short_desc}")
                        else:
                            log(f"🟡 [{name}] Connected but no tools found.")
                        all_tools += tools
                    except Exception as e:
                        log(f"🔴 [{name}] Failed to list tools: {e}")
                else:
                    log(f"🔴 [{name}] Connection failed.")

            tools_final = [await user_confirmed_tool(t, auto_approve) if is_sql_tool(t) else t for t in all_tools]
            tools_final += [run_python, _rag_agent_service]

            agent = initialize_agent(
                tools=tools_final,
                llm=model,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=True,
                agent_kwargs={"prefix": promt_oracle_db_operator},
                callbacks=[ReactLogHandler(_global_state.log_q)],  # ✅ Added here
            )

            history = []
            log("🚀 Agent loop started.")

            while not _global_state.stop_flag["stop"]:
                try:
                    prompt = _global_state.prompt_q.get(timeout=0.5)
                    log(f"🟡 Received: {prompt}")
                    history.append(HumanMessage(content=prompt))
                    ai_resp = await agent.ainvoke({"input": history})
                    msg = normalize_output(ai_resp, history)
                    _global_state.response_q.put(msg)
                    log("🟢 Response generated.")
                except queue.Empty:
                    await asyncio.sleep(0.1)
                except Exception as e:
                    log(f"🔴 Error: {e}")

            log("🟠 Agent shutting down.")
    except Exception as e:
        log(f"💥 Crash: {e}")

# ─────────────────────────────────────────────
# THREADING
# ─────────────────────────────────────────────
def start_agent_thread(auto_approve):
    if "agent" in _global_state.threads and _global_state.threads["agent"].is_alive():
        return
    _global_state.stop_flag["stop"] = False
    t = threading.Thread(target=lambda: asyncio.run(agent_loop(auto_approve)), daemon=True)
    t.start()
    _global_state.threads["agent"] = t

def start_log_stream():
    """Ultra-low latency log pump: blocks on .get() and never sleeps."""
    if "stream" in _global_state.threads and _global_state.threads["stream"].is_alive():
        return

    # Optional cap to avoid unbounded growth
    MAX_LOGS = 5000

    def pump():
        while not _global_state.stop_flag["stop"]:
            try:
                # Block briefly for new log; as soon as one arrives, drain the rest
                msg = _global_state.log_q.get(timeout=0.2)
                _global_state.trace_logs.append(msg)

                # Drain burst without sleeping
                while True:
                    try:
                        _global_state.trace_logs.append(_global_state.log_q.get_nowait())
                    except queue.Empty:
                        break

                # Trim if over cap
                if len(_global_state.trace_logs) > MAX_LOGS:
                    del _global_state.trace_logs[: len(_global_state.trace_logs) - MAX_LOGS]

            except queue.Empty:
                # no log this tick; loop back immediately
                continue

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    _global_state.threads["stream"] = t


# ─────────────────────────────────────────────
# STREAMLIT UI — ChatGPT style
# ─────────────────────────────────────────────
def main():
    import time

    st.title("Oracle DB Operator 💬")
    st.caption("Conversational Agent for Oracle MCP Servers")

    # ---------- Session ----------
    ss = st.session_state
    ss.setdefault("busy", False)
    ss.setdefault("suppress_refresh", False)
    ss.setdefault("chat_history", [])
    ss.setdefault("chat_input", "")
    ss.setdefault("pending_response", False)  # Track if waiting for AI reply

    # Clear textarea state BEFORE the widget is created (set last turn)
    if ss.get("_clear_chat_input", False):
        ss._clear_chat_input = False
        ss.pop("chat_input", None)

    auto_approve = st.sidebar.checkbox("Auto-approve SQL executions", value=False)
    agent_running = ("agent" in _global_state.threads) and _global_state.threads["agent"].is_alive()
    st.sidebar.markdown(f"**Status:** {'🟢 Running' if agent_running else '🔴 Stopped'}")

    # Test button to add a fake pending SQL (for debugging)
    if st.sidebar.button("🧪 Test Add Pending SQL"):
        from langchain_core.tools import Tool
        fake_tool = Tool(name="test_sql", description="Fake SQL tool", func=lambda x: "Fake result")
        future = concurrent.futures.Future()
        _global_state.pending_approvals["SELECT 1 FROM DUAL;"] = {"original_tool": fake_tool, "payload": {"sql": "SELECT 1 FROM DUAL;"}, "future": future}
        st.info("Added test SQL for approval.")
        st.rerun()

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ Start Agent"):
        _global_state.trace_logs.append(f"[{time.strftime('%H:%M:%S')}] 🚀 Starting MCP Agent…")
        with st.spinner("Starting MCP Agent..."):
            start_agent_thread(auto_approve)
            start_log_stream()
            st.success("✅ Agent started")
        # drain initial logs briefly
        t_end = time.time() + 1.0
        while time.time() < t_end:
            drained = False
            try:
                while not _global_state.log_q.empty():
                    _global_state.trace_logs.append(_global_state.log_q.get_nowait())
                    drained = True
            except queue.Empty:
                pass
            if not drained:
                time.sleep(0.05)
        st.rerun()

    if c2.button("🛑 Stop Agent"):
        _global_state.stop_flag["stop"] = True
        st.warning("Stopping agent...")

    # ---------- Logs ----------
    st.subheader("📜 Agent Logs (Live)")
    if st.button("🧹 Clear Logs"):
        _global_state.trace_logs.clear()

    try:
        while not _global_state.log_q.empty():
            _global_state.trace_logs.append(_global_state.log_q.get_nowait())
    except queue.Empty:
        pass

    log_text = "\n".join(_global_state.trace_logs[-1500:]) or "No logs yet..."
    st.markdown(
        f"""
        <div style="background-color:#111; color:#EEE; padding:10px;
             border-radius:8px; height:300px; overflow-y:auto;
             font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Courier New', monospace;
             font-size:13px; line-height:1.35;">
            {log_text.replace('\n', '<br>')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Conversation (render via placeholder) ----------
    st.markdown("#### 💬 Conversation")
    chat_ph = st.empty()

    # session guards
    ss.setdefault("_chat_hash", None)
    ss.setdefault("_chat_rendered_once", False)

    def _render_chat(always: bool = False):
        """Render the chat panel reliably.
        First render is forced; later renders only when content changes.
        """
        # compute hash
        try:
            new_hash = _hash_chat(ss.chat_history)
        except Exception:
            new_hash = str(time.time())  # fallback

        # decide if we should render
        should_render = always or (new_hash != ss._chat_hash) or (not ss._chat_rendered_once)
        if not should_render:
            return

        ss._chat_hash = new_hash
        ss._chat_rendered_once = True

        # build bubbles
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

        # no f-strings/triple quotes to avoid parser issues
        html_block = (
            "<div id='chatbox' style='background:#111; padding:10px; border-radius:8px; "
            "height:300px; overflow-y:auto;'>"
            + ("".join(bubbles) if bubbles else "<i>No conversation yet...</i>")
            + "</div>"
            + "<script>(function(){var cb=document.getElementById('chatbox');"
            "if(cb){cb.scrollTop=cb.scrollHeight;}})();</script>"
        )

        chat_ph.markdown(html_block, unsafe_allow_html=True)

    # force initial render once per page load
    _render_chat(always=True)


    # ---------- Pending SQL Approvals ----------
    pending_sqls = list(_global_state.pending_approvals.keys())
    if pending_sqls:
        st.markdown("### Pending SQL Approvals")
        for sql in pending_sqls:
            item = _global_state.pending_approvals[sql]
            st.write(f"**SQL Query:** {sql}")
            col1, col2 = st.columns(2)

            # ====== CHANGED: only signal approval; do NOT execute the tool here ======
            if col1.button("✅ Approve and Execute", key=f"approve_{hash(sql)}"):
                _global_state.log_q.put(f"[DEBUG] Approving SQL: {sql}")
                try:
                    item['future'].set_result({"approved": True})
                    st.success("✅ Approved. Execution will proceed.")
                    _global_state.log_q.put(f"[DEBUG] Future approval set for {sql}")
                except Exception as e:
                    _global_state.log_q.put(f"[DEBUG] Approval exception: {e}")
                    st.error(f"❌ Error: {e}")
                finally:
                    if sql in _global_state.pending_approvals:
                        del _global_state.pending_approvals[sql]
                st.rerun()

            # ====== CHANGED: only signal denial ======
            if col2.button("❌ Deny", key=f"deny_{hash(sql)}"):
                _global_state.log_q.put(f"[DEBUG] Denying SQL: {sql}")
                try:
                    item['future'].set_result({"approved": False})
                except Exception:
                    try:
                        item['future'].cancel()
                    except Exception:
                        pass
                finally:
                    if sql in _global_state.pending_approvals:
                        del _global_state.pending_approvals[sql]
                st.rerun()
    else:
        st.info(f"No pending SQL approvals. Pending count: {len(_global_state.pending_approvals)}")

    # ---------- Input (single-click submit) ----------
    status_ph = st.empty()  # thinking line above button

    # Show "thinking" if a response is pending
    if ss.pending_response:
        status_ph.markdown(
            "<div style='margin-bottom:6px;color:#9CDCFE;'>🤖 <em>Thinking…</em></div>",
            unsafe_allow_html=True,
        )

    with st.form("chat_form", clear_on_submit=False):
        user_query = st.text_area(
            "💬 Ask your question:",
            placeholder="Type your SQL or question here...",
            key="chat_input",
            height=80,
        )
        submitted = st.form_submit_button("🚀 Run", use_container_width=True)

    if submitted:
        q = (user_query or "").strip()
        agent_running_now = ("agent" in _global_state.threads) and _global_state.threads["agent"].is_alive()

        if not q:
            st.warning("Please enter a question.")
        elif not agent_running_now:
            st.error("Agent not running. Please start it first.")
        else:
            # Enqueue the prompt
            _global_state.prompt_q.put(q)
            # Append user message to history
            ss.chat_history.append(("user", q))
            # Set pending flag and thinking status
            ss.pending_response = True
            status_ph.markdown(
                "<div style='margin-bottom:6px;color:#9CDCFE;'>🤖 <em>Thinking…</em></div>",
                unsafe_allow_html=True,
            )
            # Clear textarea and rerun
            ss._clear_chat_input = True
            st.rerun()

    # ---------- Check for AI response (on every rerun) ----------
    if ss.pending_response:
        try:
            reply = _global_state.response_q.get_nowait()
            ss.chat_history.append(("ai", reply))
            ss.pending_response = False  # Clear pending flag
            status_ph.empty()  # Clear thinking status
            _render_chat()  # Update chat immediately
            st.rerun()  # Rerun to ensure UI updates
        except queue.Empty:
            pass  # No response yet; continue

    # ---------- Auto-refresh logs ----------
    try:
        from streamlit_autorefresh import st_autorefresh
        if not ss.busy and not ss.suppress_refresh:
            st_autorefresh(interval=120, key="live_log_refresh")
    except Exception:
        pass

# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
