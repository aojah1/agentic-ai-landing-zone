import os
import asyncio
import threading
import queue
import time
from contextlib import AsyncExitStack

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

class ReactLogHandler(BaseCallbackHandler):
    """Stream real-time chain-of-thought logs to the log queue."""
    def __init__(self, log_q):
        self.log_q = log_q

    def _push(self, prefix, msg, emoji="💭"):
        ts = time.strftime('%H:%M:%S')
        self.log_q.put(f"[{ts}] {emoji} {prefix}: {msg}")

    # ========== AGENT LIFE CYCLE ========== #
    def on_chain_start(self, serialized, inputs, **kwargs):
        name = serialized.get("name", "Chain")
        self._push("Chain Start", f"{name} | Inputs: {inputs}", emoji="🔗")

    def on_llm_start(self, serialized, prompts, **kwargs):
        for p in prompts:
            self._push("LLM Prompt", p.strip(), emoji="🧠")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Unknown Tool")
        self._push("Action", f"{tool_name} | Input: {input_str}", emoji="🧩")

    def on_tool_end(self, output, **kwargs):
        self._push("Observation", str(output)[:500], emoji="👁️")

    def on_llm_end(self, response, **kwargs):
        if response and hasattr(response, "generations"):
            text = response.generations[0][0].text[:400]
            self._push("LLM Response", text, emoji="💡")

    def on_chain_end(self, outputs, **kwargs):
        self._push("Chain End", f"Outputs: {outputs}", emoji="✅")

    def on_agent_action(self, action, **kwargs):
        self._push("Agent Action", str(action.log).strip()[:500], emoji="⚙️")

    def on_agent_finish(self, finish, **kwargs):
        self._push("Final Answer", finish.return_values.get("output", ""), emoji="🎯")

    def on_error(self, error, **kwargs):
        self._push("Error", str(error), emoji="💥")


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

def user_confirmed_tool(tool, auto_approve: bool):
    async def wrapper(*args, **kwargs):
        sql_query = kwargs.get("sql", args[0] if args else "select sysdate from dual")
        payload = {"sql": sql_query}
        try:
            if not auto_approve:
                if not st.sidebar.button(f"Allow SQL: {sql_query}"):
                    return "⚠️ Execution cancelled by user."
            if hasattr(tool, "ainvoke"):
                return await tool.ainvoke(payload)
            elif hasattr(tool, "invoke"):
                return tool.invoke(payload)
            return tool.run(**payload)
        except Exception as e:
            return f"ERROR executing SQLcl tool: {e}"
    return StructuredTool(name=tool.name, description=tool.description or "SQL tool", args_schema=RunSQLInput, coroutine=wrapper)

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

            tools_final = [user_confirmed_tool(t, auto_approve) if is_sql_tool(t) else t for t in all_tools]
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
    if "stream" in _global_state.threads and _global_state.threads["stream"].is_alive():
        return
    def pump():
        while not _global_state.stop_flag["stop"]:
            try:
                msg = _global_state.log_q.get(timeout=0.5)
                _global_state.trace_logs.append(msg)
            except queue.Empty:
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

    # Track busy state (used to pause any optional auto-refresh logic you might add)
    if "busy" not in st.session_state:
        st.session_state.busy = False

    auto_approve = st.sidebar.checkbox("Auto-approve SQL executions", value=False)
    agent_running = (
        "agent" in _global_state.threads
        and _global_state.threads["agent"].is_alive()
    )
    st.sidebar.markdown(f"**Status:** {'🟢 Running' if agent_running else '🔴 Stopped'}")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("▶️ Start Agent"):
        # Seed an immediate log line so panel shows something right away
        _global_state.trace_logs.append(f"[{time.strftime('%H:%M:%S')}] 🚀 Starting MCP Agent…")

        with st.spinner("Starting MCP Agent..."):
            start_agent_thread(auto_approve)
            start_log_stream()
            st.success("✅ Agent started")

        # Drain initial logs briefly so they render on the same click
        start_deadline = time.time() + 1.0
        while time.time() < start_deadline:
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

    if col2.button("🛑 Stop Agent"):
        _global_state.stop_flag["stop"] = True
        st.warning("Stopping agent...")

    # Persistent chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ───────────── LOGS AT TOP (dark, native) ─────────────
    st.subheader("📜 Agent Logs (Live)")

    if st.button("🧹 Clear Logs"):
        _global_state.trace_logs.clear()

    # Background collector to move log_q → trace_logs (idempotent)
    def collect_logs():
        while not _global_state.stop_flag["stop"]:
            try:
                while not _global_state.log_q.empty():
                    _global_state.trace_logs.append(_global_state.log_q.get_nowait())
            except queue.Empty:
                pass
            time.sleep(0.2)

    if (
        "log_collector" not in _global_state.threads
        or not _global_state.threads["log_collector"].is_alive()
    ):
        t = threading.Thread(target=collect_logs, daemon=True)
        t.start()
        _global_state.threads["log_collector"] = t

    # Render logs (dark block)
    log_text = "\n".join(_global_state.trace_logs[-1200:]) or "No logs yet..."
    st.markdown(
        f"""
        <div style="background-color:#111; color:#EEE; padding:10px;
             border-radius:8px; height:300px; overflow-y:auto;
             font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
             font-size:13px; line-height:1.35;">
            {log_text.replace('\n', '<br>')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ───────────── CONVERSATION (scrollable) ─────────────
    st.markdown("---")
    st.markdown("### 💬 Conversation")

    chat_html = ""
    for role, content in st.session_state.chat_history:
        bg = "#2a2a2a" if role == "user" else "#1e1e1e"
        fg = "#fff" if role == "user" else "#9CDCFE"
        prefix = "👤 You:" if role == "user" else "🤖 AI:"
        chat_html += f"""
        <div style='background:{bg};color:{fg};padding:10px;border-radius:8px;margin-bottom:8px;'>
            <b>{prefix}</b><br>{content}
        </div>"""

    # Scrollable container WITH auto-scroll to the bottom
    st.markdown(
        f"""
        <div id="chatbox" style="background:#111; padding:10px; border-radius:8px;
             height:400px; overflow-y:auto;">
            {chat_html or '<i>No conversation yet...</i>'}
        </div>
        <script>
        var cb = document.getElementById('chatbox');
        if (cb) {{ cb.scrollTop = cb.scrollHeight; }}
        </script>
        """,
        unsafe_allow_html=True,
    )

    # ───────────── INPUT + SINGLE-CLICK RUN FIX ─────────────
    user_query = st.text_area(
        "💬 Ask your question:",
        placeholder="Type your SQL or question here...",
        key="chat_input",
        height=80,
    )

    # Click stores the query in state; processing happens after rerun (below)
    run_clicked = st.button("🚀 Run", use_container_width=True, key="run_btn")
    if run_clicked and user_query.strip():
        st.session_state["pending_query"] = user_query

    # ───────────── PROCESS PENDING QUERY (single-click behavior) ─────────────
    if "pending_query" in st.session_state and not st.session_state.busy:
        pq = st.session_state.pop("pending_query")

        if not ("agent" in _global_state.threads and _global_state.threads["agent"].is_alive()):
            st.error("Agent not running. Please start it first.")
        else:
            _global_state.prompt_q.put(pq)
            st.session_state.chat_history.append(("user", pq))
            st.session_state.busy = True

            with st.spinner("🤖 Thinking..."):
                try:
                    reply = _global_state.response_q.get(timeout=120)
                    st.session_state.chat_history.append(("ai", reply))
                    st.success("✅ Response received!")
                except queue.Empty:
                    st.error("⏱️ Timeout waiting for agent response.")
            st.session_state.busy = False
            st.rerun()



# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
