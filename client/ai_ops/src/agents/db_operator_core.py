import os
import asyncio
import threading
import queue
import time
import concurrent.futures
from dataclasses import dataclass
import json as _json
import uuid

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentFinish
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import StructuredTool
from langchain.callbacks.base import BaseCallbackHandler
from mcp import StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools

from src.prompt_engineering.topics.db_operator import promt_oracle_db_operator
from src.tools.rag_agent import _rag_agent_service
from src.tools.python_scratchpad import run_python
from src.common.config import *
from src.utils.mcp_helper_connection import safe_connect
from src.llm.oci_genai import initialize_llm


# ─────────────────────────────────────────────
# RUNTIME (framework-agnostic)
# ─────────────────────────────────────────────
class AgentRuntime:
    """Queues/flags for running the agent; UI-agnostic."""
    def __init__(self):
        self.prompt_q = queue.Queue()
        self.response_q = queue.Queue()
        self.log_q = queue.Queue()
        self.control_q = queue.Queue()   # set_history / clear_history
        self.stop_flag = {"stop": False}
        self.trace_logs = []
        self.threads = {}
        # approval_id -> {"id","sql","original_tool","payload","future"}
        self.pending_approvals = {}
        # generic toggles (e.g., force approvals during replay)
        self.flags = {"force_auto_approve": False}


# ─────────────────────────────────────────────
# LOGGING CALLBACK
# ─────────────────────────────────────────────
class ReactLogHandler(BaseCallbackHandler):
    def __init__(self, log_q: queue.Queue):
        self.log_q = log_q

    def _ts(self): return time.strftime('%H:%M:%S')

    def _emit(self, emoji: str, prefix: str, msg):
        try:
            safe = msg if isinstance(msg, str) else _json.dumps(msg, ensure_ascii=False, default=str)
        except Exception:
            safe = str(msg)
        self.log_q.put(f"[{self._ts()}] {emoji} {prefix}: {safe}")

    def _name_from_serialized(self, serialized):
        try:
            if isinstance(serialized, dict):
                if "name" in serialized and isinstance(serialized["name"], str):
                    return serialized["name"]
                if "id" in serialized:
                    sid = serialized["id"]
                    if isinstance(sid, (list, tuple)) and sid:
                        return ".".join(map(str, sid))
                    return str(sid)
                return _json.dumps(serialized, ensure_ascii=False)[:120]
            if isinstance(serialized, (list, tuple)):
                return ".".join(map(str, serialized))
            if serialized is None:
                return "Unknown"
            return str(serialized)
        except Exception:
            return "Unknown"

    # callbacks
    def on_chain_start(self, serialized, inputs, **kwargs):
        self._emit("🔗", "Chain Start", {"name": self._name_from_serialized(serialized), "inputs": inputs})
    def on_chain_end(self, outputs, **kwargs): self._emit("✅", "Chain End", outputs)
    def on_error(self, error, **kwargs): self._emit("💥", "Error", str(error))
    def on_llm_start(self, serialized, prompts, **kwargs):
        name = self._name_from_serialized(serialized)
        if isinstance(prompts, (list, tuple)):
            for p in prompts: self._emit("🧠", f"LLM Prompt ({name})", p)
        else:
            self._emit("🧠", f"LLM Prompt ({name})", prompts)
    def on_llm_end(self, response, **kwargs):
        text = None
        try:
            gens = getattr(response, "generations", None)
            if gens and len(gens) and len(gens[0]) and getattr(gens[0][0], "text", None):
                text = gens[0][0].text
        except Exception:
            pass
        self._emit("💡", "LLM Response", text or "<no generations>")
    def on_tool_start(self, serialized, input_str, **kwargs):
        self._emit("🧩", f"Action: {self._name_from_serialized(serialized)}", input_str)
    def on_tool_end(self, output, **kwargs): self._emit("👁️", "Observation", output)
    def on_agent_action(self, action, **kwargs):
        self._emit("⚙️", "Agent Action", getattr(action, "log", None) or str(action))
    def on_agent_finish(self, finish, **kwargs):
        try:
            val = getattr(finish, "return_values", {}) or {}
        except Exception:
            val = {}
        self._emit("🎯", "Final Answer", val.get("output") or val)


# ─────────────────────────────────────────────
# CONNECTION CONFIG
# ─────────────────────────────────────────────
@dataclass
class ConnectionConfig:
    sqlcl_command: str
    tavily_remote: str
    filesystem_key: str
    redis_host: str
    redis_port: int
    dbtools_host: str
    dbtools_port: int

    def build(self):
        adb_server = StdioServerParameters(command=self.sqlcl_command, args=["-mcp"])
        tavily_server = StdioServerParameters(command="npx", args=["-y", "mcp-remote", self.tavily_remote])
        file_server = StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", self.filesystem_key])
        url_redis = f"{self.redis_host}:{self.redis_port}/mcp"
        url_dbtools = f"{self.dbtools_host}:{self.dbtools_port}/mcp"
        return adb_server, tavily_server, file_server, url_redis, url_dbtools


# ─────────────────────────────────────────────
# HELPERS (agent logic)
# ─────────────────────────────────────────────
def is_sql_tool(tool) -> bool:
    nm = getattr(tool, "name", "") or ""
    return any(kw in nm.lower() for kw in ("adb", "sql", "oracle"))

class RunSQLInput(BaseModel):
    sql: str = Field(default="select sysdate from dual")
    model: str = Field(default="oci/generativeai-chat:2024-05-01")
    sqlcl: str = Field(default="sqlcl")

def _tool_details_safe(tool):
    try: name = getattr(tool, "name", type(tool).__name__)
    except Exception: name = type(tool).__name__
    try: desc = (getattr(tool, "description", "") or "").strip()
    except Exception: desc = ""
    schema_json = None
    try:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema:
            if hasattr(args_schema, "model_json_schema"):
                schema_json = args_schema.model_json_schema()
            elif hasattr(args_schema, "schema"):
                schema_json = args_schema.schema()
            else:
                schema_json = str(args_schema)
    except Exception:
        schema_json = None
    return name, desc, schema_json

def normalize_output(ai_response, history):
    if isinstance(ai_response, AgentFinish):
        msg = ai_response.return_values.get("output")
    elif isinstance(ai_response, dict):
        msg = ai_response.get("output")
    else:
        msg = ai_response
    if isinstance(msg, AIMessage):
        history.append(msg); return msg.content
    elif isinstance(msg, str):
        history.append(AIMessage(content=msg)); return msg
    elif isinstance(msg, dict) and "content" in msg:
        history.append(AIMessage(content=msg["content"])); return msg["content"]
    return "<<no response>>"


# ─────────────────────────────────────────────
# Approval wrapper (respects replay flag)
# ─────────────────────────────────────────────
async def user_confirmed_tool(original_tool, auto_approve: bool, state: AgentRuntime):
    async def wrapper(*args, **kwargs):
        sql_query = kwargs.get("sql", args[0] if args else "select sysdate from dual")
        payload = {"sql": sql_query}

        # Force-approve path (used by replay) OR regular auto-approve
        if state.flags.get("force_auto_approve", False) or auto_approve:
            if hasattr(original_tool, "ainvoke"):
                return await original_tool.ainvoke(payload)
            elif hasattr(original_tool, "invoke"):
                return original_tool.invoke(payload)
            return original_tool.run(**payload)

        # Interactive approval path
        approval_id = f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
        state.log_q.put(f"[{time.strftime('%H:%M:%S')}] ⏳ Awaiting approval id={approval_id} sql={sql_query}")
        future = concurrent.futures.Future()
        state.pending_approvals[approval_id] = {
            "id": approval_id,
            "sql": sql_query,
            "original_tool": original_tool,
            "payload": payload,
            "future": future,
        }
        try:
            decision = await asyncio.wait_for(asyncio.wrap_future(future), timeout=300.0)
        except asyncio.TimeoutError:
            state.pending_approvals.pop(approval_id, None)
            return "⚠️ SQL approval timed out after 5 minutes. Execution denied."
        except concurrent.futures.CancelledError:
            state.pending_approvals.pop(approval_id, None)
            return "⚠️ SQL execution denied by user."

        approved = bool(decision is True or (isinstance(decision, dict) and decision.get("approved")))
        state.pending_approvals.pop(approval_id, None)
        if not approved:
            raise RuntimeError("SQL execution denied by user.")

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


# ─────────────────────────────────────────────
# CONTROL HELPERS (public API)
# ─────────────────────────────────────────────
def set_agent_history(state: AgentRuntime, chat_history_items: list[dict]):
    state.control_q.put({"type": "set_history", "history": chat_history_items})

def clear_agent_history(state: AgentRuntime):
    state.control_q.put({"type": "clear_history"})

def set_force_auto_approve(state: AgentRuntime, on: bool):
    state.flags["force_auto_approve"] = bool(on)


# ─────────────────────────────────────────────
# AGENT LOOP + THREAD HELPERS
# ─────────────────────────────────────────────
async def agent_loop(auto_approve: bool, conn_cfg: ConnectionConfig, state: AgentRuntime):
    log = lambda m: state.log_q.put(f"[{time.strftime('%H:%M:%S')}] {m}")
    try:
        load_dotenv()
        log("🟡 Initializing LLM + MCP servers...")
        model = initialize_llm()

        adb_server, tavily_server, file_server, url_redis, url_dbtools = conn_cfg.build()

        from contextlib import AsyncExitStack
        async with AsyncExitStack() as stack:
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
                if sess:
                    try:
                        tools = await load_mcp_tools(sess)
                        if tools:
                            log(f"🟢 [{name}] Connected ({len(tools)} tools)")
                        else:
                            log(f"🟡 [{name}] Connected but no tools found.")
                        all_tools += tools
                    except Exception as e:
                        log(f"🔴 [{name}] Failed to list tools: {e}")
                else:
                    log(f"🔴 [{name}] Connection failed.")

            wrapped = []
            for t in all_tools:
                if is_sql_tool(t):
                    wrapped.append(await user_confirmed_tool(t, auto_approve, state))
                else:
                    wrapped.append(t)

            tools_final = wrapped + [run_python, _rag_agent_service]
            for extra in [run_python, _rag_agent_service]:
                name, desc, _schema = _tool_details_safe(extra)
                log(f"🧩 Added built-in tool: {name} — {desc or '<no description>'}")

            agent = initialize_agent(
                tools=tools_final,
                llm=model,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=True,
                agent_kwargs={"prefix": promt_oracle_db_operator},
                callbacks=[ReactLogHandler(state.log_q)],
            )

            history = []
            log("🚀 Agent loop started.")

            while not state.stop_flag["stop"]:
                # control channel
                try:
                    while True:
                        ctrl = state.control_q.get_nowait()
                        if not isinstance(ctrl, dict):
                            continue
                        ctype = ctrl.get("type")
                        if ctype == "clear_history":
                            history.clear()
                            log("🧽 History cleared via control channel.")
                        elif ctype == "set_history":
                            raw = ctrl.get("history") or []
                            new_hist = []
                            for item in raw:
                                role = (item.get("role") or "").lower()
                                content = item.get("content") or ""
                                if role == "user":
                                    new_hist.append(HumanMessage(content=content))
                                elif role == "ai":
                                    new_hist.append(AIMessage(content=content))
                            history = new_hist
                            log(f"📌 History replaced from checkpoint (len={len(history)}).")
                except queue.Empty:
                    pass

                # user prompts
                try:
                    prompt = state.prompt_q.get(timeout=0.5)
                    log(f"🟡 Received: {prompt}")
                    if str(prompt).strip().lower() in {"clear memory", "clear history", "reset", "reset memory", "/clear", "/reset"}:
                        history.clear()
                        state.response_q.put("🧠 Memory cleared. Conversation context reset.")
                        log("🧽 Cleared agent conversation history on user request.")
                        continue

                    history.append(HumanMessage(content=prompt))
                    ai_resp = await agent.ainvoke({"input": history})
                    msg = normalize_output(ai_resp, history)
                    state.response_q.put(msg)
                    log("🟢 Response generated.")
                except queue.Empty:
                    await asyncio.sleep(0.1)
                except Exception as e:
                    log(f"🔴 Error: {e}")
                    state.response_q.put(f"❌ {e}")

            log("🟠 Agent shutting down.")
    except Exception as e:
        state.log_q.put(f"[{time.strftime('%H:%M:%S')}] 💥 Crash: {e}")
        state.response_q.put(f"❌ {e}")


def start_agent_thread(auto_approve: bool, conn_cfg: ConnectionConfig, state: AgentRuntime):
    if "agent" in state.threads and state.threads["agent"].is_alive():
        return
    state.stop_flag["stop"] = False
    t = threading.Thread(target=lambda: asyncio.run(agent_loop(auto_approve, conn_cfg, state)), daemon=True)
    t.start()
    state.threads["agent"] = t


def start_log_stream(state: AgentRuntime):
    if "stream" in state.threads and state.threads["stream"].is_alive():
        return

    MAX_LOGS = 5000
    def pump():
        while not state.stop_flag["stop"]:
            try:
                msg = state.log_q.get(timeout=0.2)
                state.trace_logs.append(str(msg))
                while True:
                    try:
                        state.trace_logs.append(str(state.log_q.get_nowait()))
                    except queue.Empty:
                        break
                if len(state.trace_logs) > MAX_LOGS:
                    del state.trace_logs[: len(state.trace_logs) - MAX_LOGS]
            except queue.Empty:
                continue
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    state.threads["stream"] = t
