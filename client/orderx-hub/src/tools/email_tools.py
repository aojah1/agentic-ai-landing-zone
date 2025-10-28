# Enable an Agent to write and send an email to a human receipt

"""
email_tool.py — SMTP email sender as an Agent Tool

Works in three modes (auto-detected):
  1) **LangChain tool** (@tool) — function name: `send_email_tool`
  2) **MCP tool** (@mcp.tool) — function name: `send_email_mcp`
  3) Plain Python function — call `send_email(payload: EmailPayload)` directly

Auth: Basic username/password SMTP (e.g., Gmail with App Password).

Environment variables (required):
  SMTP_HOST          e.g., "smtp.gmail.com"
  SMTP_PORT          e.g., "587"
  SMTP_USERNAME      e.g., "me@example.com"
  SMTP_PASSWORD      e.g., "app-password-or-smtp-pass"
  SMTP_USE_TLS       optional, default "true" (starttls)
  SMTP_FROM          optional, default SMTP_USERNAME

Attachment paths are read from local filesystem.

Return value: JSON-serializable dict with keys: status, message_id, provider, to, cc, bcc, attachments

Example (plain Python):
    from email_tool import EmailPayload, send_email
    payload = EmailPayload(
        to=["recipient@example.com"],
        subject="Test from tool",
        body="Hello from agent",
        is_html=False,
        cc=[], bcc=[], attachments=[]
    )
    print(send_email(payload))

Example (LangChain):
    from langchain.agents import initialize_agent, AgentType
    from langchain_openai import ChatOpenAI
    from email_tool import send_email_tool

    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [send_email_tool]
    agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)

    agent.invoke({
        "input": "Send an email to ops@example.com that says 'Done.' with subject 'Run complete'"
    })

Example (MCP):
  Register this module with your MCP server loader; tool name will be `send_email`.
"""

from __future__ import annotations

import base64
import json
import os
import smtplib
import ssl
import time
import uuid
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import make_msgid, formatdate
from pathlib import Path
from typing import List, Optional

# --------- Config ---------

def _get_bool_env(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_smtp_config():
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    use_tls = _get_bool_env("SMTP_USE_TLS", True)
    from_addr = os.getenv("SMTP_FROM", username)

    missing = [k for k, v in {
        "SMTP_HOST": host,
        "SMTP_USERNAME": username,
        "SMTP_PASSWORD": password,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env: {', '.join(missing)}")

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "use_tls": use_tls,
        "from_addr": from_addr,
    }

# --------- Model ---------

@dataclass
class EmailPayload:
    to: List[str]
    subject: str
    body: str
    is_html: bool = False
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.to:
            raise ValueError("`to` must contain at least one recipient")
        if not self.subject:
            raise ValueError("`subject` must be provided")
        if self.attachments:
            for p in self.attachments:
                if not Path(p).exists():
                    raise FileNotFoundError(f"Attachment not found: {p}")

# --------- Core send function ---------

def _build_message(cfg: dict, payload: EmailPayload) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = cfg["from_addr"]
    msg["To"] = ", ".join(payload.to)
    if payload.cc:
        msg["Cc"] = ", ".join(payload.cc)
    msg["Subject"] = payload.subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=cfg["host"])  # stable-ish ID

    if payload.is_html:
        msg.set_content("This is an HTML email. Your client does not support HTML.")
        msg.add_alternative(payload.body, subtype="html")
    else:
        msg.set_content(payload.body)

    for path in payload.attachments:
        file_path = Path(path)
        with file_path.open("rb") as f:
            data = f.read()
        # naive type detection
        maintype, subtype = ("application", "octet-stream")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=file_path.name)

    return msg


def send_email(payload: EmailPayload) -> dict:
    """Send email via SMTP. Returns JSON-serializable dict."""
    payload.validate()
    cfg = _load_smtp_config()

    msg = _build_message(cfg, payload)

    start = time.time()
    context = ssl.create_default_context()
    message_id = msg.get("Message-ID") or f"<{uuid.uuid4()}@local>"

    if cfg["use_tls"]:
        with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(cfg["username"], cfg["password"])
            server.send_message(msg, to_addrs=(payload.to + payload.cc + payload.bcc))
    else:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context) as server:
            server.login(cfg["username"], cfg["password"])
            server.send_message(msg, to_addrs=(payload.to + payload.cc + payload.bcc))

    elapsed_ms = int((time.time() - start) * 1000)
    return {
        "status": "sent",
        "provider": "smtp",
        "elapsed_ms": elapsed_ms,
        "message_id": message_id,
        "to": payload.to,
        "cc": payload.cc,
        "bcc": payload.bcc,
        "attachments": [Path(p).name for p in payload.attachments],
    }

# --------- LangChain tool wrapper (optional) ---------
try:
    # new-style import first
    from langchain_core.tools import tool as lc_tool  # type: ignore
except Exception:  # pragma: no cover - fallback for older LC
    try:
        from langchain.tools import tool as lc_tool  # type: ignore
    except Exception:
        lc_tool = None  # not installed

if lc_tool:
    @lc_tool("send_email")
    def send_email_tool(
        to: List[str],
        subject: str,
        body: str,
        is_html: bool = False,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
    ) -> str:
        """Send an email via SMTP. Args: to, subject, body, is_html, cc, bcc, attachments (file paths). Returns JSON string."""
        payload = EmailPayload(
            to=to,
            subject=subject,
            body=body,
            is_html=is_html,
            cc=cc or [],
            bcc=bcc or [],
            attachments=attachments or [],
        )
        try:
            result = send_email(payload)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

# --------- MCP tool wrapper (optional) ---------
try:
    # The MCP python SDK exposes a decorator `@mcp.tool()` in newer builds
    import mcp  # type: ignore
except Exception:
    mcp = None

if mcp and hasattr(mcp, "tool"):
    @mcp.tool(name="send_email", description="Send an email via SMTP. Supports attachments. Returns JSON string.")  # type: ignore
    def send_email_mcp(
        to: List[str],
        subject: str,
        body: str,
        is_html: bool = False,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
    ) -> str:
        payload = EmailPayload(
            to=to,
            subject=subject,
            body=body,
            is_html=is_html,
            cc=cc or [],
            bcc=bcc or [],
            attachments=attachments or [],
        )
        try:
            result = send_email(payload)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

# --------- Utility: base64 of last message for debugging (not exposed as tool) ---------

def _debug_preview_message(to: List[str], subject: str, body: str, is_html: bool = False) -> str:
    payload = EmailPayload(to=to, subject=subject, body=body, is_html=is_html)
    cfg = _load_smtp_config()
    msg = _build_message(cfg, payload)
    raw = msg.as_bytes()
    return base64.b64encode(raw).decode("ascii")


if __name__ == "__main__":
    # Minimal smoke test with env; set DRY_RUN=1 to skip send
    dry = _get_bool_env("DRY_RUN", False)
    test_to = os.getenv("TEST_TO")
    if not test_to:
        print("Set TEST_TO for a self-test, e.g., export TEST_TO=me@example.com")
    else:
        p = EmailPayload(to=[test_to], subject="email_tool smoke test", body="Hello from email_tool", is_html=False)
        if dry:
            print(_debug_preview_message(p.to, p.subject, p.body))
        else:
            print(json.dumps(send_email(p), indent=2))
