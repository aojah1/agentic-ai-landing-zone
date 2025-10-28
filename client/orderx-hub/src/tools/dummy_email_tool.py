# dummy_email_tool.py
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from oci.addons.adk import Toolkit, tool
# try:
#     from langchain_core.tools import tool  # LangChain >= 0.2
# except Exception:
#     from langchain.tools import tool  # fallback for older LangChain


OUTBOX_DIR = Path(os.getenv("DUMMY_EMAIL_OUTBOX", "./outbox"))
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DummyEmail:
    to: List[str]
    subject: str
    body: str
    is_html: bool = False
    cc: List[str] = None
    bcc: List[str] = None
    attachments: List[str] = None


def _save(email: DummyEmail, message_id: str) -> str:
    record = {
        "message_id": message_id,
        "timestamp": int(time.time()),
        "email": asdict(email),
    }
    path = OUTBOX_DIR / f"{message_id}.json"
    path.write_text(json.dumps(record, indent=2))
    return str(path)


@tool
def send_email_dummy(
    to: List[str],
    subject: str,
    body: str,
    is_html: bool = False,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    attachments: Optional[List[str]] = None,
) -> str:
    """
    Simulates sending an email (no SMTP). Writes a JSON record to DUMMY_EMAIL_OUTBOX (default: ./outbox).
    Returns a JSON string: {status, message_id, saved_path, to, subject, size_bytes}.
    """
    if not to:
        raise ValueError("`to` must include at least one recipient.")
    msg_id = f"dummy-{uuid.uuid4().hex}"
    email = DummyEmail(
        to=to,
        subject=subject,
        body=body,
        is_html=is_html,
        cc=cc or [],
        bcc=bcc or [],
        attachments=attachments or [],
    )
    saved_path = _save(email, msg_id)
    return json.dumps(
        {
            "status": "simulated",
            "message_id": msg_id,
            "saved_path": saved_path,
            "to": to,
            "subject": subject,
            "size_bytes": len(body.encode("utf-8")),
        }
    )


# if __name__ == "__main__":
#     # quick local test (no LangChain needed)
#     print(
#         send_email_dummy.invoke(
#             {
#                 "to": ["ops@example.com"],
#                 "subject": "Run complete",
#                 "body": "Done.",
#             }
#         )
#     )
