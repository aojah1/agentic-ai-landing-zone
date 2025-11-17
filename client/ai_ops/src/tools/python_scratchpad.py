from langchain_core.tools import tool
import re
from src.common.config import *

@tool
def run_python(code: str) -> dict:
    """
    Run untrusted Python in a sandbox-like context.
    All *writes* are restricted to output (and its children).
    Reads are allowed from anywhere.
    Stdout/stderr are captured. A variable named `result` is returned if present.

    Additionally:
    - Any files written via `open()` within OUTPUT_DIR are tracked.
    - For each such file, if DBOP_PUBLIC_BASE_URL is set, an HTTP URL is generated.
      These are returned in `files`, `file_urls`, and `file_links` fields.
    """
    import io, os, builtins, contextlib, traceback, html
    from pathlib import Path
    import tempfile

    # 1) Define base dir and OUTPUT_DIR
    base = FILE_SYSTEM_ACCESS_KEY
    if base:
        ALLOWED_DIR = Path(base).expanduser().resolve()
    else:
        ALLOWED_DIR = Path(tempfile.gettempdir(), "py_sandbox").resolve()
    ALLOWED_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR = (ALLOWED_DIR / "output").resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    PUBLIC_BASE_URL = DBOP_PUBLIC_BASE_URL

    created_files: set[Path] = set()

    # --- WRITE PATH: must stay under OUTPUT_DIR ---
    def _resolve_safe_write_path(p: str | os.PathLike) -> Path:
        p = Path(p)
        if not p.is_absolute():
            p = OUTPUT_DIR / p
        rp = p.resolve()
        if OUTPUT_DIR not in rp.parents and rp != OUTPUT_DIR:
            raise PermissionError(
                f"Access denied - path outside allowed directories: {rp} not in {OUTPUT_DIR}"
            )
        return rp

    # 2) Safe open wrapper
    _real_open = builtins.open

    def _safe_open(file, mode="r", *args, **kwargs):
        write_mode = any(ch in mode for ch in ("w", "a", "+", "x"))

        if write_mode:
            # Enforce sandbox for writes
            rp = _resolve_safe_write_path(file)
            rp.parent.mkdir(parents=True, exist_ok=True)
            created_files.add(rp)
            return _real_open(rp, mode, *args, **kwargs)

        # READ-ONLY: allow outside OUTPUT_DIR, just normalize path
        p = Path(file)
        if not p.is_absolute():
            p = OUTPUT_DIR / p  # relative reads still default into sandbox
        rp = p.resolve()
        return _real_open(rp, mode, *args, **kwargs)

    # Helper: map local Path under OUTPUT_DIR to public URL
    def _to_public_url(p: Path) -> str | None:
        if not PUBLIC_BASE_URL:
            return None
        try:
            rel = p.resolve().relative_to(OUTPUT_DIR)
        except ValueError:
            return None
        return f"{PUBLIC_BASE_URL.rstrip('/')}/{rel.as_posix()}"

    # 3) Run code with patched open and cwd set to OUTPUT_DIR
    ns, out, err = {}, io.StringIO(), io.StringIO()
    g = {
        "__builtins__": builtins.__dict__,
        "re": re,
    }

    def _build_file_metadata():
        """Builds files/file_urls/file_links from created_files."""
        # Deduplicate + stable order
        files_list = sorted({p.resolve() for p in created_files})
        file_paths = [str(p) for p in files_list]
        file_urls = []
        file_links = []
        for p in files_list:
            url = _to_public_url(p)
            file_urls.append(url)
            file_links.append({"path": str(p), "url": url})
        return file_paths, file_urls, file_links

    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cwd_before = os.getcwd()
            builtins.open = _safe_open
            try:
                os.chdir(OUTPUT_DIR)
                exec(code, g, ns)
            finally:
                os.chdir(cwd_before)
                builtins.open = _real_open

        file_paths, file_urls, file_links = _build_file_metadata()
        return {
            "ok": True,
            "result": ns.get("result"),
            "stdout": html.escape(out.getvalue()),
            "stderr": html.escape(err.getvalue()),
            # New fields (non-breaking)
            "files": file_paths,
            "file_urls": file_urls,
            "file_links": file_links,
        }
    except Exception:
        builtins.open = _real_open
        file_paths, file_urls, file_links = _build_file_metadata()
        return {
            "ok": False,
            "error": html.escape(traceback.format_exc()),
            "stdout": html.escape(out.getvalue()),
            "stderr": html.escape(err.getvalue()),
            # Still include file metadata in error case if anything was written
            "files": file_paths,
            "file_urls": file_urls,
            "file_links": file_links,
        }
