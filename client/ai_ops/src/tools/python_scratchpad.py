# from langchain_core.tools import tool

# @tool
# def run_python(code: str) -> dict:
#     """
#         Tool to run any python code by building a sandbox to execute code. This tool can also be used to plot graphs and maps.
#     """
#     import io, contextlib, traceback
#     ns, out, err = {}, io.StringIO(), io.StringIO()
#     try:
#         with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
#             exec(code, {}, ns)
#         return {"ok": True, "result": ns.get("result"), "stdout": out.getvalue(), "stderr": err.getvalue()}
#     except Exception:
#         return {"ok": False, "error": traceback.format_exc(), "stdout": out.getvalue(), "stderr": err.getvalue()}


from langchain_core.tools import tool

@tool
def run_python(code: str) -> dict:
    """
    Run untrusted Python in a sandbox-like context.
    All file writes are restricted to ALLOWED_DIR (and children).
    Stdout/stderr are captured. A variable named `result` is returned if present.
    """
    import io, os, builtins, contextlib, traceback
    from pathlib import Path
    from dotenv import load_dotenv

    THIS_DIR     = Path(__file__).resolve()
    PROJECT_ROOT = THIS_DIR.parent.parent.parent
    print(PROJECT_ROOT)
    load_dotenv(PROJECT_ROOT / "config/.env")


    # 1) Define the allowed base dir (env var override supported)
    
    ALLOWED_DIR = Path(os.getenv("FILE_SYSTEM_ACCESS_KEY")).resolve()
    ALLOWED_DIR.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(p: str | os.PathLike):
        p = Path(p)
        # allow simple names like "foo.html" by resolving relative to ALLOWED_DIR
        if not p.is_absolute():
            p = ALLOWED_DIR / p
        rp = p.resolve()
        if ALLOWED_DIR not in rp.parents and rp != ALLOWED_DIR:
            raise PermissionError(
                f"Access denied - path outside allowed directories: {rp} not in {ALLOWED_DIR}"
            )
        return rp

    # 2) Safe open wrapper
    _real_open = builtins.open
    def _safe_open(file, mode="r", *args, **kwargs):
        # block destructive reads/writes outside
        rp = _resolve_safe_path(file)
        # create parent dirs on write modes
        if any(ch in mode for ch in ("w", "a", "+", "x")):
            rp.parent.mkdir(parents=True, exist_ok=True)
        return _real_open(rp, mode, *args, **kwargs)

    # 3) Run the code with patched open and cwd set to ALLOWED_DIR
    ns, out, err = {}, io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cwd_before = os.getcwd()
            builtins.open = _safe_open
            try:
                os.chdir(ALLOWED_DIR)
                exec(code, {}, ns)
            finally:
                os.chdir(cwd_before)
                builtins.open = _real_open
        return {"ok": True, "result": ns.get("result"), "stdout": out.getvalue(), "stderr": err.getvalue()}
    except Exception:
        # ensure open is restored even on error
        builtins.open = _real_open
        return {"ok": False, "error": traceback.format_exc(), "stdout": out.getvalue(), "stderr": err.getvalue()}
