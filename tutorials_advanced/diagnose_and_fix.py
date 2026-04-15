"""
diagnose_and_fix.py — Run this FIRST before anything else
==========================================================
Diagnoses exactly why your pipeline is stuck and fixes it.

USAGE:
    python diagnose_and_fix.py              # full diagnosis
    python diagnose_and_fix.py --fix        # diagnosis + patch OAI_CONFIG
    python diagnose_and_fix.py --test-api   # test API connection only
"""

import sys
import json
import socket
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ─── COLOURS ─────────────────────────────────────────────────────────────────

OK   = "✅"
FAIL = "❌"
WARN = "⚠️ "
INFO = "ℹ️ "

def ok(msg):   print(f"  {OK}  {msg}")
def fail(msg): print(f"  {FAIL} {msg}")
def warn(msg): print(f"  {WARN} {msg}")
def info(msg): print(f"  {INFO} {msg}")
def sep():     print("  " + "─" * 58)


# ─── 1. NETWORK CHECK ────────────────────────────────────────────────────────

def check_network():
    print("\n📡 NETWORK CHECK")
    sep()

    # DNS resolution
    try:
        ip = socket.gethostbyname("api.openai.com")
        ok(f"DNS resolved api.openai.com → {ip}")
    except socket.gaierror as e:
        fail(f"DNS failed for api.openai.com: {e}")
        fail("OpenAI is completely blocked on your network.")
        info("FIX: Switch to mobile hotspot, or use a VPN.")
        return False

    # TCP connect (port 443)
    try:
        s = socket.create_connection(("api.openai.com", 443), timeout=5)
        s.close()
        ok("TCP connect to api.openai.com:443 succeeded")
    except (OSError, TimeoutError) as e:
        fail(f"TCP connect to api.openai.com:443 failed: {e}")
        fail("Port 443 is blocked — likely university/office firewall.")
        info("FIX: Use mobile hotspot or VPN.")
        return False

    # HTTP GET with short timeout
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.openai.com",
            headers={"User-Agent": "python-urllib/3"},
        )
        resp = urllib.request.urlopen(req, timeout=8)
        ok(f"HTTPS reach api.openai.com → HTTP {resp.status}")
    except Exception as e:
        warn(f"HTTPS check returned: {e}")
        info("This may be a TLS/proxy issue — see below.")

    return True


# ─── 2. OAI CONFIG CHECK ─────────────────────────────────────────────────────

def check_oai_config():
    print("\n🔑 OAI_CONFIG_LIST.json CHECK")
    sep()

    candidates = [
        Path("../OAI_CONFIG_LIST.json"),
        Path("OAI_CONFIG_LIST.json"),
        Path("../../OAI_CONFIG_LIST.json"),
    ]

    cfg_path = None
    for p in candidates:
        if p.exists():
            cfg_path = p
            ok(f"Found: {p.resolve()}")
            break

    if cfg_path is None:
        fail("OAI_CONFIG_LIST.json NOT FOUND in any expected location!")
        info("Expected locations checked:")
        for p in candidates:
            print(f"       {p.resolve()}")
        info("FIX: Create the file (see template printed at end).")
        _print_config_template()
        return None, None

    try:
        raw = cfg_path.read_text(encoding="utf-8")
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        fail(f"JSON parse error: {e}")
        info("FIX: Fix the JSON syntax in OAI_CONFIG_LIST.json")
        return None, None

    if isinstance(cfg, dict):
        cfg = cfg.get("config_list", [cfg])

    if not cfg:
        fail("config_list is empty!")
        _print_config_template()
        return None, None

    ok(f"Found {len(cfg)} model config(s)")

    issues = []
    for i, c in enumerate(cfg):
        model   = c.get("model", "")
        api_key = c.get("api_key", "")
        base_url= c.get("base_url", "")

        print(f"\n  Config [{i}]:")
        print(f"    model    : {model or '(missing!)'}")
        print(f"    api_key  : {api_key[:8]}{'...' if len(api_key) > 8 else '(short!)'}")
        if base_url:
            print(f"    base_url : {base_url}")

        if not model:
            fail("  'model' field is missing or empty")
            issues.append("missing model")
        elif model not in KNOWN_GOOD_MODELS:
            warn(f"  Model '{model}' is not in known-good list: {KNOWN_GOOD_MODELS}")
            warn("  If this model doesn't exist on your account, the API will hang.")
            issues.append(f"unknown model: {model}")
        else:
            ok(f"  Model '{model}' looks valid")

        if not api_key:
            fail("  'api_key' field is missing")
            issues.append("missing api_key")
        elif not api_key.startswith("sk-"):
            warn(f"  api_key doesn't start with 'sk-' — may be Azure or proxy key")
        elif len(api_key) < 20:
            fail("  api_key looks too short — probably a placeholder")
            issues.append("bad api_key")
        else:
            ok("  api_key format looks OK")

    return cfg_path, cfg


KNOWN_GOOD_MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview",
    "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
    "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18",
]


def _print_config_template():
    print("""
  ── TEMPLATE for OAI_CONFIG_LIST.json ──────────────────────────
  Create this file at: Finrobot_old/OAI_CONFIG_LIST.json

  [
    {
      "model": "gpt-4o-mini",
      "api_key": "sk-YOUR-REAL-KEY-HERE"
    }
  ]

  Get your key from: https://platform.openai.com/api-keys
  ────────────────────────────────────────────────────────────────
""")


# ─── 3. LIVE API TEST ────────────────────────────────────────────────────────

def test_api(cfg):
    print("\n🔌 LIVE API TEST")
    sep()

    if not cfg:
        fail("No config to test — fix OAI_CONFIG_LIST.json first")
        return False

    try:
        import urllib.request
        import urllib.error
    except ImportError:
        fail("urllib not available")
        return False

    c       = cfg[0]
    api_key = c.get("api_key", "")
    model   = c.get("model", "gpt-4o-mini")
    base_url= c.get("base_url", "https://api.openai.com").rstrip("/")
    url     = f"{base_url}/v1/chat/completions"

    payload = json.dumps({
        "model":      model,
        "messages":   [{"role": "user", "content": "Say: OK"}],
        "max_tokens": 5,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data    = payload,
        method  = "POST",
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
    )

    print(f"  Testing: POST {url}")
    print(f"  Model  : {model}")
    print(f"  Timeout: 10 seconds")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body    = json.loads(resp.read().decode("utf-8"))
            reply   = body["choices"][0]["message"]["content"]
            tokens  = body.get("usage", {})
            ok(f"API responded! Reply: '{reply.strip()}'")
            ok(f"Tokens used: {tokens}")
            return True

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(body)
            msg = err.get("error", {}).get("message", body)
        except Exception:
            msg = body[:200]

        if e.code == 401:
            fail(f"HTTP 401 Unauthorized — API key is INVALID or EXPIRED")
            info("FIX: Get a new key from https://platform.openai.com/api-keys")
        elif e.code == 404:
            fail(f"HTTP 404 — Model '{model}' not found on your account")
            info(f"FIX: Change model in OAI_CONFIG_LIST.json to one of: {KNOWN_GOOD_MODELS[:3]}")
        elif e.code == 429:
            fail("HTTP 429 — Rate limit or quota exceeded")
            info("FIX: Check billing at https://platform.openai.com/usage")
        elif e.code == 400:
            fail(f"HTTP 400 Bad Request: {msg}")
        else:
            fail(f"HTTP {e.code}: {msg}")
        return False

    except TimeoutError:
        fail("Request TIMED OUT after 10 seconds")
        fail("This is your exact problem — the API call hangs and never returns.")
        print("""
  CAUSES (most likely first):
  ─────────────────────────────────────────────────────────────
  1. UNIVERSITY/OFFICE WIFI BLOCKING OPENAI  ← most common
     Fix: Switch to mobile hotspot right now and retry

  2. WRONG BASE URL in OAI_CONFIG_LIST.json
     If you have base_url set to a proxy that is down,
     all requests will time out silently.
     Fix: Remove base_url entirely, or fix it.

  3. VPN INTERFERENCE
     Some VPNs block OpenAI. Try without VPN.

  4. WINDOWS FIREWALL / ANTIVIRUS
     Rarely, Windows firewall blocks Python's outbound connections.
     Fix: Allow python.exe in Windows Firewall settings.
  ─────────────────────────────────────────────────────────────
""")
        return False

    except OSError as e:
        fail(f"Connection error: {e}")
        info("FIX: Check your network connection.")
        return False


# ─── 4. DEPENDENCY CHECK ─────────────────────────────────────────────────────

def check_deps():
    print("\n📦 DEPENDENCY CHECK")
    sep()

    deps = {
        "autogen":      "pyautogen",
        "finrobot":     "finrobot (install from Finrobot_old/)",
        "yfinance":     "yfinance",
        "pandas":       "pandas",
        "numpy":        "numpy",
        "xgboost":      "xgboost",
        "sklearn":      "scikit-learn",
        "matplotlib":   "matplotlib",
        "requests":     "requests",
        "transformers": "transformers (for FinBERT locally)",
    }

    missing = []
    for pkg, install_name in deps.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg:<16} v{ver}")
        except ImportError:
            if pkg in ("transformers", "sklearn", "xgboost"):
                warn(f"{pkg:<16} NOT installed — optional (rule-based fallback used)")
            else:
                fail(f"{pkg:<16} NOT installed → pip install {install_name}")
                missing.append(install_name)
        except OSError as e:
            # Broken PyTorch / CUDA DLL on Windows — does NOT affect FinBERT HF API usage
            if pkg == "transformers":
                warn(f"{pkg:<16} DLL broken (PyTorch/CUDA issue) — FinBERT uses HF API instead ✅")
                info(f"  Detail: {str(e)[:80]}")
            else:
                fail(f"{pkg:<16} DLL error: {str(e)[:80]}")
                missing.append(install_name)
        except Exception as e:
            warn(f"{pkg:<16} unexpected error: {str(e)[:80]}")

    if missing:
        print(f"\n  Run this to fix missing packages:")
        print(f"  pip install {' '.join(missing)}")

    # Check autogen version specifically
    try:
        import autogen
        ver = getattr(autogen, "__version__", "unknown")
        ok(f"autogen version: {ver}")
        # LLMConfig exists only in newer autogen
        try:
            from autogen import LLMConfig
            ok("LLMConfig available (autogen ≥ 0.4)")
        except ImportError:
            warn("LLMConfig not in this autogen version — forward_test_runner.py will fail")
            warn("Fix: pip install pyautogen --upgrade")
    except ImportError:
        pass


# ─── 5. AUTOGEN VERSION FIX ──────────────────────────────────────────────────

def check_autogen_api():
    print("\n🤖 AUTOGEN API COMPATIBILITY CHECK")
    sep()

    try:
        import autogen
        ver = getattr(autogen, "__version__", "0")
        ok(f"autogen {ver} installed")
    except ImportError:
        fail("autogen not installed")
        return

    # forward_test_runner uses: from autogen import LLMConfig
    try:
        from autogen import LLMConfig
        ok("from autogen import LLMConfig  → OK")
    except ImportError:
        fail("from autogen import LLMConfig  → FAILS")
        warn("forward_test_runner.py line 943 will crash before the agent even starts!")
        info("FIX: pip install pyautogen --upgrade")
        info("  OR edit forward_test_runner.py line 951:")
        print("""
  CHANGE THIS:
    from autogen import LLMConfig
    config_list = LLMConfig.from_json(path="../OAI_CONFIG_LIST.json").config_list

  TO THIS:
    import json
    config_list = json.load(open("../OAI_CONFIG_LIST.json"))
    if isinstance(config_list, dict):
        config_list = config_list.get("config_list", [config_list])
""")

    # Cache.disk usage
    try:
        from autogen import Cache
        ok("from autogen import Cache  → OK")
    except ImportError:
        try:
            from autogen.cache import Cache
            ok("from autogen.cache import Cache  → OK")
        except ImportError:
            warn("Cache not available — initiate_chat will run without cache (fine)")


# ─── 6. AUTO-PATCH forward_test_runner.py ────────────────────────────────────

def patch_forward_test_runner():
    print("\n🔧 PATCHING forward_test_runner.py")
    sep()

    # Find it
    candidates = [
        Path("forward_test_runner.py"),
        Path("tutorials_advanced/forward_test_runner.py"),
    ]
    fpath = None
    for p in candidates:
        if p.exists():
            fpath = p
            break

    if fpath is None:
        warn("forward_test_runner.py not found in current directory")
        info("Run this script from your tutorials_advanced/ folder")
        return

    src = fpath.read_text(encoding="utf-8")

    # Patch 1: LLMConfig → plain json.load
    old_llm = (
        "        config_list = LLMConfig.from_json(path=\"../OAI_CONFIG_LIST.json\").config_list"
    )
    new_llm = (
        "        with open(\"../OAI_CONFIG_LIST.json\") as _f:\n"
        "            config_list = json.load(_f)\n"
        "        if isinstance(config_list, dict):\n"
        "            config_list = config_list.get(\"config_list\", [config_list])"
    )

    old_import = "            from autogen import LLMConfig\n"

    patched = False
    if old_llm in src:
        src = src.replace(old_import, "")          # remove LLMConfig import line
        src = src.replace(old_llm, new_llm)
        ok("Patched LLMConfig.from_json → json.load")
        patched = True
    else:
        ok("LLMConfig patch not needed (already using json.load)")

    # Patch 2: raise timeout from 180 → 300
    if '"timeout": 180' in src:
        src = src.replace('"timeout": 180', '"timeout": 300')
        ok("Raised timeout: 180s → 300s")
        patched = True

    # Patch 3: Cache.disk with cache_seed → without (safer)
    if "Cache.disk(cache_seed=None)" in src:
        old_cache = (
            "        with Cache.disk(cache_seed=None) as cache:  # cache_seed=None = no caching (fresh run)\n"
            "            result = user_proxy.initiate_chat(\n"
            "                analyst,\n"
            "                message=prompt,\n"
            "                cache=cache,\n"
            "            )"
        )
        new_cache = (
            "        try:\n"
            "            from autogen import Cache as _Cache\n"
            "            with _Cache.disk(cache_seed=None) as cache:\n"
            "                result = user_proxy.initiate_chat(analyst, message=prompt, cache=cache)\n"
            "        except Exception:\n"
            "            result = user_proxy.initiate_chat(analyst, message=prompt)"
        )
        if old_cache in src:
            src = src.replace(old_cache, new_cache)
            ok("Patched Cache.disk with safe fallback")
            patched = True

    if patched:
        # Backup original
        backup = fpath.with_suffix(".py.bak")
        backup.write_text(fpath.read_text(encoding="utf-8"), encoding="utf-8")
        fpath.write_text(src, encoding="utf-8")
        ok(f"Saved patched file: {fpath}")
        ok(f"Original backed up: {backup}")
    else:
        ok("No patches needed — file looks compatible")


# ─── 7. SUMMARY + RECOMMENDATIONS ────────────────────────────────────────────

def print_summary(network_ok, api_ok):
    print("\n" + "="*62)
    print("  SUMMARY & NEXT STEPS")
    print("="*62)

    if not network_ok:
        print("""
  ❌ ROOT CAUSE: Network is blocking OpenAI API

  IMMEDIATE FIX (30 seconds):
  1. Turn on phone mobile hotspot
  2. Connect laptop to hotspot
  3. Run your script again
""")
    elif not api_ok:
        print("""
  ❌ ROOT CAUSE: API key is INVALID or EXPIRED  (HTTP 401)

  Your network is FINE. Your code is FINE.
  The only problem: the API key in OAI_CONFIG_LIST.json is dead.

  STEP-BY-STEP FIX:
  ──────────────────────────────────────────────────────────
  1. Open browser → https://platform.openai.com/api-keys

  2. Click "Create new secret key"
     Name it anything e.g. "finrobot"
     COPY the key (starts with sk-proj-...)
     ⚠  You can only see it ONCE — copy it now!

  3. Open this file in Notepad:
       C:\\Users\\KIIT0001\\Finrobot_old\\OAI_CONFIG_LIST.json

  4. Replace the api_key value with your new key:
     [
       {
         "model": "gpt-4o-mini",
         "api_key": "sk-proj-YOUR-NEW-KEY-HERE"
       }
     ]
     ⚠  Also change model to "gpt-4o-mini" — it's cheaper
        and faster than gpt-4-turbo. Your friends likely use this.

  5. Save the file, then test:
       python diagnose_and_fix.py --test-api

  6. If test passes, run your pipeline:
       python forward_test_runner.py --run --stocks TCS.NS

  ALSO NOTE — autogen v0.11.1 detected:
  ──────────────────────────────────────────────────────────
  Run this too to patch forward_test_runner.py:
       python diagnose_and_fix.py --fix

  ALSO NOTE — PyTorch DLL broken:
  ──────────────────────────────────────────────────────────
  This does NOT affect your pipeline — FinBERT uses the
  HuggingFace API (not local PyTorch). You can ignore it.
""")
    else:
        print("""
  ✅ Everything looks good!

  Run with one stock first to confirm:
    python forward_test_runner.py --run --stocks TCS.NS
""")

    print("="*62 + "\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSE Pipeline Diagnostics")
    parser.add_argument("--fix",       action="store_true", help="Auto-patch forward_test_runner.py")
    parser.add_argument("--test-api",  action="store_true", help="Test API connection only")
    args = parser.parse_args()

    print("="*62)
    print("  NSE PIPELINE — FULL DIAGNOSTICS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*62)

    # Always run these
    network_ok        = check_network()
    cfg_path, cfg     = check_oai_config()
    api_ok            = test_api(cfg) if network_ok else False
    check_deps()
    check_autogen_api()

    if args.fix:
        patch_forward_test_runner()

    print_summary(network_ok, api_ok)


if __name__ == "__main__":
    main()