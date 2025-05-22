# early_blocker.py
import os
import builtins

# 1. Tell any code checking env vars to back off
os.environ["TINYTROUPE_DISABLE"] = "1"

# 2. Hijack import early to block TinyTroupe from loading
_real_import = builtins.__import__

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if "tinytroupe" in name.lower():
        raise ImportError(f"TinyTroupe forcibly disabled (blocked {name})")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = safe_import

print("[early_blocker] TinyTroupe disabled.")