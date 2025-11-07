
import os
import pathlib
import importlib
import pytest

# Use project-relative artifacts directory (macOS compatible)
_repo_root = pathlib.Path(__file__).parent.parent.parent
ARTIFACTS_DIR = pathlib.Path(os.getenv("PAS_COST_DIR", str(_repo_root / "artifacts" / "costs"))).resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def _import_app():
    """
    Try to import FastAPI 'app' from tools.aider_rpc.server_enhanced,
    else fallback to tools.aider_rpc.server. Skip if neither exists.
    """
    for modname in ("tools.aider_rpc.server_enhanced", "tools.aider_rpc.server"):
        try:
            mod = importlib.import_module(modname)
            app = getattr(mod, "app", None)
            if app is not None:
                return mod, app
        except ModuleNotFoundError:
            continue
    pytest.skip("No Aider RPC server module with FastAPI 'app' found")

@pytest.fixture(scope="session")
def rpc_app():
    mod, app = _import_app()
    return app

@pytest.fixture(scope="session")
def rpc_module():
    mod, app = _import_app()
    return mod
