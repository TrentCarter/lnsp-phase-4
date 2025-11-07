
import importlib, pytest

def test_heartbeat_module_present():
    try:
        importlib.import_module("tools.aider_rpc.heartbeat")
    except ModuleNotFoundError:
        pytest.skip("heartbeat module not present")
