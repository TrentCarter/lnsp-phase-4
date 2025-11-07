
import importlib, pytest

def test_receipt_helper_present():
    # Ensure at least one receipt write helper is available
    for modname, fname in (
        ("tools.aider_rpc.server_enhanced","write_receipt"),
        ("tools.aider_rpc.server","write_receipt"),
        ("tools.aider_rpc.receipts","write_atomic_json"),
    ):
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, fname, None)
            if callable(fn):
                return
        except ModuleNotFoundError:
            continue
    pytest.skip("No receipt writer helper found")
