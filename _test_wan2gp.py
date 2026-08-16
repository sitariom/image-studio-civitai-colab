import gradio as gr
import http.server
gr.Blocks.launch = lambda self, *a, **k: None
class _D(http.server.ThreadingHTTPServer):
    def __init__(self, *a, **k): pass
    def serve_forever(self, *a, **k): pass
http.server.ThreadingHTTPServer = _D
import importlib.util, os, traceback
spec = importlib.util.spec_from_file_location("ua", "/content/universal_app.py")
ua = importlib.util.module_from_spec(spec); spec.loader.exec_module(ua)
ckpt = "/content/studio/models/civitai/arthemyComicsKrea2_v10.safetensors"
print("ckpt existe:", os.path.exists(ckpt), round(os.path.getsize(ckpt)/1e9,2),"GB", flush=True)
try:
    ua._load_wan2gp_custom(ckpt, "Krea 2", "arthemyComicsKrea2_v10", progress_cb=None)
    print("WAN2GP_OK backend=", ua.STATE.get("backend"), flush=True)
except Exception as e:
    print("WAN2GP_ERRO:", repr(e), flush=True)
    traceback.print_exc()
