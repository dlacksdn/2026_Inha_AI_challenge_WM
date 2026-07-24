import torch, sys
print("torch", torch.__version__, "cuda build", torch.version.cuda, flush=True)
try:
    ok = torch.cuda.is_available()
    print("cuda.is_available:", ok, flush=True)
    if ok:
        x = torch.randn(8, device="cuda")
        y = (x*2).sum().item()
        print("CUDA op OK, y=", y, flush=True)
        print("device:", torch.cuda.get_device_name(0), flush=True)
except Exception as e:
    print("CUDA ERROR:", type(e).__name__, str(e)[:300], flush=True)
    sys.exit(2)
