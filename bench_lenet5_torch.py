# bench_lenet5_torch.py
import os, time, json, argparse, statistics
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)         # 或固定为物理核数
torch.set_num_interop_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda", choices=["cuda","cpu"])
parser.add_argument("--dtype", default="fp32", choices=["fp32","fp16","bf16"])
parser.add_argument("--batches", default="4")
parser.add_argument("--size", type=int, default=16, help="input side length S (HxW)")
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--runs", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
use_autocast = (device.type == "cuda" and dtype in [torch.float16, torch.bfloat16])

S  = args.size
s1 = S//2; s2 = s1//2; s3 = s2//2
assert s3 >= 1, f"size={S} 过小，三次 2x2 池化后空间维为 0。请至少使用 S>=8。"

class LeNet5Torch(nn.Module):
    def __init__(self, S, s1, s2, s3):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 3, padding=1, bias=True)
        self.ln1 = nn.LayerNorm([6,  S,  S])
        self.c2 = nn.Conv2d(6, 16, 3, padding=1, bias=True)
        self.ln2 = nn.LayerNorm([16, s1, s1])
        self.c3 = nn.Conv2d(16,120, 3, padding=1, bias=True)
        self.ln3 = nn.LayerNorm([120, s2, s2])
        self.pool = nn.MaxPool2d(2)
        self.fc   = nn.Linear(120*s3*s3, 10)

    def forward(self, x):
        x = self.pool(F.silu(self.ln1(self.c1(x))))   # -> [N,6, s1, s1]
        x = self.pool(F.silu(self.ln2(self.c2(x))))   # -> [N,16,s2, s2]
        x = self.pool(F.silu(self.ln3(self.c3(x))))   # -> [N,120,s3,s3]
        x = x.flatten(1)                               # -> [N,120*s3*s3]
        return self.fc(x)

model = LeNet5Torch(S,s1,s2,s3).to(device)
criterion = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=3e-3)

def tensor_of(shape):
    t = torch.rand(shape, device=device, dtype=(dtype if device.type=="cuda" else torch.float32))
    return t

def cuda_sync():
    if device.type == "cuda": torch.cuda.synchronize()

def bench_forward(batch, warmup=args.warmup, runs=args.runs):
    x = tensor_of((batch,1,S,S))
    times=[]
    for _ in range(warmup):
        with (torch.autocast("cuda", dtype=(torch.float16 if dtype==torch.float16 else torch.bfloat16))
              if use_autocast else torch.no_grad()):
            _ = model(x); cuda_sync()
    for _ in range(runs):
        t0 = time.perf_counter()
        with (torch.autocast("cuda", dtype=(torch.float16 if dtype==torch.float16 else torch.bfloat16))
              if use_autocast else torch.no_grad()):
            _ = model(x)
        cuda_sync()
        times.append((time.perf_counter()-t0)*1000.0)
    lat = sorted(times)
    p50, p95, p99 = lat[int(0.5*(runs-1))], lat[int(0.95*(runs-1))], lat[int(0.99*(runs-1))]
    thr = 1000.0 / (statistics.mean(lat)+1e-12) * batch
    mem = (torch.cuda.max_memory_allocated()/1024**2) if device.type=="cuda" else None
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    return {"latency_ms":{"p50":p50,"p95":p95,"p99":p99},"throughput_sps":thr,"max_mem_mb":mem}

def bench_train(batch, warmup=args.warmup, runs=args.runs):
    x = tensor_of((batch,1,S,S)); tgt = tensor_of((batch,10))
    times=[]
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        with (torch.autocast("cuda", dtype=(torch.float16 if dtype==torch.float16 else torch.bfloat16))
              if use_autocast else torch.enable_grad()):
            y = model(x); loss = criterion(y, tgt.to(y.dtype))
        (loss if not use_autocast else loss.float()).backward(); opt.step(); cuda_sync()
    for _ in range(runs):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with (torch.autocast("cuda", dtype=(torch.float16 if dtype==torch.float16 else torch.bfloat16))
              if use_autocast else torch.enable_grad()):
            y = model(x); loss = criterion(y, tgt.to(y.dtype))
        (loss if not use_autocast else loss.float()).backward(); opt.step(); cuda_sync()
        times.append((time.perf_counter()-t0)*1000.0)
    lat = sorted(times)
    p50, p95, p99 = lat[int(0.5*(runs-1))], lat[int(0.95*(runs-1))], lat[int(0.99*(runs-1))]
    thr = 1000.0 / (statistics.mean(lat)+1e-12) * batch
    mem = (torch.cuda.max_memory_allocated()/1024**2) if device.type=="cuda" else None
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    return {"latency_ms":{"p50":p50,"p95":p95,"p99":p99},"throughput_sps":thr,"max_mem_mb":mem}

results = {"meta":{
    "device": str(device), "dtype": args.dtype, "torch": torch.__version__,
    "warmup": args.warmup, "runs": args.runs, "size": S
}, "cases":[]}

for b in [int(x) for x in args.batches.split(",")]:
    model.eval(); f = bench_forward(b)
    model.train(); t = bench_train(b)
    results["cases"].append({"batch":b,"mode":"infer", **f})
    results["cases"].append({"batch":b,"mode":"train", **t})

os.makedirs("artifacts", exist_ok=True)
json.dump(results, open("artifacts/torch_lenet5_results.json","w"), indent=2)
print(json.dumps(results, indent=2))
