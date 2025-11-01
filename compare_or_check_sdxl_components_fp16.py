#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_or_check_sdxl_components_fp16.py

功能：
1) 两仓库比较：--components 任选 vae/unet/te1/te2，FP16友好相似度（cos/relL2/allclose/分位数）+ 分模块汇总 + 非有限值告警。
2) 单仓库体检（只传一个 repo）：逐组件扫描 NaN/Inf，统计比例、Top 异常键、分模块均值/方差/极值，判定“权重是否有效”。

组件与子目录：
- vae  -> repo/vae/
- unet -> repo/unet/
- te1  -> repo/text_encoder/
- te2  -> repo/text_encoder_2/

用法：
# 单仓库体检（默认自动检测所有存在的组件）
python compare_or_check_sdxl_components_fp16.py <repoA> --scan-nonfinite

# 比较两个仓库（自动检测双方都存在的组件）
python compare_or_check_sdxl_components_fp16.py <repoA> <repoB>

# 指定组件
python compare_or_check_sdxl_components_fp16.py <repoA> <repoB> --components vae unet
python compare_or_check_sdxl_components_fp16.py <repoA> --components te1 te2 --scan-nonfinite
"""

import os, sys, math, json, glob, hashlib, argparse
from typing import Dict, Tuple, Callable, List

import torch

# --------------------- 公共工具 ---------------------

IGNORE_CFG_KEYS = {"_class_name", "_diffusers_version", "torch_dtype"}

def read_config(dirpath: str):
    p = os.path.join(dirpath, "config.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k in IGNORE_CFG_KEYS:
        cfg.pop(k, None)
    return cfg

def file_hash(path: str, chunk: int = 1<<20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def find_weight_file(subfolder: str):
    """按优先级查找权重文件"""
    if not os.path.isdir(subfolder):
        return None
    cand = []
    cand += glob.glob(os.path.join(subfolder, "diffusion_pytorch_model.safetensors"))
    cand += glob.glob(os.path.join(subfolder, "model.safetensors"))
    cand += glob.glob(os.path.join(subfolder, "*.safetensors"))
    cand += glob.glob(os.path.join(subfolder, "pytorch_model.bin"))
    cand += glob.glob(os.path.join(subfolder, "model.bin"))
    return cand[0] if cand else None

def load_state(path: str) -> Dict[str, torch.Tensor]:
    """加载 safetensors 或 torch bin 为 state_dict（CPU）"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file as safe_load
        return safe_load(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unrecognized bin format: {path}")

def scan_nonfinite(sd: Dict[str, torch.Tensor]) -> List[Tuple[str,int,int]]:
    """
    返回列表：(key, 非有限个数, 总元素数)
    """
    bad = []
    for k, t in sd.items():
        if torch.is_floating_point(t):
            total = t.numel()
            nf = (~torch.isfinite(t)).sum().item()
            if nf > 0:
                bad.append((k, nf, total))
    return bad

# --------------------- 分组规则（便于定位差异/异常） ---------------------

def group_name_textenc(k: str) -> str:
    ks = k.lower()
    if ("token_emb" in ks) or ("position_emb" in ks) or ("positional" in ks) or ("embed" in ks and "encoder" not in ks):
        return "embeddings"
    if ("attention" in ks) or (".attn" in ks) or ("self_attn" in ks):
        return "attn"
    if ("mlp" in ks) or ("fc1" in ks) or ("fc2" in ks) or ("proj" in ks and "text_projection" not in ks):
        return "mlp"
    if ("layer_norm" in ks) or ("layernorm" in ks) or ("ln_" in ks) or (".ln" in ks) or ("final_layer_norm" in ks):
        return "ln"
    if "encoder" in ks:
        return "encoder"
    if "text_projection" in ks:
        return "text_projection"
    return "other"

def group_name_vae(k: str) -> str:
    if "encoder." in k: return "encoder"
    if "decoder." in k: return "decoder"
    if "quant_conv" in k: return "quant_conv"
    if "post_quant_conv" in k: return "post_quant_conv"
    return "other"

def group_name_unet(k: str) -> str:
    ks = k.lower()
    if "time_embedding" in ks or "time_embed" in ks or "time_mlp" in ks:
        return "time_embed"
    if "conv_in" in ks:
        return "conv_in"
    if "conv_out" in ks:
        return "conv_out"
    if "mid_block" in ks:
        return "mid_block"
    if "down_blocks" in ks:
        return "down_blocks"
    if "up_blocks" in ks:
        return "up_blocks"
    if "proj" in ks and "to_out" in ks:
        return "attn_proj"
    if "norm" in ks and "group" in ks:
        return "group_norm"
    return "other"

# --------------------- 相似度比较（双仓库时使用） ---------------------

def compare_state_dicts_fp16(sdA: Dict[str, torch.Tensor],
                             sdB: Dict[str, torch.Tensor],
                             group_fn,
                             *,
                             atol=2e-3, rtol=1e-3,
                             quantiles=(50,90,99,99.9),
                             per_layer_sample_max=20000,
                             global_sample_cap=2_000_000,
                             warn_nonfinite=True):

    keysA, keysB = set(sdA.keys()), set(sdB.keys())
    onlyA = sorted(keysA - keysB)
    onlyB = sorted(keysB - keysA)
    common = sorted(keysA & keysB)

    per_layer_mismatch = []
    allclose_hits = 0
    total_float_layers = 0

    agg = dict(dot=0.0, na2=0.0, nb2=0.0, rl2_num=0.0, rl2_den=0.0, elems=0)
    groups = {}
    agg_abs_sum = 0.0
    abs_max = 0.0
    sample_buf = []

    for k in common:
        a = sdA[k]; b = sdB[k]
        if a.shape != b.shape:
            per_layer_mismatch.append((k, tuple(a.shape), tuple(b.shape)))
            continue
        if not torch.is_floating_point(a):
            continue

        a32 = a.to(torch.float32, copy=False).reshape(-1)
        b32 = b.to(torch.float32, copy=False).reshape(-1)

        # 屏蔽非有限
        mask = torch.isfinite(a32) & torch.isfinite(b32)
        if not bool(mask.all()):
            if warn_nonfinite:
                bad_a = (~torch.isfinite(a32)).sum().item()
                bad_b = (~torch.isfinite(b32)).sum().item()
                print(f"[WARN nonfinite] {k}: A_bad={bad_a} B_bad={bad_b}")
            a_eff = a32[mask]; b_eff = b32[mask]
            if a_eff.numel() < 10:
                continue
        else:
            a_eff, b_eff = a32, b32

        n = a_eff.numel()
        diff = a_eff - b_eff
        adiff = diff.abs()

        # 全局
        agg["elems"] += n
        agg["dot"]  += float(torch.dot(a_eff, b_eff))
        agg["na2"]  += float(torch.dot(a_eff, a_eff))
        agg["nb2"]  += float(torch.dot(b_eff, b_eff))
        num = float(torch.linalg.vector_norm(diff))
        den = float(torch.linalg.vector_norm(a_eff))
        agg["rl2_num"] += num
        agg["rl2_den"] += (den if den > 0 else 0.0)

        is_close = torch.allclose(a_eff, b_eff, atol=atol, rtol=rtol)
        allclose_hits += int(is_close)
        total_float_layers += 1

        # |A-B| 统计
        agg_abs_sum += float(adiff.sum())
        cur_max = float(adiff.max())
        if cur_max > abs_max: abs_max = cur_max

        # 分组
        g = group_fn(k)
        s = groups.setdefault(g, dict(dot=0.0, na2=0.0, nb2=0.0, rl2_num=0.0, rl2_den=0.0, elems=0))
        s["elems"]   += n
        s["dot"]     += float(torch.dot(a_eff, b_eff))
        s["na2"]     += float(torch.dot(a_eff, a_eff))
        s["nb2"]     += float(torch.dot(b_eff, b_eff))
        s["rl2_num"] += num
        s["rl2_den"] += (den if den > 0 else 0.0)

        # 分层抽样
        m = min(per_layer_sample_max, n)
        if m > 0:
            idx = torch.randint(0, n, (m,), dtype=torch.int64)
            sample_buf.append(adiff[idx].cpu())

    # 整体
    if agg["na2"] == 0 or agg["nb2"] == 0:
        cosine = float("nan") if agg["na2"] != agg["nb2"] else 1.0
    else:
        cosine = agg["dot"] / (math.sqrt(agg["na2"]) * math.sqrt(agg["nb2"]))
    rel_l2 = (agg["rl2_num"] / agg["rl2_den"]) if agg["rl2_den"] > 0 else float("inf")
    allclose_ratio = (allclose_hits / total_float_layers) if total_float_layers > 0 else float("nan")

    # 分位数（抽样近似）
    if len(sample_buf) > 0:
        samples = torch.cat(sample_buf, dim=0)
        if samples.numel() > global_sample_cap:
            idx = torch.randperm(samples.numel())[:global_sample_cap]
            samples = samples[idx]
    else:
        samples = torch.empty(0, dtype=torch.float32)
    q_stats = {}
    for q in (50,90,99,99.9):
        q_stats[q] = float(torch.quantile(samples, q/100)) if samples.numel() > 0 else float("nan")
    abs_mean = (agg_abs_sum / agg["elems"]) if agg["elems"] > 0 else float("nan")

    # 分组
    group_stats = {}
    for g, s in groups.items():
        if s["na2"] == 0 or s["nb2"] == 0:
            cos_g = float("nan") if s["na2"] != s["nb2"] else 1.0
        else:
            cos_g = s["dot"] / (math.sqrt(s["na2"]) * math.sqrt(s["nb2"]))
        rl2_g = (s["rl2_num"] / s["rl2_den"]) if s["rl2_den"] > 0 else float("inf")
        group_stats[g] = dict(cosine=cos_g, rel_l2=rl2_g, elems=s["elems"])

    return {
        "onlyA": sorted(onlyA), "onlyB": sorted(onlyB),
        "shape_mismatch": per_layer_mismatch,
        "total_elems": agg["elems"],
        "cosine": cosine, "rel_l2": rel_l2, "allclose_ratio": allclose_ratio,
        "q_stats": q_stats, "abs_mean": abs_mean, "abs_max": abs_max,
        "group_stats": group_stats, "float_layers": total_float_layers,
        "sampled": int(samples.numel()),
    }

# --------------------- 单仓库体检（扫描非有限 + 统计分布） ---------------------

def stats_one_repo_component(repo: str, subdir: str, group_fn):
    folder = os.path.join(repo, subdir)
    w = find_weight_file(folder)
    print(f"\n=== 体检 {subdir}/ ===")
    if not w:
        print("⚠️ 未找到权重文件，跳过")
        return

    print(f"权重：{w}  |  sha256: {file_hash(w)[:16]}...")
    cfg = read_config(folder)
    if cfg is not None:
        print("发现 config.json：关键键数量 =", len(cfg))

    sd = load_state(w)

    # 汇总
    total_params = 0
    float_params = 0
    groups = {}
    nonfinite_total = 0
    top_bad = []  # (k, nf, total)

    # 遍历参数
    for k, t in sd.items():
        n = t.numel()
        total_params += n
        if not torch.is_floating_point(t):
            continue
        float_params += n

        # 统计基础分布（有限值）
        t32 = t.to(torch.float32, copy=False).reshape(-1)
        mask = torch.isfinite(t32)
        nf = int((~mask).sum().item())
        if nf > 0:
            nonfinite_total += nf
            top_bad.append((k, nf, n))

        eff = t32[mask]
        if eff.numel() == 0:
            continue
        g = group_fn(k)
        s = groups.setdefault(g, dict(elems=0, sum=0.0, sumsq=0.0, absmax=0.0))
        s["elems"] += eff.numel()
        s["sum"]   += float(eff.sum())
        s["sumsq"] += float((eff*eff).sum())
        cur_absmax = float(eff.abs().max())
        if cur_absmax > s["absmax"]:
            s["absmax"] = cur_absmax

    # 输出
    print(f"- 参数总量：{total_params:,}  （浮点参数：{float_params:,}）")
    print(f"- 非有限值总数（NaN/Inf）：{nonfinite_total:,}  占浮点参数比例："
          f"{(nonfinite_total/float_params*100 if float_params else 0):.6f}%")

    if top_bad:
        top_bad.sort(key=lambda x: x[1], reverse=True)
        print(f"- 含非有限值的键：{len(top_bad)}  Top-10：")
        for k, nf, n in top_bad[:10]:
            print(f"  * {k}  nf={nf:,} ({nf/n*100:.6f}%)  size={n:,}")
    else:
        print("- 未发现非有限值。")

    print("\n— 分模块统计（均值/标准差/absmax/elem）—")
    for g, s in sorted(groups.items(), key=lambda kv: kv[0]):
        elems = s["elems"]
        mean = s["sum"]/elems if elems else float("nan")
        var = s["sumsq"]/elems - mean*mean if elems else float("nan")
        std = math.sqrt(max(var, 0.0)) if math.isfinite(var) else float("nan")
        print(f"{g:16s}  mean={mean:.3e}  std={std:.3e}  absmax={s['absmax']:.3e}  elems={elems:,}")

    # 结论（体检）
    if nonfinite_total > 0:
        print("\n判定：❌ 检测到 NaN/Inf，请修复或重新导出（至少将非有限值替换掉）。")
    else:
        print("\n判定：✅ 未发现 NaN/Inf，就数值健康性而言“有效”。（功能是否匹配仍需按需做对照测试）")

# --------------------- 组件映射 & CLI ---------------------

COMP_MAP = {
    "vae": ("vae", group_name_vae),
    "unet": ("unet", group_name_unet),
    "te1": ("text_encoder", group_name_textenc),
    "te2": ("text_encoder_2", group_name_textenc),
    "text_encoder": ("text_encoder", group_name_textenc),
    "text_encoder_2": ("text_encoder_2", group_name_textenc),
}

def auto_components_in(repo: str) -> List[str]:
    res = []
    for key, (subdir, _) in COMP_MAP.items():
        if key in ("text_encoder", "text_encoder_2"):  # 别名不列
            continue
        if os.path.isdir(os.path.join(repo, subdir)) and find_weight_file(os.path.join(repo, subdir)):
            res.append(key)
    # 固定顺序
    order = ["vae", "unet", "te1", "te2"]
    return [k for k in order if k in res]

def auto_components_both(repoA: str, repoB: str) -> List[str]:
    inA = set(auto_components_in(repoA))
    inB = set(auto_components_in(repoB))
    order = ["vae", "unet", "te1", "te2"]
    return [k for k in order if k in inA & inB]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repoA")
    ap.add_argument("repoB", nargs="?", default=None)
    ap.add_argument("--components", nargs="+", default=["auto"],
                    help="选择组件：auto / vae / unet / te1 / te2（可多选）")
    ap.add_argument("--atol", type=float, default=2e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--quantiles", type=float, nargs="+", default=[50,90,99,99.9])
    ap.add_argument("--per-layer-sample", type=int, default=20000)
    ap.add_argument("--global-sample", type=int, default=2_000_000)
    ap.add_argument("--scan-nonfinite", action="store_true",
                    help="比较前打印非有限键（双仓库）；单仓库体检默认就会统计非有限")
    args = ap.parse_args()

    # 规范组件列表
    comps = args.components
    if len(comps) == 1 and comps[0].lower() == "auto":
        comps = auto_components_in(args.repoA) if args.repoB is None else auto_components_both(args.repoA, args.repoB)
        if not comps:
            mode = "单仓库" if args.repoB is None else "双仓库"
            print(f"{mode} auto 未找到可用组件；请用 --components 指定，如：--components vae unet te1")
            sys.exit(1)
    norm = []
    for c in comps:
        c = c.lower()
        if c in ("text_encoder", "te1"): norm.append("te1")
        elif c in ("text_encoder_2", "te2"): norm.append("te2")
        elif c in ("vae", "unet"): norm.append(c)
        else:
            print(f"忽略未知组件：{c}")
    order = ["vae", "unet", "te1", "te2"]
    norm = [x for x in order if x in set(norm)]

    if args.repoB is None:
        # 单仓库体检模式
        print(f"== 单仓库体检：{args.repoA} ==")
        for c in norm:
            subdir, group_fn = COMP_MAP[c]
            stats_one_repo_component(args.repoA, subdir, group_fn)
    else:
        # 双仓库比较模式
        print(f"== 双仓库比较：A={args.repoA}  B={args.repoB} ==")
        for c in norm:
            subdir, group_fn = COMP_MAP[c]
            a_dir = os.path.join(args.repoA, subdir)
            b_dir = os.path.join(args.repoB, subdir)
            wa = find_weight_file(a_dir)
            wb = find_weight_file(b_dir)

            print(f"\n=== 比较 {subdir}/ ===")
            if not wa or not wb:
                print(f"⚠️ 找不到权重（A:{wa} B:{wb}）——跳过")
                continue

            print(f"A: {wa}")
            print(f"B: {wb}")
            print("哈希：", "相同" if file_hash(wa)==file_hash(wb) else "不同")

            cfgA, cfgB = read_config(a_dir), read_config(b_dir)
            if cfgA is not None and cfgB is not None:
                same_cfg = (cfgA == cfgB)
                print("配置一致：", same_cfg)
                if not same_cfg:
                    keys = sorted(set(cfgA.keys()) | set(cfgB.keys()))
                    diffs = [k for k in keys if cfgA.get(k) != cfgB.get(k)]
                    if diffs: print("配置差异键：", diffs)
            else:
                print("配置：有缺失（跳过对比）")

            sdA = load_state(wa)
            sdB = load_state(wb)

            if args.scan_nonfinite:
                badA = scan_nonfinite(sdA); badB = scan_nonfinite(sdB)
                print(f"A 非有限键数：{len(badA)} 示例：{badA[:5]}")
                print(f"B 非有限键数：{len(badB)} 示例：{badB[:5]}")

            res = compare_state_dicts_fp16(
                sdA, sdB, group_fn,
                atol=args.atol, rtol=args.rtol,
                quantiles=tuple(args.quantiles),
                per_layer_sample_max=args.per_layer_sample,
                global_sample_cap=args.global_sample,
                warn_nonfinite=True,
            )

            if res["onlyA"]:
                print(f"仅 A 有的键：{len(res['onlyA'])} 示例：{res['onlyA'][:5]}")
            if res["onlyB"]:
                print(f"仅 B 有的键：{len(res['onlyB'])} 示例：{res['onlyB'][:5]}")

            print("\n— 形状一致性 —")
            if res["shape_mismatch"]:
                print(f"❌ 有 {len(res['shape_mismatch'])} 个键形状不同，示例：")
                for k, sa, sb in res["shape_mismatch"][:5]:
                    print(f"  - {k}: {sa} vs {sb}")
            else:
                print("✅ 共同层形状全部一致")

            print("\n— 整体相似度（FP16 友好）—")
            print(f"参数总元素：{res['total_elems']:,}")
            print(f"cosine(整体)：{res['cosine']:.8f}")
            print(f"relL2(整体)：{res['rel_l2']:.3e}")
            print(f"allclose 比例（atol={args.atol:g}, rtol={args.rtol:g}）：{res['allclose_ratio']:.3%}")
            print("- |A-B| 分位数：", ", ".join([f"p{q:g}={res['q_stats'][q]:.3e}" for q in args.quantiles]))
            print(f"- |A-B| mean：{res['abs_mean']:.3e}, max：{res['abs_max']:.3e}")
            print(f"- 分位数抽样量：{res['sampled']:,}")

            print("\n— 分模块（cos / relL2 / elems）—")
            for g in sorted(res["group_stats"].keys()):
                s = res["group_stats"][g]
                print(f"{g:16s}  cos={s['cosine']:.8f}  relL2={s['rel_l2']:.3e}  elems={s['elems']:,}")

            cos, rl2 = res["cosine"], res["rel_l2"]
            print("\n结论：", end="")
            if math.isfinite(cos) and math.isfinite(rl2):
                if cos > 0.99999 and rl2 < 5e-3:
                    print("几乎相同 / 功能等价。")
                elif cos > 0.9999 and rl2 < 2e-2:
                    print("非常接近。")
                else:
                    print("差异较大。")
            else:
                print("无法判断（存在非有限或有效样本不足）。")

if __name__ == "__main__":
    main()
