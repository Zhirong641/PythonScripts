# app_multi_ext.py
# -*- coding: utf-8 -*-
import os, json, torch, gradio as gr
from typing import Optional, Dict, Any
from dataclasses import dataclass
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

DTYPE = torch.float16
DEVICE = 'cuda'

# ========================
# 1) 模型注册表（按需增删）
# ========================
# 说明：新增模型只需复制一个条目，改 name、type、load 与 presets。
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # A) 你的 SD15（已注入自训 UNet 的 diffusers 目录）
    # "sd15_custom": {
    #     "name": "SD15 (sd15_unet_custom)",
    #     "type": "sd15",  # sd15 | sdxl
    #     "load": {
    #         "mode": "local",              # local | pretrained | singlefile
    #         "path": "./sd15_unet_custom"  # local 模式下的目录
    #     },
    #     "presets": {
    #         "widths":  [384, 448, 512, 576, 640, 704, 768],
    #         "heights": [384, 448, 512, 576, 640, 704, 768],
    #         "default_w": 512,
    #         "default_h": 512,
    #         "steps": 28,
    #         "guidance": 7.0,
    #     }
    # },
    # # B) 官方 SD15（对照）
    # "sd15_official": {
    #     "name": "SD15 Official (runwayml/stable-diffusion-v1-5)",
    #     "type": "sd15",
    #     "load": {
    #         "mode": "pretrained",
    #         "repo": "runwayml/stable-diffusion-v1-5"
    #     },
    #     "presets": {
    #         "widths":  [384, 448, 512, 576, 640, 704, 768],
    #         "heights": [384, 448, 512, 576, 640, 704, 768],
    #         "default_w": 512,
    #         "default_h": 512,
    #         "steps": 28,
    #         "guidance": 7.0,
    #     }
    # },
    # C) SDXL：Illustrious-XL v2.0
    "sdxl_illustrious_v2": {
        "name": "Illustrious-XL v2.0 (SDXL)",
        "type": "sdxl",
        "load": {
            "mode": "local",   # 推荐：单文件更通用；也可改为 pretrained
            "path": "/root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/"
            # "repo": "OnomaAIResearch/Illustrious-XL-v2.0",
            # "filename": "Illustrious-XL-v2.0.safetensors"  # 若离线则把同名文件放到当前目录
        },
        "presets": {
            "widths":  [512, 640, 768, 896, 1024, 1152, 1280],
            "heights": [512, 640, 768, 896, 1024, 1152, 1280],
            "default_w": 1024,
            "default_h": 1024,
            "steps": 28,
            "guidance": 5.5,
        }
    },
    # "illustrious_emberveil": {
    #     "name": "【illustrious】EmberVeilMix (SDXL)",
    #     "type": "sdxl",
    #     "load": {
    #         "mode": "singlefile",   # 推荐：单文件更通用；也可改为 pretrained
    #         "filename": "IllustriousEmberveilmix_v10.safetensors"  # 若离线则把同名文件放到当前目录
    #     },
    #     "presets": {
    #         "widths":  [512, 640, 768, 896, 1024, 1152, 1232, 1280],
    #         "heights": [512, 640, 768, 896, 1024, 1152, 1232, 1280],
    #         "default_w": 1024,
    #         "default_h": 1024,
    #         "steps": 28,
    #         "guidance": 5.5,
    #     }
    # },
    "my-sdxl": {
        "name": "my-sdxl (SDXL)",
        "type": "sdxl",
        "load": {
            "mode": "local",   # 推荐：单文件更通用；也可改为 pretrained
            "path": "./IllustriousEmberveilmix_v10_repo",  # 若离线则把同名文件放到当前目录
            "prediction_type": "epsilon"  # 可选：local 模式下可指定导出时的 prediction_type
        },
        "presets": {
            "widths":  [512, 640, 768, 896, 1024, 1152, 1232, 1280],
            "heights": [512, 640, 768, 896, 1024, 1152, 1232, 1280],
            "default_w": 1024,
            "default_h": 1024,
            "steps": 28,
            "guidance": 5.5,
        }
    },
    #  "my-sd15": {
    #     "name": "my-sd15 (SD15)",
    #     "type": "sd15",
    #     "load": {
    #         "mode": "local",   # 推荐：单文件更通用；也可改为 pretrained
    #         "path": "./my-sd15"  # 若离线则把同名文件放到当前目录
    #     },
    #     "presets": {
    #         "widths":  [512, 640, 768, 896, 1024, 1152, 1232, 1280],
    #         "heights": [512, 640, 768, 896, 1024, 1152, 1232, 1280],
    #         "default_w": 1024,
    #         "default_h": 1024,
    #         "steps": 28,
    #         "guidance": 5.5,
    #     }
    # },
}

# 可选：用环境变量覆写注册表中的字段，便于部署 CI/CD
# 约定：环境变量名以 REG__ 开头，后接 JSON 路径，用 __ 分段。例如：
#   REG__sdxl_illustrious_v2__load__mode=pretrained
#   REG__sdxl_illustrious_v2__load__repo=local/path/or_hf_repo
#   REG__sd15_custom__load__path=/abs/sd15_unet_custom

def apply_env_overrides(reg: Dict[str, Dict[str, Any]]):
    prefix = 'REG__'
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        parts = k[len(prefix):].split('__')
        cur = reg
        for p in parts[:-1]:
            if p not in cur:
                cur[p] = {}
            cur = cur[p]
        leaf = parts[-1]
        # 尝试把字符串解析为 JSON，否则就用原字符串
        try:
            cur[leaf] = json.loads(v)
        except Exception:
            cur[leaf] = v

apply_env_overrides(MODEL_REGISTRY)

# ===============
# 2) 加载器与缓存
# ===============
@dataclass
class PipeCache:
    pipe: Optional[object] = None
    model_key: Optional[str] = None

CACHE = PipeCache()

scheduler_map = {
    'euler': EulerDiscreteScheduler,
    'ddim': DDIMScheduler,
    'dpmpp2m': DPMSolverMultistepScheduler,
}

def _free_pipe():
    if CACHE.pipe is not None:
        try:
            CACHE.pipe.to('cpu')
        except Exception:
            pass
        del CACHE.pipe
        CACHE.pipe = None
    torch.cuda.empty_cache()


def _load_from_cfg(cfg: Dict[str, Any]):
    mtype = cfg['type']
    load = cfg['load']
    mode = load.get('mode', 'local')

    if mtype == 'sd15':
        if mode == 'local':
            path = load['path']
            p = StableDiffusionPipeline.from_pretrained(path, torch_dtype=DTYPE)
        elif mode == 'pretrained':
            repo = load['repo']
            p = StableDiffusionPipeline.from_pretrained(repo, torch_dtype=DTYPE)
        else:
            raise ValueError(f"SD15 unsupported mode: {mode}")
    elif mtype == 'sdxl':
        if mode == 'local':
            path = load['path']
            p = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=DTYPE, use_safetensors=True, low_cpu_mem_usage=False)
        elif mode == 'pretrained':
            repo = load['repo']
            p = StableDiffusionXLPipeline.from_pretrained(repo, torch_dtype=DTYPE, use_safetensors=True, low_cpu_mem_usage=False)
        elif mode == 'singlefile':
            if hf_hub_download is None or "repo" not in load:
                # 离线：要求 filename 在本地当前目录
                file_path = load['filename']
            else:
                file_path = hf_hub_download(repo_id=load['repo'], filename=load['filename'])
            p = StableDiffusionXLPipeline.from_single_file(file_path, torch_dtype=DTYPE)
        else:
            raise ValueError(f"SDXL unsupported mode: {mode}")
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    p = p.to(DEVICE)
    try:
        p.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    p.enable_vae_slicing(); p.enable_vae_tiling()
    return p


def ensure_pipe(model_key: str):
    if CACHE.pipe is not None and CACHE.model_key == model_key:
        return
    _free_pipe()
    cfg = MODEL_REGISTRY[model_key]
    CACHE.pipe = _load_from_cfg(cfg)
    CACHE.model_key = model_key

# =================
# 3) 推理与 UI 逻辑
# =================

def generate(model_key: str, prompt: str, neg: Optional[str], steps: int, guidance: float,
             width: int, height: int, scheduler: str, seed: Optional[str]):
    ensure_pipe(model_key)
    cfg = MODEL_REGISTRY[model_key]

    # 调度器
    Sched = scheduler_map.get(scheduler, EulerDiscreteScheduler)
    CACHE.pipe.scheduler = Sched.from_config(CACHE.pipe.scheduler.config)

    g = None
    if seed and str(seed).strip() != '':
        g = torch.Generator(device=DEVICE).manual_seed(int(seed))

    image = CACHE.pipe(
        prompt=prompt,
        negative_prompt=(neg or None),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        generator=g,
    ).images[0]
    return image


def on_model_change(model_key: str):
    cfg = MODEL_REGISTRY[model_key]
    p = cfg['presets']
    return (
        gr.update(choices=p['widths'], value=p['default_w']),
        gr.update(choices=p['heights'], value=p['default_h']),
        gr.update(value=p['guidance']),
        gr.update(value=p['steps']),
    )

# ============
# 4) 构建 UI
# ============
model_keys = list(MODEL_REGISTRY.keys())
model_names = [MODEL_REGISTRY[k]['name'] for k in model_keys]
key_by_name = {MODEL_REGISTRY[k]['name']: k for k in model_keys}

DEFAULT_KEY = model_keys[0]
DEFAULT_NAME = MODEL_REGISTRY[DEFAULT_KEY]['name']

with gr.Blocks(title='SD / SDXL Multi-Model (Extensible)') as demo:
    gr.Markdown("""
# SD15 / SDXL 多模型一页切换（可扩展）
- 上方下拉选择模型。要新增模型只需在 `MODEL_REGISTRY` 里添加一项即可。
- 切换时惰性加载并释放显存；分辨率/推荐参数会随模型自动联动。
""")

    model_sel_name = gr.Dropdown(model_names, value=DEFAULT_NAME, label='Model')

    with gr.Row():
        prompt = gr.Textbox(label='Prompt', value='masterpiece, best quality, 1girl, looking at viewer')
        neg = gr.Textbox(label='Negative', value='nsfw, lowres, blurry, watermark')
    with gr.Row():
        steps = gr.Slider(5, 100, value=MODEL_REGISTRY[DEFAULT_KEY]['presets']['steps'], step=1, label='Steps')
        guidance = gr.Slider(0.5, 20.0, value=MODEL_REGISTRY[DEFAULT_KEY]['presets']['guidance'], step=0.1, label='Guidance')
    with gr.Row():
        width = gr.Dropdown(choices=MODEL_REGISTRY[DEFAULT_KEY]['presets']['widths'], value=MODEL_REGISTRY[DEFAULT_KEY]['presets']['default_w'], label='Width')
        height = gr.Dropdown(choices=MODEL_REGISTRY[DEFAULT_KEY]['presets']['heights'], value=MODEL_REGISTRY[DEFAULT_KEY]['presets']['default_h'], label='Height')
    with gr.Row():
        scheduler = gr.Dropdown(choices=['euler','ddim','dpmpp2m'], value='euler', label='Scheduler')
        seed = gr.Textbox(label='Seed (empty=random)', value='')

    btn = gr.Button('Generate')
    out = gr.Image(label='Result', type='pil')

    # 把 name ↔ key 做一次映射
    def _name2key(name: str):
        return key_by_name.get(name, DEFAULT_KEY)

    model_sel_name.change(lambda n: on_model_change(_name2key(n)), inputs=[model_sel_name], outputs=[width, height, guidance, steps])
    btn.click(lambda n, *args: generate(_name2key(n), *args),
              inputs=[model_sel_name, prompt, neg, steps, guidance, width, height, scheduler, seed],
              outputs=[out])

if __name__ == '__main__':
    demo.queue(max_size=32).launch(server_name='0.0.0.0', server_port=7860, share=False)