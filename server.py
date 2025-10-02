# -*- coding: utf-8 -*-
import io, base64, torch, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler

MODEL_DIR = './sd15_unet_custom'
DEFAULT_SCHED = 'euler'   # euler|ddim|dpmpp2m
DEFAULT_PRED = 'eps'      # eps|vpred

app = FastAPI(title='SD1.5 UNet Inference API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class GenReq(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    steps: int = 28
    guidance: float = 7.0
    width: int = 512
    height: int = 512
    scheduler: str = DEFAULT_SCHED
    prediction_type: str = DEFAULT_PRED
    seed: int | None = None

class GenResp(BaseModel):
    image_base64: str

scheduler_map = {
    'euler': EulerDiscreteScheduler,
    'ddim': DDIMScheduler,
    'dpmpp2m': DPMSolverMultistepScheduler,
}

pipe: StableDiffusionPipeline | None = None

def load_pipeline():
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to('cuda')
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

@app.on_event('startup')
def _startup():
    load_pipeline()

@app.post('/txt2img', response_model=GenResp)
def txt2img(req: GenReq):
    assert pipe is not None
    # scheduler & pred
    Sched = scheduler_map.get(req.scheduler, EulerDiscreteScheduler)
    pipe.scheduler = Sched.from_config(pipe.scheduler.config)
    pipe.scheduler.config.prediction_type = 'v_prediction' if req.prediction_type=='vpred' else 'epsilon'

    g = None
    if req.seed is not None:
        g = torch.Generator(device='cuda').manual_seed(int(req.seed))

    image = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=int(req.steps),
        guidance_scale=float(req.guidance),
        height=int(req.height),
        width=int(req.width),
        generator=g,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return GenResp(image_base64=b64)

if __name__ == '__main__':
    uvicorn.run('server:app', host='0.0.0.0', port=8000, workers=1)