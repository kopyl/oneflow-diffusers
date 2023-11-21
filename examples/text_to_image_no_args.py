from onediff.infer_compiler import oneflow_compile, oneflow_load_compiled
from onediff.schedulers import EulerDiscreteScheduler
from onediff.optimization import rewrite_self_attention
from diffusers import StableDiffusionPipeline
import torch
import os


compiled_graph_path = "/workspace/test"
prompt = "an icon of a cat"
model_id = "/workspace/sd-1_5-icons-172800_steps-4e_7"
height = 512
width = 512
steps = 30
warmup = 1
seed = 5

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    use_auth_token=True,
    revision="fp16",
    variant="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe = pipe.to("cuda")

compiled_graph_exists = os.path.exists(compiled_graph_path)

if not compiled_graph_exists:
    rewrite_self_attention(pipe.unet)
    pipe.unet = oneflow_compile(pipe.unet)
else:
    pipe.unet = oneflow_load_compiled(pipe.unet, compiled_graph_path, device="cuda")


torch.manual_seed(seed)

prompt = "an icon of a star"
images = pipe(
    prompt, height=height, width=width, num_inference_steps=steps
).images

if not compiled_graph_exists:
    print("Saving compiled graph")
    pipe.unet.save_graph(compiled_graph_path)

for i, image in enumerate(images):
    image.save(f"{prompt}-of-{i}.png")
