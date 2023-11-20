import torch
import random
import gc
import gradio as gr
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/japanese-stable-diffusion-xl",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
# if using torch < 2.0
# pipeline.enable_xformers_memory_efficient_attention()
pipe.to("cuda")


# @title Launch the demo
def infer_func(
    prompt,
    scale=7.5,
    steps=40,
    W=1024,
    H=1024,
    n_samples=1,
    seed="random",
    negative_prompt="",
):
    scale = float(scale)
    steps = int(steps)
    W = int(W)
    H = int(H)
    n_samples = int(n_samples)
    if seed == "random":
        seed = random.randint(0, 2**32)
    seed = int(seed)

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if len(negative_prompt) > 0 else None,
        guidance_scale=scale,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        num_images_per_prompt=n_samples,
        num_inference_steps=steps,
        height=H,
        width=W,
    ).images
    grid = make_image_grid(images, 1, len(images))
    gc.collect()
    torch.cuda.empty_cache()
    return grid, images, {"seed": seed}


with gr.Blocks() as demo:
    gr.Markdown("# Japanese Stable Diffusion XL Demo")
    gr.Markdown(
        """[Japanese Stable Diffusion XL](https://huggingface.co/stabilityai/japanese-stable-diffusion-xl) is a Japanese-specific SDXL by [Stability AI](https://ja.stability.ai/).
                - Blog: https://ja.stability.ai/blog/japanese-stable-diffusion-xl
                - Twitter: https://twitter.com/StabilityAI_JP
                - Discord: https://discord.com/invite/StableJP"""
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="prompt", max_lines=1, value="カラフルなペンギン、アート")
            scale = gr.Number(value=7.5, label="cfg_scale")
            steps = gr.Number(value=40, label="steps")
            width = gr.Number(value=1024, label="width")
            height = gr.Number(value=1024, label="height")
            n_samples = gr.Number(value=1, label="n_samples", precision=0, maximum=5)
            seed = gr.Text(value="42", label="seed (integer or 'random')")
            negative_prompt = gr.Textbox(label="negative prompt", value="")
            btn = gr.Button("Run")
        with gr.Column():
            out = gr.Image(label="grid")
            gallery = gr.Gallery(label="Generated images", show_label=False)
            info = gr.JSON(label="sampling_info")
    inputs = [
        prompt,
        scale,
        steps,
        width,
        height,
        n_samples,
        seed,
        negative_prompt,
    ]
    prompt.submit(infer_func, inputs=inputs, outputs=[out, gallery, info])
    btn.click(infer_func, inputs=inputs, outputs=[out, gallery, info])

demo.launch(debug=True, share=True, show_error=True)