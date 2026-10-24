import os
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig
import torch

pipe = AnimaImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="circlestone-labs/Anima",
            origin_file_pattern="split_files/diffusion_models/anima-preview3-base.safetensors",
        ),
        ModelConfig(
            model_id="circlestone-labs/Anima",
            origin_file_pattern="split_files/text_encoders/qwen_3_06b_base.safetensors",
        ),
        ModelConfig(
            model_id="circlestone-labs/Anima",
            origin_file_pattern="split_files/vae/qwen_image_vae.safetensors",
        ),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-0.6B", origin_file_pattern="./"),
    tokenizer_t5xxl_config=ModelConfig(
        model_id="stabilityai/stable-diffusion-3.5-large",
        origin_file_pattern="tokenizer_3/",
    ),
)

prompt = (
    "masterpiece, best quality, score_7, safe, "
    "solo, 1girl, long hair, silver hair, blue eyes, blue dress, "
    "underwater, air bubble, floating hair, refraction, portrait"
)
negative_prompt = "worst quality, low quality, score_1, score_2, score_3, artist name"

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    seed=0,
    num_inference_steps=30,
    height=1024,
    width=1024,
    cfg_scale=4.0,
)
image.save("test_preview3.jpg")
print("Done -> test_preview3.jpg")
