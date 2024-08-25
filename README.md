# Flux-Dev-24GB-Quantize-8bit

```python
from flux_inference.py import *


base_model = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

print('Loading and fusing lora, please wait...')

# pipe.load_lora_weights("Load Lora Model here")
# pipe.fuse_lora(lora_scale=0.125)
# pipe.unload_lora_weights()

print('Quantizing, please wait...')
quantize(pipe.transformer, qfloat8)
freeze(pipe.transformer)
print('Model quantized!')
pipe.enable_model_cpu_offload()

ts_cutoff = 2
lora_prompts = [
    "a photo man 'ameer' written on tshirt "]

print("Model Loading Time is ",time.time()-st )

st =  time.time()
generator = torch.Generator().manual_seed(12345)
images = []
latents = pipe(
    prompt=lora_prompts[0], 
    width=1024,
    height=1024,
    num_inference_steps=15, 
    generator=generator,
    guidance_scale=4.5,
    # output_type = "pil",
    output_type="latent",
    timestep_to_start_cfg=ts_cutoff,
).images

del pipe.transformer
del pipe



flush()

vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.bfloat16).to(
    "cuda"
)

vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

with torch.no_grad():
    print("Running decoding.")
    latents = FluxPipeline._unpack_latents(latents, 1024, 1024, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")
    image[0].save("image.png")

print("Model inference time ",time.time() - st)
```


```bash
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  5.17it/s]
Loading pipeline components...:  71%|██████████████████████████████████████████████████▋                    | 5/7 [00:00<00:00,  5.75it/s]You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.82it/s]
Loading and fusing lora, please wait...
Quantizing, please wait...
Model quantized!
Model Loading Time is  208.84216332435608
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:59<00:00,  3.96s/it]
Running decoding.
Model inference time  143.85715985298157

```
![screenshot](image.png)
