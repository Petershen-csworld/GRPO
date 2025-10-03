# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py
import sys;sys.path.insert(0, '/mnt/sphere/2025intern/has052/GRPO/show-o2')

from typing import Any, Dict, List, Optional, Union, Callable
import torch
import numpy as np
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .showo2_sde_with_logprob import sde_step_with_logprob
from models import omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import denorm
from tqdm import tqdm 
from PIL import Image 

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model
@torch.no_grad()
def pipeline_with_logprob(
    model = None,
    vae_model = None,
    prompt = None,
    text_tokenizer  = None,
    showo_token_ids = None,
    weight_type = None,
    device = None,
    scheduler = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    latents: Optional[torch.FloatTensor] = None,
    noise_level: float = 0.5,
    num_images_per_prompt: int = 1,
):
    
    if not isinstance(prompt, list):
        prompt = [prompt]
    image_latent_dim = 16
    latent_height = 27
    latent_width = 27
    patch_size = 2
    num_t2i_image_tokens = 730
    max_seq_len = 1024
    max_text_len = max_seq_len - num_t2i_image_tokens - 4
    pad_id = text_tokenizer.pad_token_id
    bos_id = showo_token_ids['bos_id']
    eos_id = showo_token_ids['eos_id']
    boi_id = showo_token_ids['boi_id']
    eoi_id = showo_token_ids['eoi_id']
    img_pad_id = showo_token_ids['img_pad_id']

    batch_size = len(prompt)
    z = latents 
    if z is None:
         z = torch.randn((
                batch_size * num_images_per_prompt,
                image_latent_dim, 
                latent_height * patch_size,
                latent_width * patch_size
         )).to(device).float()
    # 6. Prepare image embeddings print

    prompt = [p for p in prompt for _ in range(num_images_per_prompt)]
    batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
    prepare_gen_input(
                    prompt, text_tokenizer, num_t2i_image_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                    max_text_len, device
                )

    if guidance_scale > 0:
            # [conditional generation, unconditional generation]
            #z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
            block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                              max_seq_len,
                                              modality_positions,
                                              device).to(weight_type)
            # print("shape of block mask", block_mask.shape) [8, 1, 4252, 4352]
    else:
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions
            block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                              max_seq_len,
                                              modality_positions,
                                              device).to(weight_type)

    all_latents = [z.unsqueeze(1)] 
    all_log_probs = []
    all_timesteps = []
    # Denoising loop

    for _, ori_timestep in tqdm(enumerate(scheduler.timesteps[1:-1])):
            t = 1 - ori_timestep/1000
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            t = t.expand(z.shape[0] * 2).to(z.dtype).to(device)

            if guidance_scale > 0.0:

                _, v = model(text_tokens,
                            image_latents= torch.cat([z, z], dim=0),
                            t=t,
                            attention_mask=block_mask,
                            modality_positions=modality_positions,
                            first_frame_as_cond=False,
                            only_denoise_last_image=False,
                            guidance_scale=guidance_scale,
                            output_hidden_states=True,
                            max_seq_len=max_seq_len)
                v_cond, v_uncond = torch.chunk(v, 2)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            else:
                _, v = model(text_tokens,
                            image_latents=z,
                            t=t,
                            attention_mask=block_mask,
                            modality_positions=modality_positions,
                            first_frame_as_cond=False,
                            only_denoise_last_image=False,
                            guidance_scale=0.0,
                            output_hidden_states=True,
                            max_seq_len=max_seq_len)
                
            
            unsqueezed_timestep = ori_timestep.unsqueeze(0).unsqueeze(0).expand(z.shape[0], 1, 1)
            z, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                scheduler, 
                v.float(), 
                unsqueezed_timestep[0,0].cpu(), 
                z.float(),
                noise_level=noise_level,
            )
            #z = z.to(weight_type)
            all_latents.append(z.unsqueeze(1)) # [batch_size, 16, 54, 54]
            all_log_probs.append(log_prob.unsqueeze(1)) 
            all_timesteps.append(unsqueezed_timestep)

    samples = z
    samples = samples.unsqueeze(2)
    images = vae_model.batch_decode(samples)
    images = images.squeeze(2)
    images = denorm(images)
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images, all_latents,  all_log_probs, all_timesteps, text_tokens, modality_positions, block_mask 

