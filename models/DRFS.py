from typing import Optional, Tuple, Union
import torch
from tqdm import tqdm
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


def lr_cos(k, N, alpha_1, alpha_N):
    return alpha_N + 0.5 * (alpha_1 - alpha_N) * (1 + np.cos(np.pi * (k-1)/(N-1)))


def lr_cos_smoother(k, N, alpha_1, alpha_N, power=2.5):
    progress = (k - 1) / (N - 1)
    progress = progress**power
    return alpha_N + 0.5 * (alpha_1 - alpha_N) * (1 + np.cos(np.pi * progress))


def lr_hump_beta(k: int, N: int, alpha_max: float,
                 a: float = 3.0, b: float = 6.0) -> float:
    """
    Beta-distribution shaped learning rate: starts at 0, peaks before midpoint, ends at 0.
    The peak position is (a-1)/(a+b-2) < 0.5 when a < b.
    """
    if not (1 <= k <= N):
        raise ValueError("k must be in [1, N]")
    x = (k - 1) / (N - 1)
    pdf = x**(a - 1) * (1 - x)**(b - 1)
    peak = ((a - 1) / (a + b - 2))**(a - 1) * ((b - 1)/(a + b - 2))**(b - 1)
    return alpha_max * pdf / peak


def lr_hump_tail_beta(k: int, N: int, alpha_max: float, beta: float,
                      a: float = 3.0, b: float = 6.0) -> float:
    """Beta-distribution hump with a linear tail: starts at 0, ends at beta."""
    x = (k - 1) / (N - 1)
    hump = lr_hump_beta(k, N, alpha_max - beta, a, b)
    tail = beta * x
    return hump + tail


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """Forward process in flow-matching."""
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample
    return sample


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def calc_v_sd3(pipe, src_tgt_latent_model_input, src_tgt_prompt_embeds,
               src_tgt_pooled_prompt_embeds, src_guidance_scale,
               tgt_guidance_scale, t):
    timestep = t.expand(src_tgt_latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred_src_tgt = pipe.transformer(
            hidden_states=src_tgt_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tgt_prompt_embeds,
            pooled_projections=src_tgt_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tgt_noise_pred_uncond, tgt_noise_pred_text = noise_pred_src_tgt.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tgt = tgt_noise_pred_uncond + tgt_guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_uncond)

    return noise_pred_src, noise_pred_tgt


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds,
                guidance, text_ids, latent_image_ids, t):
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        latents = latents.to(prompt_embeds.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred


def DRFS(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tgt_prompt,
    negative_prompt,
    T_steps: int = 50,
    B: int = 1,
    src_guidance_scale: float = 3.5,
    tgt_guidance_scale: float = 13.5,
    num_steps: int = 50,
    eta: float = 0,
    scheduler_strategy: str = "random",
    lr: float = 0.1,
    optim: str = 'SGD',
):
    """
    Delta Rectified Flow Sampling — memory-optimized implementation.

    Uses direct gradient assignment (no autograd graph) + fp32 accumulation
    to achieve the same results as the autograd version with much less VRAM.
    """
    zt_edit = x_src.float().clone().requires_grad_(True)

    if optim == 'SGD':
        if isinstance(lr, float):
            optimizer = torch.optim.SGD([zt_edit], lr=lr)
        else:
            optimizer = torch.optim.SGD([zt_edit], lr=0.02)
    elif optim == 'SGD_Nesterov':
        optimizer = torch.optim.SGD([zt_edit], lr=lr, momentum=0.9, nesterov=True)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop([zt_edit], lr=lr, alpha=0.9)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW([zt_edit], lr=lr)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam([zt_edit], lr=lr)
    else:
        raise ValueError(f'Optimizer {optim} not supported.')

    device = x_src.device
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)

    with torch.no_grad():
        pipe._guidance_scale = src_guidance_scale
        (
            src_prompt_embeds,
            src_negative_prompt_embeds,
            src_pooled_prompt_embeds,
            src_negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=src_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            device=device,
        )

        pipe._guidance_scale = tgt_guidance_scale
        (
            tgt_prompt_embeds,
            tgt_negative_prompt_embeds,
            tgt_pooled_prompt_embeds,
            tgt_negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=tgt_prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            device=device,
        )

        src_tgt_prompt_embeds = torch.cat(
            [src_negative_prompt_embeds, src_prompt_embeds,
             tgt_negative_prompt_embeds, tgt_prompt_embeds], dim=0
        )
        src_tgt_pooled_prompt_embeds = torch.cat(
            [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
             tgt_negative_pooled_prompt_embeds, tgt_pooled_prompt_embeds], dim=0
        )

    # Main optimization loop — uses zt_edit.data to avoid autograd graph
    if scheduler_strategy == "random":
        for i in range(num_steps):
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(B):
                ind = torch.randint(2, T_steps - 1, (1,)).item()
                t = timesteps[ind]
                t_i = t / 1000

                fwd_noise = torch.randn_like(x_src, device=device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tgt = (1 - t_i) * zt_edit.data + t_i * fwd_noise + eta * t_i * (zt_edit.data - x_src)

                src_tgt_latent_model_input = (
                    torch.cat([zt_src, zt_src, zt_tgt, zt_tgt])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tgt)
                )

                with torch.no_grad():
                    src_tgt_latent_model_input_fp16 = src_tgt_latent_model_input.half()
                    Vt_src, Vt_tgt = calc_v_sd3(
                        pipe,
                        src_tgt_latent_model_input_fp16,
                        src_tgt_prompt_embeds,
                        src_tgt_pooled_prompt_embeds,
                        src_guidance_scale,
                        tgt_guidance_scale,
                        t,
                    )
                V_delta_avg += (Vt_tgt - Vt_src) / B

            optimizer.zero_grad()
            zt_edit.grad = V_delta_avg.float()
            optimizer.step()

    else:  # descending
        alpha_T_steps = (timesteps[T_steps-2]/1000 - timesteps[T_steps-1] / 1000) / 1.6

        for i, t in enumerate(timesteps):
            if T_steps - i > num_steps:
                continue

            t_i = t / 1000
            if i + 1 < len(timesteps):
                t_im1 = timesteps[i + 1] / 1000
            else:
                t_im1 = torch.zeros_like(t_i)

            alpha_i = 2.2 * lr_hump_tail_beta(i+1, T_steps+28, alpha_T_steps/1.6, alpha_T_steps/4, a=10, b=8)
            V_delta_avg = torch.zeros_like(x_src)

            for k in range(B):
                fwd_noise = torch.randn_like(x_src, device=device)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tgt = (1 - t_i) * zt_edit.data + t_i * fwd_noise + eta * i/T_steps * t_i * (zt_edit.data - x_src)

                src_tgt_latent_model_input = (
                    torch.cat([zt_src, zt_src, zt_tgt, zt_tgt])
                    if pipe.do_classifier_free_guidance
                    else (zt_src, zt_tgt)
                )

                with torch.no_grad():
                    src_tgt_latent_model_input_fp16 = src_tgt_latent_model_input.half()
                    Vt_src, Vt_tgt = calc_v_sd3(
                        pipe,
                        src_tgt_latent_model_input_fp16,
                        src_tgt_prompt_embeds,
                        src_tgt_pooled_prompt_embeds,
                        src_guidance_scale,
                        tgt_guidance_scale,
                        t,
                    )
                V_delta_avg += (Vt_tgt - Vt_src) / B

            current_lr_FE = t_i - t_im1
            current_lr = alpha_i
            if lr == "FE":
                optimizer.param_groups[0]['lr'] = float(current_lr_FE)
            elif isinstance(lr, str):
                optimizer.param_groups[0]['lr'] = current_lr

            grad = V_delta_avg.float() + (1 - eta * i / T_steps) * (zt_edit.data - x_src.float())

            optimizer.zero_grad()
            zt_edit.grad = grad
            optimizer.step()

    return zt_edit
