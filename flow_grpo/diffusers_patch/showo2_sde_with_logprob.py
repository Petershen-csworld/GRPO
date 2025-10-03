# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
def sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):

    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]

    sigma = 1 - scheduler.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = 1 - scheduler.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    
    # after casting, the sigma is monotically increasing
    dt =  sigma_prev - sigma
    std_dev_t = torch.sqrt((1 - sigma) / sigma)*noise_level
    prev_sample_mean = sample*(1-std_dev_t**2*dt/(2*(1 - sigma)))+model_output*(1+std_dev_t**2*sigma/(2*(1 - sigma)))*dt
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(dt) * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(dt))**2))
        - torch.log(std_dev_t * torch.sqrt(dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


    return prev_sample, log_prob, prev_sample_mean, std_dev_t