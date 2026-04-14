from diffusers import FluxPipeline
import torch
import numpy as np

import turbo_ddcm.utils as utils
from huggingface_hub import login

# ----- Based on Vonderfecht, 2025 ------

class Flux:
    s_guidance_scale = 1

    # in order not to load the model several times, for example in roundtrip.
    s_models = {}

    def __init__(self, model_id, torch_dtype, T, image_height, image_width, device='cuda'):
        self.device = device
        self.dtype = torch_dtype

        if (model_id, torch_dtype) not in Flux.s_models.keys():
            Flux.s_models[(model_id, torch_dtype)] = FluxPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)

        self.model = Flux.s_models[(model_id, torch_dtype)]

        self.model.scheduler.num_inference_steps = T
        self.model.scheduler.timesteps = torch.tensor(utils.evenly_spaced(utils.SCHEDULER, T), device=self.device)

        self.image_height = image_height
        self.image_width = image_width
        self.image_ids = self.get_image_ids(image_height, image_width)


    @staticmethod
    def prepare_ref_latents(vae_latent):
        batch_size = vae_latent.shape[0]
        num_channels = vae_latent.shape[1]
        height = vae_latent.shape[2]
        width = vae_latent.shape[3]

        latent = vae_latent.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latent = latent.permute(0, 2, 4, 1, 3, 5)

        latent = latent.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latent

    @staticmethod
    def unpack_latents(latents, height, width, vae_scale_factor=8):
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        channels = latents.shape[-1] // 4  # Each patch is 2x2 so divide by 4

        # Reshape back to patches
        latents = latents.view(latents.shape[0], latent_height // 2, latent_width // 2, channels, 2, 2)
        # Permute dimensions back to image format
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        # Flatten patch dimensions
        latents = latents.reshape(latents.shape[0], channels, latent_height, latent_width)

        return latents

    @torch.no_grad()
    def decode_img(self, x_t):
        x_t_latent = self.unpack_latents(
            x_t,
            height=self.image_height,
            width=self.image_width,
            vae_scale_factor=8
        )

        latents = (x_t_latent / self.model.vae.config.scaling_factor) + self.model.vae.config.shift_factor
        image = self.model.vae.decode(latents, return_dict=False)[0]
        return image

    @torch.no_grad()
    def get_image_ids(self, image_height, image_width):
        height = 2 * (int(image_height) // (self.model.vae_scale_factor * 2))
        width = 2 * (int(image_width) // (self.model.vae_scale_factor * 2))

        return self.model._prepare_latent_image_ids(1, height // 2, width // 2, self.device, self.dtype)

    @torch.no_grad()
    def encode_image(self, img):
        img = img.to(self.dtype)
        ref_latent = self.model.vae.encode(img).latent_dist.mode().to(self.dtype)
        ref_latent = (ref_latent - self.model.vae.config.shift_factor) * self.model.vae.config.scaling_factor
        return ref_latent

    @torch.no_grad()
    def decode_image(self, latents):
        latents = (latents / self.model.vae.config.scaling_factor) + self.model.vae.config.shift_factor
        image = self.model.vae.decode(latents, return_dict=False)[0]
        return image

    @torch.no_grad()
    def encode_text(self, prompt):
        prompt_embeds, pooled_prompt_embeds, text_ids = (
            self.model.encode_prompt(prompt=prompt, prompt_2=None, device=self.device))

        output_dict = {'encoder_hidden_states' : prompt_embeds,
                       'pooled_projections' : pooled_prompt_embeds,
                       'txt_ids' : text_ids}

        return output_dict

    def get_prev_timestep(self, timestep):
        timesteps: torch.Tensor = self.model.scheduler.timesteps
        idx = torch.where(timesteps == timestep)[0].item()

        if idx + 1 == timesteps.numel():
            return torch.tensor(0, device=self.device)

        return timesteps[idx + 1]

    def get_timestep_snr(self, timestep):
        if timestep.item() == 0:
            return torch.inf
        sigmas = np.arange(1000) / 1000
        sigma = torch.tensor(sigmas[int(timestep.item())], device=self.device)
        return utils.sigma_to_snr(sigma)

    @torch.no_grad()
    def p_mu_and_std(self, noisy_latent, noise_prediction, current_snr, prev_snr):
        alpha_prod_t, beta_prod_t = utils.get_alpha_prod_and_beta_prod(current_snr)
        alpha_prod_t_prev, beta_prod_t_prev = utils.get_alpha_prod_and_beta_prod(prev_snr)
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (noisy_latent - beta_prod_t ** (0.5) * noise_prediction) / alpha_prod_t ** (0.5)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
                pred_original_sample_coeff * pred_original_sample
                + current_sample_coeff * noisy_latent
        )

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        std = variance ** (0.5)
        return pred_prev_sample, std

    @torch.no_grad()
    def predict_v(self, latents, timestep, scheduler_kwargs):
        guidance = torch.tensor([Flux.s_guidance_scale], device=self.device)
        guidance = guidance.expand(latents.shape[0])

        timestep = timestep.expand(latents.shape[0])

        flow_noise_pred = self.model.transformer(
            hidden_states=latents,
            timestep= timestep / 1000,
            guidance=guidance,
            joint_attention_kwargs=None,
            return_dict=False,
            img_ids=self.image_ids,
            **scheduler_kwargs
        )[0]

        return flow_noise_pred

    @torch.no_grad()
    def x_0_hat_by_denoise_result(self, latents, model_prediction, timestep):
        current_snr = self.get_timestep_snr(timestep)
        # 1. compute alphas, betas
        alpha_prod_t, beta_prod_t = utils.get_alpha_prod_and_beta_prod(current_snr)
        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_prediction) / alpha_prod_t ** (0.5)
        return pred_original_sample

    @torch.no_grad()
    def reverse_step(self, noise_prediction, timestep, sample, eta, variance_noise):
        current_snr = self.get_timestep_snr(timestep)
        prev_snr = self.get_timestep_snr(self.get_prev_timestep(timestep))

        if eta == 1:
            p_mu, p_std = self.p_mu_and_std(sample, noise_prediction, current_snr, prev_snr)
            return p_mu + p_std * variance_noise

        elif eta == 0:
            alpha_prod_t, beta_prod_t = utils.get_alpha_prod_and_beta_prod(current_snr)
            alpha_prod_t_prev, beta_prod_t_prev = utils.get_alpha_prod_and_beta_prod(prev_snr)
            beta_prod_t = 1 - alpha_prod_t

            # 3. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            model_output = noise_prediction
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output

            # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

            # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            latent = (alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction)
            return latent.to(sample.dtype)

        else:
            assert NotImplementedError
            return None

    @torch.no_grad()
    def predict_noise(self, x_t, timestep, text_encoding):
        snr = self.get_timestep_snr(timestep)

        # Get scaling factor to convert between DDPM and OT flow spaces
        ot_flow_to_ddpm_factor = utils.get_ot_flow_to_ddpm_factor(snr)

        # Convert DDPM space latent to OT flow space
        ot_flow_latent = x_t / ot_flow_to_ddpm_factor

        ot_flow_noise_pred = self.predict_v(ot_flow_latent, timestep, text_encoding)

        sigma = 1 / (snr + 1)
        alpha_prod_t, beta_prod_t = utils.get_alpha_prod_and_beta_prod(snr)
        x0_hat = ot_flow_latent - sigma * ot_flow_noise_pred

        # back-calculate the DDPM noise pred from noisy_latent and x0_hat
        ddpm_noise_pred = (x_t - alpha_prod_t**0.5 * x0_hat) / beta_prod_t**0.5
        # Convert prediction back to DDPM space
        #ddpm_noise_pred = ot_flow_noise_pred * ot_flow_to_ddpm_factor * (alpha_prod_t ** 0.5)

        return ddpm_noise_pred.to(x_t.dtype)

