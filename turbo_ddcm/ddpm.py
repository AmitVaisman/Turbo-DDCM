import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

import turbo_ddcm.utils as utils

class DDPM:
    def __init__(self, model_id, torch_dtype, T, device='cuda', seed=42):
        self.device = device
        self.seed = seed
        self.model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device=device)
        self.model.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler', timestep_spacing='linspace', device=device, torch_dtype=torch_dtype)
        self.scheduler_initialized = False
        self.model.scheduler.set_timesteps(T)

    def encode_image(self, img):
        utils.set_seed(self.seed)
        with torch.no_grad():
            w0 = (self.model.vae.encode(img).latent_dist.mode() * self.model.vae.config.scaling_factor)
        return w0

    def decode_img(self, w):
        w_dec = self.model.vae.decode(w / self.model.vae.config.scaling_factor).sample.clamp(-1, 1)
        return w_dec
    
    def encode_text(self, prompt):
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(prompt, self.device, 1, do_classifier_free_guidance=False)
        return prompt_embeds

    def predict_noise(self, x_t: torch.Tensor, timestep, text_encoding):
        with torch.no_grad():
            base_out = self.model.unet(x_t, timestep=timestep, encoder_hidden_states=text_encoding, return_dict=False)[0]

        if self.model.scheduler.config.prediction_type == 'v_prediction':
            alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            noise_pred = (alpha_prod_t ** 0.5) * base_out + (beta_prod_t ** 0.5) * x_t
        elif self.model.scheduler.config.prediction_type == 'epsilon':
            noise_pred = base_out
        else:
            raise NotImplementedError

        return noise_pred

    def x_0_hat_by_denoise_result(self, sample, noise_prediction, timestep):
        # 1. compute alphas, betas
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * noise_prediction) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def get_variance(self, timestep):
        prev_timestep = (
                timestep - self.model.scheduler.config.num_train_timesteps //
                self.model.scheduler.num_inference_steps)
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.model.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def reverse_step(self, model_output, timestep, sample, eta, variance_noise=None, pred_original_sample=None):
        # 1. get previous step value (=t-1)
        prev_timestep = (
                timestep - self.model.scheduler.config.num_train_timesteps //
                self.model.scheduler.num_inference_steps)
        # 2. compute alphas, betas
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.model.scheduler.final_alpha_cumprod

        if pred_original_sample is None:
            beta_prod_t = 1 - alpha_prod_t
            
            # 3. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = self.get_variance(timestep)  # , prev_timestep)
        std_dev_t = eta * variance ** (0.5)
        # Take care of asymetric reverse process (asyrp)
        model_output_direction = model_output
        # 6. compute "direction pointing to x_t" of formula (12) from
        # https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) *
        # model_output_direction
        pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (
            0.5) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from
        # https://arxiv.org/pdf/2010.02502.pdf
        noisy_predicted_original_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample
        prev_sample = noisy_predicted_original_sample + pred_sample_direction
        # 8. Add noice if eta > 0
        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn(model_output.shape, device=self.model.device)
            sigma_z = eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z

        return prev_sample
