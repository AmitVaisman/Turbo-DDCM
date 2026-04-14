import torch
import time

from turbo_ddcm.ddpm import DDPM
from turbo_ddcm.flux import Flux
import turbo_ddcm.utils as utils
from turbo_ddcm.bit_stream_lex_enc import BitStreamEncoder


class TurboDDCM():
    s_encoding_eta = 1
    s_denoising_eta = 0

    def __init__(self, model_id, T, K, M, seed=42, float32=False, device='cuda'):
        self.device = 'cuda'
        self.seed = seed
        self.torch_dtype = torch.float32 if float32 else torch.float16

        self.T = T

        if model_id in ['stabilityai/stable-diffusion-2-1-base', 'Manojb/stable-diffusion-2-1-base']:
            latent_space_shape = [4, 64, 64]
            self.H, self.W = 512, 512
            self.model = DDPM(model_id, self.torch_dtype, self.T, device)
        elif model_id == 'stabilityai/stable-diffusion-2-1':
            latent_space_shape = [4, 96, 96]
            self.H, self.W = 768, 768
            self.model = DDPM(model_id, self.torch_dtype, self.T, device)
        elif model_id == 'black-forest-labs/FLUX.1-dev':
            latent_space_shape = [16, 128, 128]
            self.H, self.W = 1024, 1024
            self.model = Flux(model_id, self.torch_dtype, self.T, self.H, self.W, device)
        else:
            raise ValueError("not supported model")

        torch.manual_seed(self.seed)
        self.x_T = torch.randn(latent_space_shape, device=self.device, dtype=self.torch_dtype).unsqueeze(0)
        self.x_T = self.model.prepare_ref_latents(self.x_T)

        # compression step by step members (sbs - step by step)
        self.comp_sbs_started = False
        self.null_text_encode = self.model.encode_text("")

        self.x_T_denoised = self.model.predict_noise(self.x_T, self.model.model.scheduler.timesteps[0], self.null_text_encode).to(torch.float32)

        self.K = K
        self.M = M
        self.C = 1 # as described in the paper - we use C=1
        self.no_bits_steps, bpp = utils.get_no_bits_steps(self.T, self.K, self.M, self.C, self.H, self.W)
        print(f"Expected BPP: {bpp}")

        self.bit_stream_obj = BitStreamEncoder(self.K, self.M, self.C)

        # to save memory and not allocating on each step
        self.codebook = torch.empty(self.x_T_denoised.numel(), self.K, dtype=self.torch_dtype, device=self.device)

    def compress(self, image, weight_pixel_vector):
        assert image.shape == torch.Size([1, 3, self.H, self.W])

        self.compress_start(image, weight_pixel_vector)
        for i in range(self.T - 1 - self.no_bits_steps): # minus one since last step is without bits
            self.compress_encode_step()
        compression_end_time = time.process_time()

        for _ in range(self.no_bits_steps):
            self.comp_sbs_current_x_t = self.compress_denoise_step(self.comp_sbs_current_x_t, self.get_comp_sbs_current_total_step_idx(), TurboDDCM.s_denoising_eta, is_last_step=False)
            self.comp_sbs_steps_without_bits_counter += 1

        self.comp_sbs_current_x_t = self.compress_denoise_step(self.comp_sbs_current_x_t, self.get_comp_sbs_current_total_step_idx(), TurboDDCM.s_denoising_eta, is_last_step=True)
        self.comp_sbs_steps_without_bits_counter += 1
        return self.compress_end(), compression_end_time


    def compress_start(self, image, weight_pixel_vector):
        self.comp_sbs_started = True
        self.comp_sbs_steps_with_bits_counter = 0
        self.comp_sbs_steps_without_bits_counter = 0

        image = image.to(self.torch_dtype)
        self.comp_sbs_enc_img = self.model.prepare_ref_latents(self.model.encode_image(image))
        self.comp_sbs_current_x_t = self.x_T

        self.comp_sbs_pac_ind = False
        if weight_pixel_vector is not None:
            self.comp_sbs_pac_ind = True
            assert self.H == self.W and self.H % self.x_T.shape[-1] == 0 # squared kernel
            kernel_size = self.H // self.x_T.shape[-1]
            weight_pixel_vector = weight_pixel_vector.unsqueeze(0).unsqueeze(0).to(self.torch_dtype)
            self.comp_sbs_weight_latent_vector = utils.down_sample_mask(weight_pixel_vector, kernel_size, self.device).view(-1)

    def compress_encode_step(self):
        assert self.comp_sbs_started

        # timesteps are reversed by default
        idx = self.get_comp_sbs_current_total_step_idx()
        t = self.model.model.scheduler.timesteps[idx]

        current_x_t = self.comp_sbs_current_x_t.to(self.torch_dtype)
        current_epsilon_hat = self.x_T_denoised if idx == 0 else self.model.predict_noise(current_x_t, t, self.null_text_encode)

        current_x_0_hat = self.model.x_0_hat_by_denoise_result(current_x_t, current_epsilon_hat, t).to(self.torch_dtype)
        residual = (self.comp_sbs_enc_img - current_x_0_hat).to(self.torch_dtype)

        if self.comp_sbs_pac_ind:
            residual = (self.comp_sbs_weight_latent_vector * residual.view(-1) ).view(*residual.shape)

        torch.manual_seed(self.seed + idx + 1)
        self.codebook.normal_()
        best_noise, chosen_indexes, coeff_indices = self.get_iteration_best_noise_from_codebook_optimized(self.codebook, residual)

        current_x_t = self.model.reverse_step(current_epsilon_hat, t, current_x_t, TurboDDCM.s_encoding_eta, best_noise)
        chosen_indexes_list = [chosen_index.item() for chosen_index in chosen_indexes]
        coeff_indices_list = [coeff_indice.item() for coeff_indice in coeff_indices]
        self.bit_stream_obj.add(chosen_indexes_list, coeff_indices_list)
        self.comp_sbs_steps_with_bits_counter += 1
        self.comp_sbs_current_x_t = current_x_t

    def compress_denoise_step(self, current_x_t, step_idx, eta, is_last_step):
        t = self.model.model.scheduler.timesteps[step_idx]
        current_epsilon_hat = self.model.predict_noise(current_x_t.to(self.torch_dtype), t, self.null_text_encode)
        noise = torch.zeros_like(current_epsilon_hat) if is_last_step else torch.randn(current_epsilon_hat.shape)
        noise = noise.to(self.device).to(self.torch_dtype)
        current_x_t = self.model.reverse_step(current_epsilon_hat, t, current_x_t, eta, noise)
        return current_x_t

    def compress_end(self):
        assert self.comp_sbs_started and (self.T == self.get_comp_sbs_current_total_step_idx())
        self.comp_sbs_started = False
        encoding = self.bit_stream_obj.get_encoding()
        self.bit_stream_obj.clear()
        reconstruction = self.model.decode_img(self.comp_sbs_current_x_t.to(self.torch_dtype))
        return reconstruction, encoding


    def get_iteration_best_noise_from_codebook_optimized(self, codebook, residual):
        assert self.C == 1 # using torch.sign we assume C = 1

        flat_residual = residual.view(-1)
        x = torch.zeros(self.K, dtype=codebook.dtype, device=codebook.device)

        inner_products_with_residual = (codebook.T @ flat_residual).view(-1)
        abs_inner_products_with_residual = inner_products_with_residual.abs()
        _, top_s_indices = torch.topk(abs_inner_products_with_residual, self.M)  # Shape: s

        x[top_s_indices] = torch.sign(inner_products_with_residual[top_s_indices])
        coeffs_indices = (x[top_s_indices] > 0).long() # -1 to 0, 1 to 1

        best_noise = codebook @ x
        best_noise /= best_noise.std()

        return best_noise.view(residual.shape), top_s_indices, coeffs_indices


    @torch.no_grad()
    def decompress(self, encoding):
        decoded_list = self.bit_stream_obj.decode(encoding)

        current_x_t = self.x_T
        x_est = torch.zeros(self.K, dtype=torch.float16, device=self.device)

        assert self.C == 1 # used in the mapping from -1 to 0 and 1 to 1 in the loop below

        steps_counter = 0
        for _ in range(self.T - 1 - self.no_bits_steps): # decode steps
            t = self.model.model.scheduler.timesteps[steps_counter]

            current_x_t = current_x_t.to(self.torch_dtype)
            current_epsilon_hat = self.x_T_denoised if steps_counter == 0 else self.model.predict_noise(current_x_t, t, self.null_text_encode)
            decoded_noise_indexes, decoded_coeffs_indexes = decoded_list[steps_counter]

            torch.manual_seed(self.seed + steps_counter + 1)
            self.codebook.normal_()

            x_est.zero_()
            x_est[decoded_noise_indexes] = 2*torch.tensor(decoded_coeffs_indexes, dtype=self.torch_dtype, device=self.device) - 1 # -1 to 0 and 1 to 1
            best_noise = (self.codebook @ x_est).view(self.x_T.shape)
            best_noise /= best_noise.std()
            current_x_t = self.model.reverse_step(current_epsilon_hat, t, current_x_t, TurboDDCM.s_encoding_eta, best_noise)
            steps_counter += 1

        for _ in range(self.no_bits_steps): # denoise steps
            current_x_t = self.compress_denoise_step(current_x_t, steps_counter, TurboDDCM.s_denoising_eta, is_last_step=False)
            steps_counter += 1

        current_x_t = self.compress_denoise_step(current_x_t, steps_counter, TurboDDCM.s_denoising_eta, is_last_step=True)
        steps_counter += 1

        assert steps_counter == self.T
        w_dec = self.model.decode_img(current_x_t.to(self.torch_dtype))
        return w_dec

    def get_comp_sbs_current_total_step_idx(self):
        assert self.comp_sbs_started
        return self.comp_sbs_steps_with_bits_counter + self.comp_sbs_steps_without_bits_counter
