# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt
import random

from einops import rearrange
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from boltz.data import const
import boltz.model.layers.initialize as init
from boltz.model.loss.diffusion import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    FourierEmbedding,
    PairwiseConditioning,
    SingleConditioning,
)
from boltz.model.modules.transformers import (
    ConditionedTransitionBlock,
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    default,
    log,
)


class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        sigma_data : int, optional
            The standard deviation of the data distribution, by default 16.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.
        atom_encoder_depth : int, optional
            The depth of the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_depth : int, optional
            The depth of the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_depth : int, optional
            The depth of the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        atom_feature_dim : int, optional
            The atom feature dimension, by default 128.
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.
        offload_to_cpu : bool, optional
            Whether to offload the activations to CPU, by default False.

        """

        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        multiplicity=1,
        model_cache=None,
    ):
        s, normed_fourier = self.single_conditioner(
            times=times,
            s_trunk=s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs=s_inputs.repeat_interleave(multiplicity, 0),
        )

        if model_cache is None or len(model_cache) == 0:
            z = self.pairwise_conditioner(
                z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
            )
        else:
            z = None

        # Compute Atom Attention Encoder and aggregation to coarse-grained tokens
        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z,
            r=r_noisy,
            multiplicity=multiplicity,
            model_cache=model_cache,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            z=z,  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
            model_cache=model_cache,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=model_cache,
        )

        return {"r_update": r_update, "token_a": a}


class OutTokenFeatUpdate(Module):
    """Output token feature update"""

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        """Initialize the Output token feature update for confidence model.

        Parameters
        ----------
        sigma_data : float
            The standard deviation of the data distribution.
        token_s : int, optional
            The token dimension, by default 384.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.

        """

        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


class AtomDiffusion(Module):
    """Atom diffusion module"""

    def __init__(
        self,
        score_model_args,
        num_sampling_steps=5,
        sigma_min=0.0004,
        sigma_max=160.0,
        sigma_data=16.0,
        rho=7,
        P_mean=-1.2,
        P_std=1.5,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        coordinate_augmentation=True,
        compile_score=False,
        alignment_reverse_diff=False,
        synchronize_sigmas=False,
        use_inference_model_cache=False,
        accumulate_token_repr=False,
        **kwargs,
    ):
        """Initialize the atom diffusion module.

        Parameters
        ----------
        score_model_args : dict
            The arguments for the score model.
        num_sampling_steps : int, optional
            The number of sampling steps, by default 5.
        sigma_min : float, optional
            The minimum sigma value, by default 0.0004.
        sigma_max : float, optional
            The maximum sigma value, by default 160.0.
        sigma_data : float, optional
            The standard deviation of the data distribution, by default 16.0.
        rho : int, optional
            The rho value, by default 7.
        P_mean : float, optional
            The mean value of P, by default -1.2.
        P_std : float, optional
            The standard deviation of P, by default 1.5.
        gamma_0 : float, optional
            The gamma value, by default 0.8.
        gamma_min : float, optional
            The minimum gamma value, by default 1.0.
        noise_scale : float, optional
            The noise scale, by default 1.003.
        step_scale : float, optional
            The step scale, by default 1.5.
        coordinate_augmentation : bool, optional
            Whether to use coordinate augmentation, by default True.
        compile_score : bool, optional
            Whether to compile the score model, by default False.
        alignment_reverse_diff : bool, optional
            Whether to use alignment reverse diff, by default False.
        synchronize_sigmas : bool, optional
            Whether to synchronize the sigmas, by default False.
        use_inference_model_cache : bool, optional
            Whether to use the inference model cache, by default False.
        accumulate_token_repr : bool, optional
            Whether to accumulate the token representation, by default False.

        """
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.coordinate_augmentation = coordinate_augmentation
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.use_inference_model_cache = use_inference_model_cache

        self.accumulate_token_repr = accumulate_token_repr
        self.token_s = score_model_args["token_s"]
        if self.accumulate_token_repr:
            self.out_token_feat_update = OutTokenFeatUpdate(
                sigma_data=sigma_data,
                token_s=score_model_args["token_s"],
                dim_fourier=score_model_args["dim_fourier"],
            )

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,
        sigma,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )
        return denoised_coords, net_out["token_a"]

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        train_accumulate_token_repr=False,
        **network_condition_kwargs,
    ):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
    
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr = None
        token_a = None

        # gradually denoise
        for sigma_tm, sigma_t, gamma in sigmas_and_gammas:

            atom_coords, atom_coords_denoised = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
                return_second_coords=True,
                second_coords=atom_coords_denoised,
            )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            eps = (
                self.noise_scale
                * sqrt(t_hat**2 - sigma_tm**2)
                * torch.randn(shape, device=self.device)
            )
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised, token_a = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        model_cache=model_cache,
                        **network_condition_kwargs,
                    ),
                )

            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                    )

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def sample_guided(
        self,
        atom_mask,
        guide_coords: torch.Tensor,
        num_sampling_steps=None,
        multiplicity=1,
        guide_strength: float = 0.1,
        train_accumulate_token_repr: bool = False,
        **network_condition_kwargs,
    ):
        """Sample structures while steering towards a guide structure.

        Parameters
        ----------
        atom_mask : torch.Tensor
            Atom mask for the diffused structure.
        guide_coords : torch.Tensor
            Coordinates from the guide PDB (CA atoms). Can have a different
            number of residues than the diffused structure.
        num_sampling_steps : int, optional
            Number of sampling steps.
        multiplicity : int
            Number of diffusion samples.
        guide_strength : float
            Interpolation factor between the diffusion step and the guided
            coordinates.
        train_accumulate_token_repr : bool
            Whether to accumulate token representations during training.
        """

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)

        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*atom_mask.shape, 3)

        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr = None
        token_a = None

        guide_coords = guide_coords.to(self.device)

        def align_template(template, pred):
            d = torch.cdist(pred.unsqueeze(0), template.unsqueeze(0))[0]
            idx = d.argmin(dim=1)
            matched = template[idx]
            weights = torch.ones(pred.shape[0], device=pred.device)
            mask = torch.ones(pred.shape[0], device=pred.device)
            return weighted_rigid_align(
                matched.unsqueeze(0),
                pred.unsqueeze(0),
                weights.unsqueeze(0),
                mask.unsqueeze(0),
            )[0]

        for sigma_tm, sigma_t, gamma in sigmas_and_gammas:
            atom_coords, atom_coords_denoised = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
                return_second_coords=True,
                second_coords=atom_coords_denoised,
            )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            eps = (
                self.noise_scale
                * sqrt(t_hat**2 - sigma_tm**2)
                * torch.randn(shape, device=self.device)
            )
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised, token_a = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        model_cache=model_cache,
                        **network_condition_kwargs,
                    ),
                )

            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                    )

            # steer towards guide coordinates
            aligned_templates = []
            for i in range(atom_coords_noisy.shape[0]):
                aligned_templates.append(align_template(guide_coords, atom_coords_denoised[i]))
            aligned_templates = torch.stack(aligned_templates)
            atom_coords_noisy = (
                (1 - guide_strength) * atom_coords_noisy
                + guide_strength * aligned_templates
            )

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        relative_position_encoding,
        feats,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0]

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        atom_coords = feats["coords"]
        B, N, L = atom_coords.shape[0:3]
        atom_coords = atom_coords.reshape(B * N, L, 3)
        atom_coords = atom_coords.repeat_interleave(multiplicity // N, 0)
        feats["coords"] = atom_coords

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        denoised_atom_coords, _ = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            training=True,
            network_condition_kwargs=dict(
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                relative_position_encoding=relative_position_encoding,
                feats=feats,
                multiplicity=multiplicity,
            ),
        )

        return dict(
            noised_atom_coords=noised_atom_coords,
            denoised_atom_coords=denoised_atom_coords,
            sigmas=sigmas,
            aligned_true_atom_coords=atom_coords,
        )

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
    ):
        denoised_atom_coords = out_dict["denoised_atom_coords"]
        noised_atom_coords = out_dict["noised_atom_coords"]
        sigmas = out_dict["sigmas"]

        resolved_atom_mask = feats["atom_resolved_mask"]
        resolved_atom_mask = resolved_atom_mask.repeat_interleave(multiplicity, 0)

        align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

        align_weights = align_weights * (
            1
            + nucleotide_loss_weight
            * (
                torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
            )
            + ligand_loss_weight
            * torch.eq(atom_type_mult, const.chain_type_ids["NONPOLYMER"]).float()
        )

        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            atom_coords = out_dict["aligned_true_atom_coords"]
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach().float(),
                denoised_atom_coords.detach().float(),
                align_weights.detach().float(),
                mask=resolved_atom_mask.detach().float(),
            )

        # Cast back
        atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
            denoised_atom_coords
        )

        # weighted MSE loss of denoised atom positions
        mse_loss = ((denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(
            dim=-1
        )
        mse_loss = torch.sum(
            mse_loss * align_weights * resolved_atom_mask, dim=-1
        ) / torch.sum(3 * align_weights * resolved_atom_mask, dim=-1)

        # weight by sigma factor
        loss_weights = self.loss_weight(sigmas)
        mse_loss = (mse_loss * loss_weights).mean()

        total_loss = mse_loss

        # proposed auxiliary smooth lddt loss
        lddt_loss = self.zero
        if add_smooth_lddt_loss:
            lddt_loss = smooth_lddt_loss(
                denoised_atom_coords,
                feats["coords"],
                torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                coords_mask=feats["atom_resolved_mask"],
                multiplicity=multiplicity,
            )

            total_loss = total_loss + lddt_loss

        loss_breakdown = dict(
            mse_loss=mse_loss,
            smooth_lddt_loss=lddt_loss,
        )

        return dict(loss=total_loss, loss_breakdown=loss_breakdown)
