from typing import Any, Union

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class WanAnimateConditioningStage(PipelineStage):
    def __init__(self, vae: Any):
        super().__init__()
        self.vae = vae

    def encode(
        self,
        video_condition: torch.Tensor,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        # Encode Image
        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output: DiagonalGaussianDistribution = self.vae.encode(
                video_condition
            )

        generator = batch.generator

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()

        latent_condition = self.retrieve_latents(
            encoder_output, generator, sample_mode=sample_mode
        )
        latent_condition = server_args.pipeline_config.postprocess_vae_encode(
            latent_condition, self.vae
        )

        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=latent_condition.device,
                dtype=latent_condition.dtype,
                vae=self.vae,
            )
        )

        # apply shift & scale if needed
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(latent_condition.device)

        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(latent_condition.device)

        latent_condition -= shift_factor
        latent_condition = latent_condition * scaling_factor

        # output = server_args.pipeline_config.postprocess_image_latent(
        #     latent_condition, batch
        # )
        return latent_condition

    def retrieve_latents(
        self,
        encoder_output: DiagonalGaussianDistribution,
        generator: torch.Generator | None = None,
        sample_mode: str = "sample",
    ):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def get_i2v_mask(
        self,
        batch_size: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        mask_len: int = 1,
        mask_pixel_values: torch.Tensor | None = None,
        dtype: torch.dtype = None,
        device: Union[str, torch.device] = "cuda",
    ) -> torch.Tensor:
        """Build I2V mask in latent space.

        Matches Wan2.2 Lightning `get_i2v_mask` behavior:
        - Base mask is zeros, or provided `mask_pixel_values` (replacement mode)
        - Force the first `mask_len` *pixel frames* to 1
        - Repeat the first frame 4x and reshape into 4 channels.

        Args:
            mask_pixel_values: Optional per-pixel-frame mask at latent spatial resolution,
                shape [B, T_pix, H_lat, W_lat], where T_pix == (latent_t - 1) * 4 + 1.
        """

        target_t_pix = (latent_t - 1) * 4 + 1
        if mask_pixel_values is None:
            mask_lat_size = torch.zeros(
                batch_size,
                1,
                target_t_pix,
                latent_h,
                latent_w,
                dtype=dtype,
                device=device,
            )
        else:
            if mask_pixel_values.ndim != 4:
                raise ValueError(
                    "mask_pixel_values must have shape [B, T, H_lat, W_lat]"
                )
            if mask_pixel_values.shape[0] != batch_size:
                raise ValueError(
                    f"mask_pixel_values batch is {mask_pixel_values.shape[0]} but expected {batch_size}"
                )
            if mask_pixel_values.shape[1] != target_t_pix:
                raise ValueError(
                    f"mask_pixel_values T is {mask_pixel_values.shape[1]} but expected {target_t_pix}"
                )
            if mask_pixel_values.shape[2] != latent_h or mask_pixel_values.shape[3] != latent_w:
                raise ValueError(
                    "mask_pixel_values spatial size must match latent_h/latent_w"
                )
            mask_lat_size = mask_pixel_values.to(device=device, dtype=dtype).unsqueeze(1).clone()

        mask_lat_size[:, :, :mask_len] = 1
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=4)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, 4, latent_h, latent_w
        ).transpose(
            1, 2
        )  # [B, C = 1, 4 * T_lat, H_lat, W_lat] --> [B, C = 4, T_lat, H_lat, W_lat]

        return mask_lat_size

    def prepare_prev_segment_cond_latents(
        self,
        batch,
        server_args,
        prev_segment_cond_video,
        background_video: torch.Tensor | None = None,
        mask_video: torch.Tensor | None = None,
        replace_flag: bool = False,
        batch_size: int = 1,
        segment_frame_length: int = 77,
        height: int = 720,
        width: int = 1280,
        prev_segment_cond_frames: int = 1,
        interpolation_mode: str = "bicubic",
        dtype=torch.float32,
        device="cuda",
    ) -> torch.Tensor:
        # prev_segment_cond_video shape: (B, C, T, H, W) in pixel space if supplied
        # background_video shape: (B, C, T, H, W) in pixel space (replacement mode)
        # mask_video shape: (B, 1, T, H, W) in [0,1] (replacement mode)
        first_frame = prev_segment_cond_video is None
        cond_frames_shape = (
            batch_size,
            3,
            prev_segment_cond_frames,
            height,
            width,
        )  # In pixel space
        if prev_segment_cond_video is None:
            prev_segment_cond_video = torch.zeros(
                cond_frames_shape, dtype=dtype, device=device
            )
        else:
            assert prev_segment_cond_video.shape == cond_frames_shape

        data_batch_size, channels, _, segment_height, segment_width = (
            prev_segment_cond_video.shape
        )
        num_latent_frames = (segment_frame_length - 1) // 4 + 1
        latent_height = height // 8
        latent_width = width // 8
        if segment_height != height or segment_width != width:
            print(
                f"Interpolating prev segment cond video from ({segment_width}, {segment_height}) to ({width}, {height})"
            )
            # Perform a 4D (spatial) rather than a 5D (spatiotemporal) reshape, following the original code
            prev_segment_cond_video = prev_segment_cond_video.transpose(1, 2).flatten(
                0, 1
            )  # [B * T, C, H, W]
            prev_segment_cond_video = F.interpolate(
                prev_segment_cond_video, size=(height, width), mode=interpolation_mode
            )
            prev_segment_cond_video = prev_segment_cond_video.unflatten(
                0, (batch_size, -1)
            ).transpose(1, 2)

        # Build the pixel-space video that will be VAE-encoded into y_reft.
        # - Standard animate:   [refer_t (or zeros for first segment)] + zeros
        # - Replacement animate: [refer_t (if any)] + bg for the remaining frames
        if replace_flag:
            if background_video is None or mask_video is None:
                raise ValueError(
                    "background_video and mask_video are required when replace_flag is True"
                )
            if background_video.shape != (
                batch_size,
                3,
                segment_frame_length,
                height,
                width,
            ):
                raise ValueError(
                    "background_video must have shape (B, 3, T, H, W) matching the segment"
                )
            if mask_video.shape != (
                batch_size,
                1,
                segment_frame_length,
                height,
                width,
            ):
                raise ValueError(
                    "mask_video must have shape (B, 1, T, H, W) matching the segment"
                )

            mask_len = prev_segment_cond_frames if not first_frame else 0
            if mask_len > 0:
                full_segment_cond_video = torch.cat(
                    [
                        prev_segment_cond_video.to(dtype=dtype),
                        background_video[:, :, mask_len:].to(dtype=dtype),
                    ],
                    dim=2,
                )
            else:
                full_segment_cond_video = background_video.to(dtype=dtype)
        else:
            remaining_segment_frames = segment_frame_length - prev_segment_cond_frames
            remaining_segment = torch.zeros(
                batch_size,
                channels,
                remaining_segment_frames,
                height,
                width,
                dtype=dtype,
                device=device,
            )

            # Prepend the conditioning frames from the previous segment to the remaining segment video in the frame dim
            prev_segment_cond_video = prev_segment_cond_video.to(dtype=dtype)
            full_segment_cond_video = torch.cat(
                [prev_segment_cond_video, remaining_segment], dim=2
            )

        prev_segment_cond_latents = self.encode(
            full_segment_cond_video, batch, server_args
        )

        # Prepare I2V mask
        if replace_flag:
            # Official code uses mask_pixel_values = 1 - mask_video and downsamples to latent spatial size.
            # Expected mask_video is [0,1].
            mask_pixel_values = 1.0 - mask_video.to(dtype=dtype)
            mask_pixel_values = mask_pixel_values.permute(0, 2, 1, 3, 4).reshape(
                batch_size * segment_frame_length, 1, height, width
            )
            mask_pixel_values = F.interpolate(
                mask_pixel_values,
                size=(latent_height, latent_width),
                mode="nearest",
            )
            mask_pixel_values = mask_pixel_values.reshape(
                batch_size, segment_frame_length, 1, latent_height, latent_width
            )[:, :, 0]

            prev_segment_cond_mask = self.get_i2v_mask(
                batch_size,
                num_latent_frames,
                latent_height,
                latent_width,
                mask_len=prev_segment_cond_frames if not first_frame else 0,
                mask_pixel_values=mask_pixel_values,
                dtype=dtype,
                device=device,
            )
        else:
            prev_segment_cond_mask = self.get_i2v_mask(
                batch_size,
                num_latent_frames,
                latent_height,
                latent_width,
                mask_len=prev_segment_cond_frames if not first_frame else 0,
                dtype=dtype,
                device=device,
            )

        # Prepend cond I2V mask to prev segment cond latents along channel dimension
        prev_segment_cond_latents = torch.cat(
            [prev_segment_cond_mask, prev_segment_cond_latents], dim=1
        )
        return prev_segment_cond_latents

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self.vae = self.vae.to(get_local_torch_device())

        clip_len = server_args.pipeline_config.clip_len
        refert_num = server_args.pipeline_config.refert_num
        cur_segment = batch.extra.get("cur_segment")
        start_frame = cur_segment * (clip_len - refert_num)
        end_frame = start_frame + clip_len

        pose_video_tensor = batch.extra.get("pose_video")[
            :, :, start_frame:end_frame, :, :
        ]
        face_video_tensor = batch.extra.get("face_video")[
            :, :, start_frame:end_frame, :, :
        ]

        bg_video_tensor = batch.extra.get("bg_video")
        if bg_video_tensor is not None:
            bg_video_tensor = bg_video_tensor[:, :, start_frame:end_frame, :, :]

        mask_video_tensor = batch.extra.get("mask_video")
        if mask_video_tensor is not None:
            mask_video_tensor = mask_video_tensor[:, :, start_frame:end_frame, :, :]
        if cur_segment == 0:
            prev_segment_cond_video = None
        else:
            # Use unclamped decoded frames in [-1,1] to align with Wan2.2 Lightning,
            # which feeds previous decoded frames back into VAE.encode directly.
            prev_segment_cond_video = batch.extra.get("all_frames_raw")
            if prev_segment_cond_video is None:
                # Fallback to normalized frames if raw is not available.
                prev_segment_cond_video = (
                    batch.extra.get("all_frames")[:, :, -refert_num:].clone().detach()
                )
                if (
                    prev_segment_cond_video.dtype.is_floating_point
                    and prev_segment_cond_video.min() >= 0
                    and prev_segment_cond_video.max() <= 1
                ):
                    prev_segment_cond_video = prev_segment_cond_video * 2 - 1
            else:
                prev_segment_cond_video = (
                    prev_segment_cond_video[:, :, -refert_num:].clone().detach()
                )

        pose_latents_no_ref = self.encode(pose_video_tensor, batch, server_args)

        batch.extra["pose_hidden_states"] = pose_latents_no_ref
        batch.extra["face_pixel_values"] = face_video_tensor

        batch.extra["prev_segment_cond_latents"] = (
            self.prepare_prev_segment_cond_latents(
                batch,
                server_args,
                prev_segment_cond_video,
                background_video=bg_video_tensor,
                mask_video=mask_video_tensor,
                replace_flag=getattr(batch, "replace_flag", False),
                segment_frame_length=clip_len,
                height=batch.height,
                width=batch.width,
                prev_segment_cond_frames=refert_num,
                device=get_local_torch_device(),
                dtype=pose_latents_no_ref.dtype,
            )
        )

        self.maybe_free_model_hooks()
        return batch
