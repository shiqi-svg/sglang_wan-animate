# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

from typing import List

import numpy as np
import PIL
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WanVideoProcessor:
    def __init__(self, vae_scale_factor=8):
        self.vae_scale_factor = vae_scale_factor

    def _resize_frames(
        self,
        video: List[PIL.Image.Image],
        target_height: int | None,
        target_width: int | None,
    ) -> List[PIL.Image.Image]:
        if target_height is None or target_width is None:
            return video

        resized = []
        for img in video:
            if isinstance(img, PIL.Image.Image):
                pil_img = img
            else:
                pil_img = PIL.Image.fromarray(np.array(img))

            if pil_img.size != (target_width, target_height):
                pil_img = pil_img.resize(
                    (target_width, target_height), PIL.Image.Resampling.LANCZOS
                )
            resized.append(pil_img)

        return resized

    def preprocess_video(
        self,
        video: List[PIL.Image.Image],
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(video, list):
            video = [video]

        video = self._resize_frames(video, target_height, target_width)

        frames = []
        for img in video:
            arr = np.array(img)
            tensor = torch.from_numpy(arr).float() / 255.0
            tensor = tensor.permute(2, 0, 1)  # CHW
            frames.append(tensor)

        video_tensor = torch.stack(frames, dim=1)
        video_tensor = 2.0 * video_tensor - 1.0

        # (1, C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0)

        return video_tensor


class VideoProcessingStage(PipelineStage):
    def __init__(self, vae_scale_factor: int = 8) -> None:
        super().__init__()
        self.video_processor = WanVideoProcessor(vae_scale_factor)

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        pose_video = batch.extra.get("pose_video")
        pose_video_tensor = self.video_processor.preprocess_video(
            pose_video,
            target_height=batch.height,
            target_width=batch.width,
        ).to(get_local_torch_device(), dtype=torch.float32)
        batch.extra["pose_video"] = pose_video_tensor

        face_video = batch.extra.get("face_video")
        face_video_tensor = self.video_processor.preprocess_video(
            face_video,
            target_height=512,
            target_width=512,
        ).to(get_local_torch_device(), dtype=torch.float32)
        batch.extra["face_video"] = face_video_tensor

        bg_video = batch.extra.get("bg_video")
        if bg_video is not None:
            bg_video_tensor = self.video_processor.preprocess_video(
                bg_video,
                target_height=batch.height,
                target_width=batch.width,
            ).to(get_local_torch_device(), dtype=torch.float32)
            batch.extra["bg_video"] = bg_video_tensor

        mask_video = batch.extra.get("mask_video")
        if mask_video is not None:
            if not isinstance(mask_video, list):
                mask_video = [mask_video]
            mask_frames = []
            for mask in mask_video:
                arr = np.array(mask)
                if arr.ndim == 2:
                    arr = arr[:, :, None]
                if arr.shape[2] != 1:
                    arr = arr[:, :, :1]
                tensor = torch.from_numpy(arr).float()
                if tensor.max() > 1:
                    tensor = tensor / 255.0
                tensor = tensor.permute(2, 0, 1)
                mask_frames.append(tensor)

            mask_tensor = torch.stack(mask_frames, dim=1)
            mask_tensor = mask_tensor.unsqueeze(0)
            batch.extra["mask_video"] = mask_tensor.to(
                get_local_torch_device(), dtype=torch.float32
            )

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()
        result.add_check("pose", batch.extra.get("pose_video"), V.list_not_empty)
        result.add_check("face", batch.extra.get("face_video"), V.list_not_empty)
        if batch.extra.get("bg_video") is not None:
            result.add_check("bg", batch.extra.get("bg_video"), V.list_not_empty)
        if batch.extra.get("mask_video") is not None:
            result.add_check("mask", batch.extra.get("mask_video"), V.list_not_empty)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        result.add_check(
            "pose", batch.extra.get("pose_video"), [V.is_tensor, V.with_dims(5)]
        )
        result.add_check(
            "face", batch.extra.get("face_video"), [V.is_tensor, V.with_dims(5)]
        )
        if batch.extra.get("bg_video") is not None:
            result.add_check(
                "bg", batch.extra.get("bg_video"), [V.is_tensor, V.with_dims(5)]
            )
        if batch.extra.get("mask_video") is not None:
            result.add_check(
                "mask", batch.extra.get("mask_video"), [V.is_tensor, V.with_dims(5)]
            )
        return result
