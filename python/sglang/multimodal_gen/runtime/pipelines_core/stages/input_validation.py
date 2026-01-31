# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Input validation stage for diffusion pipelines.
"""
import os
import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2

from sglang.multimodal_gen.configs.pipeline_configs import WanI2V480PConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_Animate_14B_Config
from sglang.multimodal_gen.runtime.models.vision_utils import load_image, load_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import best_output_size, try_load_image

from decord import VideoReader
from PIL import Image

logger = init_logger(__name__)

# Alias for convenience
V = StageValidators


# TODO: since this might change sampling params after logging, should be do this beforehand?


class InputValidationStage(PipelineStage):
    """
    Stage for validating and preparing inputs for diffusion pipelines.

    This stage validates that all required inputs are present and properly formatted
    before proceeding with the diffusion process.

    In this stage, input image and output image may be resized
    """

    def __init__(self, vae_image_processor=None):
        super().__init__()
        self.vae_image_processor = vae_image_processor

    def _generate_seeds(self, batch: Req, server_args: ServerArgs):
        """Generate seeds for the inference"""
        seed = batch.seed
        num_videos_per_prompt = batch.num_outputs_per_prompt

        assert seed is not None
        seeds = [seed + i for i in range(num_videos_per_prompt)]
        batch.seeds = seeds

        # Create generators based on generator_device parameter
        # Note: This will overwrite any existing batch.generator
        generator_device = batch.generator_device

        if generator_device == "cpu":
            device_str = "cpu"
        else:
            device_str = current_platform.device_type

        batch.generator = [
            torch.Generator(device_str).manual_seed(seed) for seed in seeds
        ]

    def preprocess_condition_image(
        self,
        batch: Req,
        server_args: ServerArgs,
        condition_image_width,
        condition_image_height,
    ):
        """
        preprocess condition image
        NOTE: condition image resizing is only allowed in InputValidationStage
        """
        if batch.condition_image is not None and (
            server_args.pipeline_config.task_type == ModelTaskType.I2I
            or server_args.pipeline_config.task_type == ModelTaskType.TI2I
        ):
            # calculate new condition image size
            if not isinstance(batch.condition_image, list):
                batch.condition_image = [batch.condition_image]

            processed_images = []
            final_image = batch.condition_image[-1]
            config = server_args.pipeline_config
            config.preprocess_vae_image(batch, self.vae_image_processor)

            for img in batch.condition_image:
                size = config.calculate_condition_image_size(img, img.width, img.height)
                if size is not None:
                    width, height = size
                    img, _ = config.preprocess_condition_image(
                        img, width, height, self.vae_image_processor
                    )

                processed_images.append(img)

            batch.condition_image = processed_images
            calculated_size = config.prepare_calculated_size(final_image)

            # adjust output image size
            if calculated_size is not None:
                calculated_width, calculated_height = calculated_size
                width = batch.width or calculated_width
                height = batch.height or calculated_height
                multiple_of = (
                    server_args.pipeline_config.vae_config.get_vae_scale_factor() * 2
                )
                width = width // multiple_of * multiple_of
                height = height // multiple_of * multiple_of
                batch.width = width
                batch.height = height

        elif server_args.pipeline_config.task_type == ModelTaskType.TI2V:
            # duplicate with vae_image_processor
            # further processing for ti2v task
            if isinstance(
                batch.condition_image, list
            ):  # not support multi image input yet.
                batch.condition_image = batch.condition_image[0]

            img = batch.condition_image
            ih, iw = img.height, img.width
            patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size
            vae_stride = (
                server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
            )
            dh, dw = patch_size[1] * vae_stride, patch_size[2] * vae_stride
            max_area = 704 * 1280
            ow, oh = best_output_size(iw, ih, dw, dh, max_area)

            scale = max(ow / iw, oh / ih)
            img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
            logger.info("resized img height: %s, img width: %s", img.height, img.width)

            # center-crop
            x1 = (img.width - ow) // 2
            y1 = (img.height - oh) // 2
            img = img.crop((x1, y1, x1 + ow, y1 + oh))
            assert img.width == ow and img.height == oh

            # to tensor
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)
            img = img.unsqueeze(0)
            batch.height = oh
            batch.width = ow
            # TODO: should we store in a new field: pixel values?
            batch.condition_image = img

        elif isinstance(server_args.pipeline_config, WanI2V480PConfig):
            # TODO: could we merge with above?
            # resize image only, Wan2.1 I2V
            if isinstance(batch.condition_image, list):
                batch.condition_image = batch.condition_image[
                    0
                ]  # not support multi image input yet.

            max_area = server_args.pipeline_config.max_area
            aspect_ratio = condition_image_height / condition_image_width
            mod_value = (
                server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
                * server_args.pipeline_config.dit_config.arch_config.patch_size[1]
            )
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

            batch.condition_image = batch.condition_image.resize((width, height))
            batch.height = height
            batch.width = width

    @staticmethod
    def _get_valid_len(real_len: int, clip_len: int, overlap: int) -> int:
        real_clip_len = clip_len - overlap
        last_clip_num = (real_len - overlap) % real_clip_len
        extra = 0 if last_clip_num == 0 else real_clip_len - last_clip_num
        return real_len + extra

    @staticmethod
    def _inputs_padding(array, target_len):
        from copy import deepcopy

        if len(array) == 0:
            raise ValueError("video inputs must not be empty")
        if len(array) == 1:
            return [deepcopy(array[0]) for _ in range(target_len)]

        idx = 0
        flip = False
        target_array = []
        while len(target_array) < target_len:
            target_array.append(deepcopy(array[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(array) - 1:
                flip = not flip
        return target_array[:target_len]

    def padding_resize(self, img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
        ori_height = img_ori.shape[0]
        ori_width = img_ori.shape[1]
        channel = img_ori.shape[2]

        img_pad = np.zeros((height, width, channel))
        if channel == 1:
            img_pad[:, :, 0] = padding_color[0]
        else:
            img_pad[:, :, 0] = padding_color[0]
            img_pad[:, :, 1] = padding_color[1]
            img_pad[:, :, 2] = padding_color[2]

        if (ori_height / ori_width) > (height / width):
            new_width = int(height / ori_height * ori_width)
            img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
            padding = int((width - new_width) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[:, padding: padding + new_width, :] = img
        else:
            new_height = int(width / ori_width * ori_height)
            img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
            padding = int((height - new_height) / 2)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]  
            img_pad[padding: padding + new_height, :, :] = img

        img_pad = np.uint8(img_pad)

        return img_pad

    def prepare_source(self, src_pose_path, src_face_path, src_ref_path):
        pose_video_reader = VideoReader(src_pose_path)
        pose_len = len(pose_video_reader)
        pose_idxs = list(range(pose_len))
        cond_images = pose_video_reader.get_batch(pose_idxs).asnumpy()

        face_video_reader = VideoReader(src_face_path)
        face_len = len(face_video_reader)
        face_idxs = list(range(face_len))
        face_images = face_video_reader.get_batch(face_idxs).asnumpy()
        height, width = cond_images[0].shape[:2]
        refer_images = try_load_image(src_ref_path)
        refer_images = self.padding_resize(refer_images, height=height, width=width)
        return cond_images, face_images, refer_images
    
    def prepare_source_for_replace(self, src_bg_path, src_mask_path):
        bg_video_reader = VideoReader(src_bg_path)
        bg_len = len(bg_video_reader)
        bg_idxs = list(range(bg_len))
        bg_images = bg_video_reader.get_batch(bg_idxs).asnumpy()

        mask_video_reader = VideoReader(src_mask_path)
        mask_len = len(mask_video_reader)
        mask_idxs = list(range(mask_len))
        mask_images = mask_video_reader.get_batch(mask_idxs).asnumpy()
        mask_images = mask_images[:, :, :, 0] / 255
        return bg_images, mask_images

    @staticmethod
    def _load_wan_animate_videos(batch: Req):
        if batch.pose_video_path is not None or batch.face_video_path is not None:
            if batch.pose_video_path is None or batch.face_video_path is None:
                raise ValueError(
                    "pose_video_path and face_video_path must both be provided"
                )
            return load_video(batch.pose_video_path), load_video(batch.face_video_path)

        pose_video = batch.extra.get("pose_video")
        face_video = batch.extra.get("face_video")
        if pose_video is None or face_video is None:
            raise ValueError("pose_video and face_video must be provided")

        return pose_video, face_video

    def verify_wan_animate(self, batch: Req, server_args: ServerArgs) -> None:
        config = server_args.pipeline_config
        refert_num = config.refert_num
        if refert_num not in (1, 5):
            raise ValueError("refert_num must be 1 or 5")

        # pose_video, face_video = self._load_wan_animate_videos(batch) 原来的代码

        #手动读取文件
        face_path = batch.extra.get("face_video_path") 
        pose_path = batch.extra.get("pose_video_path") 
        bg_path = batch.extra.get("bg_video_path")
        mask_path = batch.extra.get("mask_video_path")
        ref_path = batch.extra.get("ref_image_path")
        # logger.info(f"查看一下: face_path: {face_path}, pose_path: {pose_path}, bg_path: {bg_path}, mask_path: {mask_path}, ref_path: {ref_path}")
        pose_video, face_video, refer_video = self.prepare_source(src_pose_path=pose_path, src_face_path=face_path, src_ref_path=ref_path)
        

        real_frame_len = len(pose_video)
        if real_frame_len == 0 or len(face_video) == 0:
            raise ValueError("pose_video and face_video must not be empty")

        # clip_len = config.clip_len
        # segment_len = clip_len - refert_num
        # if segment_len <= 0:
        #     raise ValueError("clip_len must be greater than refert_num")
        # target_len = self._get_valid_len(real_frame_len, clip_len, overlap=refert_num)
        clip_len = config.clip_len
        print(f"Original clip_len: {clip_len}")
        if clip_len == -1:
            assert refert_num == 1, "Auto calculation of clip_len only supports refert_num=1."
            # clip_len = int((real_frame_len//(6-2))//4)*4+5
            target_len = real_frame_len-(real_frame_len-1)%4
            divide_n = (target_len-100)//99 + 1 #进行divide_n+1次滑动
            divide_n = max(0, divide_n)
            clip_len = (target_len+divide_n)//(divide_n+1) if divide_n>0 else target_len
            logger.info(f"Auto calculating clip_len: {clip_len}, divide_n: {divide_n}.")
            if clip_len > 100:
                clip_len = 97
                target_len = (clip_len-1)*(divide_n)+clip_len
                if abs(target_len - real_frame_len) > clip_len/2:
                    target_len += clip_len
            config.clip_len = clip_len
            logger.info(f"Auto setting clip_len to {clip_len}.")
        else:
            target_len = self._get_valid_len(real_frame_len, clip_len, overlap=refert_num)
        
        print(f"Using clip_len: {clip_len}, target_len: {target_len}.")
        segment_len = clip_len - refert_num
        if segment_len <= 0:
            raise ValueError("clip_len must be greater than refert_num")

        logger.info(f"查看shape: cond video shape: {pose_video.shape}, face video shape: {face_video.shape}, refer video shape: {refer_video.shape}")
        pose_video = self._inputs_padding(pose_video, target_len)
        face_video = self._inputs_padding(face_video, target_len)
        logger.info(f"查看padding后shape: cond video shape: {len(pose_video)}, face video shape: {len(face_video)}")
        if bg_path is not None and mask_path is not None:
            bg_video, mask_video = self.prepare_source_for_replace(src_bg_path=bg_path, src_mask_path=mask_path)
            bg_video_tensor = self._inputs_padding(bg_video, target_len)
            mask_video_tensor = self._inputs_padding(mask_video, target_len)
            logger.info(f"查看padding后shape: bg video shape: {len(bg_video_tensor)}, mask video shape: {len(mask_video_tensor)}")

        batch.num_frames = target_len
        batch.extra["real_frame_len"] = real_frame_len
        batch.extra["pose_video"] = pose_video
        batch.extra["face_video"] = face_video
        batch.extra["bg_video"] = bg_video_tensor if bg_path is not None else None
        batch.extra["mask_video"] = mask_video_tensor if mask_path is not None else None
        batch.extra["num_segments"] = target_len // segment_len
        batch.extra["cur_segment"] = 0
        logger.info(f"查看一下: pose_video长度{len(batch.extra.get('pose_video'))}, face_video长度{len(batch.extra.get('face_video'))}, num_segments: {batch.extra.get('num_segments')}")
        logger.info(f"查看一下: mask_video长度{len(batch.extra.get('mask_video')) if batch.extra.get('mask_video') is not None else 'None'}, face_video形状{len(batch.extra.get('face_video'))}")
        logger.info(f"查看一下: pose_video形状{batch.extra.get('pose_video')[0].shape}, face_video形状{batch.extra.get('face_video')[0].shape}")
        logger.info(f"查看一下: mask_video形状{batch.extra.get('mask_video')[0].shape if batch.extra.get('mask_video') is not None else 'None'}, face_video形状{batch.extra.get('face_video')[0].shape}")
        

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Validate and prepare inputs.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The validated batch information.
        """

        self._generate_seeds(batch, server_args)

        # Ensure prompt is properly formatted
        if batch.prompt is None and batch.prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided")

        # Ensure negative prompt is properly formatted if using classifier-free guidance
        if (
            batch.do_classifier_free_guidance
            and batch.negative_prompt is None
            and batch.negative_prompt_embeds is None
        ):
            raise ValueError(
                "For classifier-free guidance, either `negative_prompt` or "
                "`negative_prompt_embeds` must be provided"
            )

        # Validate number of inference steps
        if batch.num_inference_steps <= 0:
            raise ValueError(
                f"Number of inference steps must be positive, but got {batch.num_inference_steps}"
            )

        # Validate guidance scale if using classifier-free guidance
        if batch.do_classifier_free_guidance and batch.guidance_scale < 0:
            raise ValueError(
                f"Guidance scale must be positive, but got {batch.guidance_scale}"
            )

        # for i2v, get image from image_path
        # @TODO(Wei) hard-coded for wan2.2 5b ti2v for now. Should put this in image_encoding stage
        if batch.image_path is not None:
            if isinstance(batch.image_path, list):
                batch.condition_image = []
                for path in batch.image_path:
                    if path.endswith(".mp4"):
                        image = load_video(path)[0]
                    else:
                        image = load_image(path)
                    batch.condition_image.append(image)

                # Use the first image for size reference
                condition_image_width = batch.condition_image[0].width
                condition_image_height = batch.condition_image[0].height
                batch.original_condition_image_size = (
                    condition_image_width,
                    condition_image_height,
                )
            else:
                if batch.image_path.endswith(".mp4"):
                    image = load_video(batch.image_path)[0]
                else:
                    image = load_image(batch.image_path)
                batch.condition_image = image
                condition_image_width, condition_image_height = (
                    image.width,
                    image.height,
                )
                batch.original_condition_image_size = image.size

            self.preprocess_condition_image(
                batch, server_args, condition_image_width, condition_image_height
            )

        # if height or width is not specified at this point, set default to 720p
        default_height = 720
        default_width = 1280
        if batch.height is None and batch.width is None:
            batch.height = default_height
            batch.width = default_width
        elif batch.height is None:
            batch.height = batch.width * default_height // default_width
        elif batch.width is None:
            batch.width = batch.height * default_width // default_height

        if isinstance(server_args.pipeline_config, Wan2_2_Animate_14B_Config):
            self.verify_wan_animate(batch, server_args)

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify input validation stage inputs."""
        result = VerificationResult()
        result.add_check("seed", batch.seed, [V.not_none, V.non_negative_int])
        result.add_check(
            "num_videos_per_prompt", batch.num_outputs_per_prompt, V.positive_int
        )
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt)
            or V.list_not_empty(batch.prompt_embeds),
        )

        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check(
            "guidance_scale",
            batch.guidance_scale,
            lambda x: not batch.do_classifier_free_guidance or V.non_negative_float(x),
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify input validation stage outputs."""
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("seeds", batch.seeds, V.list_not_empty)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        return result
