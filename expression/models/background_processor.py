"""
Background Processor - Remove, replace, and generate backgrounds.

Uses RMBG v2.0 for foreground extraction and Stable Diffusion XL for
prompt-based background generation.

Modes:
  1. Image input: User uploads background image -> composite with avatar
  2. Video input: User uploads background video -> frame-by-frame composite
  3. Prompt input: Generate background from text via SD inpainting
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_RMBG_AVAILABLE = False
_SD_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available")

try:
    from transformers import AutoModelForImageSegmentation

    _RMBG_AVAILABLE = True
    logger.info("RMBG v2.0 available")
except ImportError:
    logger.warning("RMBG not available - background removal will use fallback")

try:
    from diffusers import StableDiffusionXLInpaintPipeline

    _SD_AVAILABLE = True
    logger.info("Stable Diffusion XL available")
except ImportError:
    logger.warning("Stable Diffusion not available - prompt backgrounds will use solid color")


class BackgroundProcessor:
    """Background removal, replacement, and generation."""

    def __init__(self, device: str = "cuda"):
        self.device = device if _TORCH_AVAILABLE and device == "cuda" else "cpu"
        self._rmbg_model = None
        self._sd_pipeline = None
        self.rmbg_loaded = False
        self.sd_loaded = False

        if _RMBG_AVAILABLE and _TORCH_AVAILABLE:
            self._load_rmbg()

    def _load_rmbg(self) -> None:
        """Load RMBG v2.0 model."""
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
            from torchvision import transforms

            model_name = os.environ.get("RMBG_MODEL", "briaai/RMBG-2.0")
            logger.info("Loading RMBG model: %s", model_name)

            self._rmbg_model = AutoModelForImageSegmentation.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._rmbg_model = self._rmbg_model.to(self.device)
            self._rmbg_model.eval()

            self._rmbg_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            self.rmbg_loaded = True
            logger.info("RMBG v2.0 loaded on %s", self.device)
        except Exception as e:
            logger.error("Failed to load RMBG: %s", e)

    def _load_sd(self) -> None:
        """Load Stable Diffusion XL inpainting pipeline (lazy load)."""
        if self.sd_loaded:
            return
        try:
            import torch
            from diffusers import StableDiffusionXLInpaintPipeline

            model_id = os.environ.get(
                "SD_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
            )
            logger.info("Loading SD XL inpainting: %s", model_id)

            self._sd_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)

            self.sd_loaded = True
            logger.info("SD XL inpainting loaded")
        except Exception as e:
            logger.error("Failed to load SD XL: %s", e)

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        """
        Remove background from a frame using RMBG v2.0.

        Args:
            frame: RGB uint8 numpy array (H, W, 3)

        Returns:
            RGBA uint8 numpy array (H, W, 4) with transparent background
        """
        if self.rmbg_loaded:
            return self._rmbg_remove(frame)
        return self._stub_remove(frame)

    def _rmbg_remove(self, frame: np.ndarray) -> np.ndarray:
        """Real background removal with RMBG v2.0."""
        import torch
        from PIL import Image

        h, w = frame.shape[:2]
        pil_img = Image.fromarray(frame)

        # Transform and predict
        input_tensor = self._rmbg_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self._rmbg_model(input_tensor)[-1]
            mask = torch.sigmoid(preds[0, 0])

        # Resize mask to original dimensions
        mask_np = mask.cpu().numpy()
        from PIL import Image as PILImage

        mask_pil = PILImage.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((w, h), PILImage.LANCZOS)
        alpha = np.array(mask_pil)

        # Composite RGBA
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = frame
        rgba[:, :, 3] = alpha

        return rgba

    def _stub_remove(self, frame: np.ndarray) -> np.ndarray:
        """Stub: add full-opacity alpha channel (no removal)."""
        h, w = frame.shape[:2]
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        return np.concatenate([frame, alpha], axis=2)

    def generate_background(
        self,
        prompt: str,
        width: int = 1920,
        height: int = 1080,
    ) -> np.ndarray:
        """
        Generate a background from a text prompt using SD XL.

        Returns RGB uint8 numpy array (H, W, 3).
        """
        if not self.sd_loaded:
            self._load_sd()

        if self.sd_loaded:
            return self._sd_generate(prompt, width, height)
        return self._stub_generate_bg(prompt, width, height)

    def _sd_generate(
        self, prompt: str, width: int, height: int
    ) -> np.ndarray:
        """Generate background with Stable Diffusion XL."""
        import torch
        from PIL import Image

        # Create blank image and full mask for inpainting (generates entire image)
        init_image = Image.new("RGB", (width, height), (128, 128, 128))
        mask_image = Image.new("L", (width, height), 255)  # full white = inpaint everything

        with torch.no_grad():
            result = self._sd_pipeline(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                width=width,
                height=height,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

        return np.array(result)

    def _stub_generate_bg(
        self, prompt: str, width: int, height: int
    ) -> np.ndarray:
        """Stub: generate a gradient background."""
        bg = np.zeros((height, width, 3), dtype=np.uint8)

        # Create a simple gradient based on prompt keywords
        if "night" in prompt.lower() or "dark" in prompt.lower():
            for y in range(height):
                t = y / height
                bg[y, :] = [int(10 + 20 * t), int(10 + 15 * t), int(30 + 40 * t)]
        elif "ocean" in prompt.lower() or "beach" in prompt.lower():
            for y in range(height):
                t = y / height
                if t < 0.6:
                    bg[y, :] = [int(135 + 50 * t), int(206 - 30 * t), int(235 - 20 * t)]
                else:
                    bg[y, :] = [int(194 + 40 * (t - 0.6)), int(178 + 30 * (t - 0.6)), int(128 + 60 * (t - 0.6))]
        else:
            for y in range(height):
                t = y / height
                bg[y, :] = [int(100 + 80 * t), int(120 + 60 * t), int(140 + 40 * t)]

        return bg

    def composite(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        position: str = "center",
    ) -> np.ndarray:
        """
        Composite a foreground (RGBA) onto a background (RGB).

        Args:
            foreground: RGBA uint8 (H, W, 4)
            background: RGB uint8 (H, W, 3)
            position: "center", "left", "right"

        Returns:
            RGB uint8 (H, W, 3)
        """
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = foreground.shape[:2]

        # Resize foreground to fit background height
        if fg_h != bg_h:
            scale = bg_h / fg_h
            new_w = int(fg_w * scale)
            from PIL import Image

            fg_pil = Image.fromarray(foreground)
            fg_pil = fg_pil.resize((new_w, bg_h), Image.LANCZOS)
            foreground = np.array(fg_pil)
            fg_h, fg_w = foreground.shape[:2]

        # Position
        if position == "center":
            x_offset = (bg_w - fg_w) // 2
        elif position == "left":
            x_offset = bg_w // 10
        elif position == "right":
            x_offset = bg_w - fg_w - bg_w // 10
        else:
            x_offset = (bg_w - fg_w) // 2

        x_offset = max(0, min(x_offset, bg_w - fg_w))

        # Alpha composite
        result = background.copy()
        alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
        fg_rgb = foreground[:, :, :3].astype(np.float32)

        y_start = 0
        y_end = min(fg_h, bg_h)
        x_end = min(x_offset + fg_w, bg_w)
        actual_w = x_end - x_offset

        roi = result[y_start:y_end, x_offset:x_end].astype(np.float32)
        fg_crop = fg_rgb[:y_end, :actual_w]
        alpha_crop = alpha[:y_end, :actual_w]

        blended = fg_crop * alpha_crop + roi * (1 - alpha_crop)
        result[y_start:y_end, x_offset:x_end] = blended.astype(np.uint8)

        return result
