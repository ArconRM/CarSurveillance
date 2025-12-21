import numpy as np
import cv2
from PIL import Image, ImageEnhance
from abc import ABC, abstractmethod


class BaseUpscaler(ABC):
    """Base class for all upscalers"""

    @abstractmethod
    def process_pil(self, img: Image.Image) -> Image.Image:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class RealESRGANUpscaler(BaseUpscaler):
    """Real-ESRGAN NCNN upscaler (works on Mac/Windows, fails in Docker)"""

    def __init__(self, gpuid: int = 0):
        try:
            from realesrgan_ncnn_py import Realesrgan
            self.upscaler = Realesrgan(gpuid=gpuid)
            self.available = True
            print(f"✓ Real-ESRGAN loaded (GPU: {gpuid})")
        except Exception as e:
            print(f"✗ Real-ESRGAN unavailable: {e}")
            self.available = False

    def process_pil(self, img: Image.Image) -> Image.Image:
        if not self.available:
            raise RuntimeError("Real-ESRGAN not available")
        return self.upscaler.process_pil(img)

    def get_name(self) -> str:
        return "Real-ESRGAN"


class LanczosUpscaler(BaseUpscaler):
    """Lanczos interpolation upscaler (fast, works everywhere)"""

    def __init__(self, scale: int = 4):
        self.scale = scale
        self.available = True
        print(f"✓ Lanczos upscaler loaded (x{scale})")

    def process_pil(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_size = (w * self.scale, h * self.scale)
        return img.resize(new_size, Image.Resampling.LANCZOS)

    def get_name(self) -> str:
        return f"Lanczos-x{self.scale}"


class EnhancedLanczosUpscaler(BaseUpscaler):
    """Lanczos + sharpening + denoising (best traditional method for plates)"""

    def __init__(self, scale: int = 4):
        self.scale = scale
        self.available = True
        print(f"✓ Enhanced Lanczos upscaler loaded (x{scale})")

    def process_pil(self, img: Image.Image) -> Image.Image:
        # Convert to OpenCV for preprocessing
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Denoise before upscaling
        denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)

        # Upscale with Lanczos
        h, w = denoised.shape[:2]
        upscaled = cv2.resize(denoised, (w * self.scale, h * self.scale),
                              interpolation=cv2.INTER_LANCZOS4)

        # Sharpen
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)

        # Convert back to PIL
        result_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    def get_name(self) -> str:
        return f"Enhanced-Lanczos-x{self.scale}"


class NoUpscaler(BaseUpscaler):
    """No upscaling - baseline comparison"""

    def __init__(self):
        self.available = True
        print("✓ No upscaling (baseline)")

    def process_pil(self, img: Image.Image) -> Image.Image:
        return img

    def get_name(self) -> str:
        return "None"


class UpscalerManager:
    """Manages multiple upscalers with automatic fallback"""

    def __init__(self, preferred_method: str = "auto"):
        self.upscalers = []
        self.current_upscaler = None
        self.preferred_method = preferred_method

        # Try to initialize upscalers in order of preference
        self._initialize_upscalers()

    def _initialize_upscalers(self):
        """Initialize all available upscalers"""

        # Try Real-ESRGAN first (best quality)
        try:
            upscaler = RealESRGANUpscaler()
            if upscaler.available:
                self.upscalers.append(upscaler)
        except Exception as e:
            print(f"Skipping Real-ESRGAN: {e}")

        # Enhanced Lanczos (fast, decent quality for plates)
        self.upscalers.append(EnhancedLanczosUpscaler())

        # Standard Lanczos (fast fallback)
        self.upscalers.append(LanczosUpscaler())

        # No upscaling (baseline)
        self.upscalers.append(NoUpscaler())

        # Set current upscaler based on preference
        if self.preferred_method == "auto":
            self.current_upscaler = self.upscalers[0]
        else:
            for upscaler in self.upscalers:
                if self.preferred_method.lower() in upscaler.get_name().lower():
                    self.current_upscaler = upscaler
                    break
            if not self.current_upscaler:
                self.current_upscaler = self.upscalers[0]

        print(f"\n{'=' * 60}")
        print(f"Active Upscaler: {self.current_upscaler.get_name()}")
        print(f"Available Upscalers: {[u.get_name() for u in self.upscalers]}")
        print(f"{'=' * 60}\n")

    def process_pil(self, img: Image.Image) -> Image.Image:
        """Process image with current upscaler"""
        return self.current_upscaler.process_pil(img)

    def set_upscaler(self, method: str):
        """Switch to a different upscaler"""
        for upscaler in self.upscalers:
            if method.lower() in upscaler.get_name().lower():
                self.current_upscaler = upscaler
                print(f"Switched to: {upscaler.get_name()}")
                return True
        print(f"Upscaler '{method}' not found")
        return False

    def get_available_methods(self) -> list:
        """Get list of available upscaling methods"""
        return [u.get_name() for u in self.upscalers]