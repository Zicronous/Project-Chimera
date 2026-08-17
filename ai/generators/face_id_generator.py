# ai/generators/face_id_generator.py
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import torch
from PIL import Image

class FaceIDGenerator:
    def __init__(self):
        print("🔄 Loading Face Injection Engine (IP-Adapter)...")
        print("   (This downloads ~3GB on first run. Please wait.)")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load base Stable Diffusion 1.5
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(self.device)
        
        # 2. Load the IP-Adapter (This is the secret sauce for faces)
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        
        # 3. Set how strongly it steals the face (0.5 to 0.8 is best)
        self.pipe.set_ip_adapter_scale(0.7)
        
        # 4. Use a faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        print("✅ Engine ready!")

    def generate(self, prompt: str, face_image_path: str, negative_prompt: str = "", output_path: str = "output.png") -> Image:
        print(f"📸 Loading face: {face_image_path}")
        
        # Load the image and make it square
        face_image = load_image(face_image_path).resize((512, 512))
        
        print("🎨 Generating with face injection... (1-3 mins on CPU)")
        
        with torch.autocast(self.device):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image=face_image,  # The magic
                num_inference_steps=25,
                guidance_scale=7.5,
                width=512,
                height=512
            )
        
        image = result.images[0]
        image.save(output_path)
        print(f"✅ Image saved to {output_path}")
        return image