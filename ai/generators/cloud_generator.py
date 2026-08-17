# ai/generators/cloud_generator.py
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

class CloudGenerator:
    def __init__(self):
        print("🔄 Loading local Stable Diffusion 1.5... (This uses your downloaded models)")
        print("   ⏱️ First load takes 2-3 minutes. Be patient!")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use the same model you already downloaded
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Load the pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            safety_checker=None,          # Disable safety checker for speed
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory-efficient attention if using GPU
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            
        print("✅ Local Stable Diffusion loaded successfully!")

    def generate(self, prompt: str, negative_prompt: str = "", output_path: str = "output.png") -> Image:
        print("🎨 Generating image locally...")
        print("   ⏱️ This takes 2-5 minutes on CPU, 30-60 seconds on GPU.")
        
        with torch.autocast(self.device):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,      # Higher = better quality but slower
                guidance_scale=7.5,
                width=512,
                height=512
            )
        
        image = result.images[0]
        image.save(output_path)
        print(f"✅ Image saved to {output_path}")
        return image