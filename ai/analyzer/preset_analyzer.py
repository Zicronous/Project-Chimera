# ai/analyzers/preset_analyzer.py
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class PresetAnalyzer:
    def __init__(self):
        print("📸 Loading Vision Analyzer (BLIP) to read your physique images...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
    def describe_image(self, image_path: str) -> str:
        """Takes a single image path, returns a clean description."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=50, num_beams=4)
        desc = self.processor.decode(out[0], skip_special_tokens=True)
        # Remove face descriptors to keep it body-focused
        desc = desc.replace("man", "character").replace("woman", "character").replace("person", "character")
        return desc

    def describe_physiques(self, folder_path: str) -> str:
        """
        Scans the folder, analyzes ALL images, and merges their descriptions
        into ONE super detailed body description.
        """
        if not os.path.exists(folder_path):
            return "a muscular character in a dynamic fighting pose"
            
        # 1. Find all image files in the folder
        image_files = [f for f in os.listdir(folder_path) 
                       if f.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))]
        
        if not image_files:
            return "a muscular character in a dynamic fighting pose"
            
        print(f"🔍 Analyzing {len(image_files)} physique reference images...")
        
        # 2. Analyze each image
        all_descriptions = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            desc = self.describe_image(img_path)
            all_descriptions.append(desc)
            print(f"   - {img_file}: {desc}")
            
        # 3. Merge them into one powerful description
        # We just combine them - more keywords = better for Stable Diffusion.
        # Remove exact duplicates to avoid repetition.
        unique_descs = list(set(all_descriptions))
        
        if len(unique_descs) == 1:
            final_desc = unique_descs[0]
        else:
            # Join them with "and" to create a rich composite
            final_desc = " and ".join(unique_descs)
            
        # Clean up any weird double spaces
        final_desc = " ".join(final_desc.split())
        
        print(f"\n✅ Merged Body Description: {final_desc}\n")
        return final_desc