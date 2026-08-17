# ai/generators/prompt_builder_v2.py

class PromptBuilderV2:
    def __init__(self):
        self.style_map = {
            "naruto": "Naruto anime style, Masashi Kishimoto art style, bold lines",
            "dragon ball": "Akira Toriyama anime style, bold lines, vibrant primary colors",
            "league of legends": "Riot Games splash art style, realistic fantasy painting",
            "marvel": "American comic book style, dynamic superhero pose",
            "default": "professional concept art, highly detailed, sharp focus, cinematic lighting"
        }

    def build(self, character_name: str, physique_description: str, universe: str = "") -> str:
        style = self.style_map.get("default")
        if universe:
            for key in self.style_map:
                if key in universe.lower():
                    style = self.style_map[key]
                    break

        # Build the prompt - the longer 'physique_description' now contains all the merged data
        prompt = (f"{style}, a character with a body and clothes exactly like the following description: {physique_description}. "
                  f"The character is named {character_name}. The face should be generic and blank, do not focus on facial features. "
                  f"Full body shot, dynamic action pose, masterpiece, 8k resolution, high detail")

        self.negative_prompt = "extra limbs, bad anatomy, deformed, blurry, lowres, watermark, text, cropped, ugly, disfigured, distorted face, mismatched face"
        return prompt