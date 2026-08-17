# app/main_v2.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.analyzers.preset_analyzer import PresetAnalyzer
from ai.generators.prompt_builder_v2 import PromptBuilderV2
from ai.generators.cloud_generator import CloudGenerator
from ai.generators.face_swapper import FaceSwapper

def main():
    print("\n" + "="*60)
    print("🧬 PROJECT CHIMERA - Multi-Reference Physique Merger")
    print("="*60 + "\n")

    # 1. User inputs
    name = input("Enter character name (e.g., Naruto, Goku): ").strip()
    if not name:
        return

    universe = input("Enter universe (e.g., Naruto, Dragon Ball): ").strip()
    
    face_path = input("Enter path to user's FACE image: ").strip()
    if not os.path.exists(face_path):
        print(f"❌ Face not found: {face_path}")
        return

    # 2. Path to your folder of body reference images
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    physiques_folder = os.path.join(base_dir, "presets", "physiques")

    if not os.path.exists(physiques_folder):
        print(f"❌ Physique folder not found: {physiques_folder}")
        print("Please create 'presets/physiques/' and put your body images inside.")
        return

    # 3. Analyze ALL images in the folder (BLIP - ~2 seconds per image)
    analyzer = PresetAnalyzer()
    physique_desc = analyzer.describe_physiques(physiques_folder)

    # 4. Build prompt
    builder = PromptBuilderV2()
    prompt = builder.build(name, physique_desc, universe)
    neg_prompt = builder.negative_prompt

    print("\n" + "="*60)
    print("📝 FINAL PROMPT (Sent to AI)")
    print("="*60)
    print(prompt)
    print("="*60 + "\n")

    # 5. Generate the BODY image via Cloud API (5-10 seconds)
    print("🌩️ Generating full body image in the cloud...")
    cloud_gen = CloudGenerator()
    body_output = "temp_body.png"
    cloud_gen.generate(prompt, neg_prompt, body_output)

    # 6. Swap the user's face onto the generated body (2 seconds)
    print("🔄 Swapping user's face onto the body...")
    swapper = FaceSwapper()
    final_output = f"merged_{name.lower()}.png"
    swapper.swap(face_path, body_output, final_output)

    # 7. Clean up
    if os.path.exists(body_output):
        os.remove(body_output)

    print(f"\n🎉 Done! Final merged image saved as: {final_output}")

if __name__ == "__main__":
    main()