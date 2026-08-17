import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as get_insight_model
from PIL import Image
import numpy as np

class FaceSwapper:
    def __init__(self):
        print("🔄 Loading Face Swapper (insightface)...")
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = get_insight_model('inswapper_128.onnx')
        print("✅ Face Swapper ready!")

    def swap(self, source_image_path: str, target_image_path: str, output_path: str) -> Image:
        print(f"📸 Source face: {source_image_path}")
        print(f"🎯 Target body: {target_image_path}")
        
        source_img = cv2.imread(source_image_path)
        target_img = cv2.imread(target_image_path)
        
        source_faces = self.app.get(source_img)
        target_faces = self.app.get(target_img)
        
        if len(source_faces) == 0:
            raise ValueError("No face found in source image!")
        if len(target_faces) == 0:
            raise ValueError("No face found in target image!")
        
        swapped = self.swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)
        
        result = Image.fromarray(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB))
        result.save(output_path)
        print(f"✅ Face-swapped image saved to {output_path}")
        return result
