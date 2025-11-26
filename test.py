from rex_omni import RexOmniWrapper, RexOmniVisualize
from PIL import Image
import os

# Initialize Rex-Omni model
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",
    backend="transformers",
    attn_implementation="eager",
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

# Current folder (where all 34 images are)
input_folder = "."
# Folder to save annotated images
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the current folder
for img_file in os.listdir(input_folder):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, img_file)
        image = Image.open(img_path)

        # Run detection
        detections = rex.detect_objects(image)

        # Visualize and save results
        output_path = os.path.join(output_folder, f"detected_{img_file}")
        RexOmniVisualize.show_detections(image, detections, save_path=output_path)

        print(f"Processed {img_file} -> saved to {output_path}")

print("All images processed successfully!")
