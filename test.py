from PIL import Image
from rex_omni import RexOmniWrapper

# Initialize model (already working for you)
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",
    backend="transformers",
    attn_implementation="eager"
)

categories = [
    "brick kiln",
    "circular kiln",
    "rectangular kiln",
    "chimney structure"
]

import glob
image_paths = glob.glob("gujrat_images/*.jpg")  # Update the folder if needed

for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")
    results = rex.inference(images=image, task="detection", categories=categories)
    print(f"Processed {img_path}, Results: {results}")
    # To save, add visualization logic here if available
