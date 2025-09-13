import cv2
import numpy as np
from PIL import Image

def floodfill_white_to_transparent(input_path: str, output_path: str):
    # Read image with alpha if available
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Ensure BGRA (4-channel)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]

    # Create mask (2 pixels bigger than image)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Work on BGR copy (drop alpha)
    bgr = img[:, :, :3].copy()

    # Flood fill starting from (0,0)
    cv2.floodFill(
        bgr, mask, (0, 0),
        (0, 0, 0),                 # dummy color
        (210, 210, 210),           # lower diff
        (210, 210, 210),           # upper diff
        flags=cv2.FLOODFILL_FIXED_RANGE
    )

    # Mask where flood fill happened (mask==1)
    filled_area = mask[1:h+1, 1:w+1] == 1

    # Make flood-filled area transparent
    result = img.copy()
    result[filled_area] = (255, 255, 255, 0)

    # Convert to Pillow for cropping
    pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
    bbox = pil_img.getbbox()
    if bbox:
        pil_img = pil_img.crop(bbox)

    pil_img.save(output_path, "PNG")

# Example usage
floodfill_white_to_transparent("input.png", "output.png")
