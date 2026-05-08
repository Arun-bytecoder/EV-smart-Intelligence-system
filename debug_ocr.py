import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import easyocr

# Load the plate crop we already saved
plate = cv2.imread('data/outputs/plate_crop.jpg')
print(f"Plate size: {plate.shape}")

reader = easyocr.Reader(['en'], gpu=False)

# Test 1: Raw plate crop
print("\n--- Test 1: Raw crop ---")
results = reader.readtext(plate, detail=1)
for r in results:
    print(f"Text: {r[1]} | Conf: {r[2]:.2f}")

# Test 2: Upscaled
print("\n--- Test 2: Upscaled x3 ---")
upscaled = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
results2 = reader.readtext(upscaled, detail=1)
for r in results2:
    print(f"Text: {r[1]} | Conf: {r[2]:.2f}")

# Test 3: Grayscale + threshold
print("\n--- Test 3: Grayscale + threshold ---")
gray    = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
results3  = reader.readtext(binary, detail=1)
for r in results3:
    print(f"Text: {r[1]} | Conf: {r[2]:.2f}")

# Save processed versions
cv2.imwrite('data/outputs/plate_upscaled.jpg', upscaled)
cv2.imwrite('data/outputs/plate_binary.jpg', binary)
print("\nSaved: plate_upscaled.jpg and plate_binary.jpg")