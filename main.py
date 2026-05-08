import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import EVDetectionPipeline
import cv2

pipeline = EVDetectionPipeline()

# Read and resize image for better detection
frame = cv2.imread('data/samples/test_vehicle.jpg')
frame = cv2.resize(frame, (1280, 720))
cv2.imwrite('data/samples/test_vehicle.jpg', frame)

annotated, results = pipeline.process_image('data/samples/test_vehicle.jpg')

print(f"\nTotal detections: {len(results)}")

if not results:
    print("No vehicles detected in image.")
else:
    for i, r in enumerate(results):
        print(f"\n--- Detection #{i+1} ---")
        print(f"Decision    : {r.decision}")
        print(f"Reg Number  : {r.registration_number or 'Not extracted'}")
        print(f"Score       : {r.score} / 3")
        print(f"Plate Color : {r.plate_color}")
        print(f"Alert Level : {r.alert_level}")
        print(f"Summary     : {r.summary}")

output_path = "data/outputs/result.jpg"
cv2.imwrite(output_path, annotated)
print(f"\nAnnotated image saved to: {output_path}")