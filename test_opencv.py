import cv2
import numpy as np
import matplotlib.pyplot as plt

print(f"OpenCV version: {cv2.__version__}")

# Create a simple image
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(img, (25, 25), (75, 75), (0, 255, 0), -1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Green square - OpenCV test")
plt.show() 