import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os

img = cv2.imread("bird.png")
img_pil = Image.open("bird.png")
    
if not os.path.exists("processed_images"):
    os.makedirs("processed_images")

blur = cv2.GaussianBlur(img, (21, 21), 0)
cv2.imwrite(os.path.join("processed_images", 'blur.png'), blur)
    
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[..., 0] = (hsv[..., 0] + 0) % 180  # Rotate hue by 90 degrees
hue_rotated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite(os.path.join("processed_images", 'hue_rotation.png'), hue_rotated)
    
noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
grain = cv2.add(img, noise)
cv2.imwrite(os.path.join("processed_images", 'grain.png'), grain)
    
enhancer = ImageEnhance.Brightness(img_pil)
bright_image = enhancer.enhance(3)  # Increase brightness
bright_image.save(os.path.join("processed_images", 'brightness.png'))
    
inverted = cv2.bitwise_not(img)
cv2.imwrite(os.path.join("processed_images", 'inversion.png'), inverted)
    
noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
noisy_image = cv2.add(img, noise)
cv2.imwrite(os.path.join("processed_images", 'noise.png'), noisy_image)