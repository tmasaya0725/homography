import cv2
import numpy as np

image_path = "/home/takanashi.masaya/workspace/eval/real_image/source/2.png" 
mask_path = "/home/takanashi.masaya/workspace/eval/real_image/obj/2.png"
target_path = "/home/takanashi.masaya/workspace/eval/real_image/target/2.png"

image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

H = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, image.shape[0]],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

warped_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
warped_mask = cv2.warpPerspective(mask, H, (image.shape[1], image.shape[0]))

concat_image = cv2.bitwise_and(warped_image, warped_image, mask=warped_mask)
concat_image = cv2.bitwise_or(concat_image, image)

cv2.imwrite("./output_image.png", concat_image)
cv2.imwrite("./mask.png", mask)
cv2.imwrite("./input_image.png", image)
cv2.imwrite("./target_image.png", cv2.imread(target_path))