import cv2
import numpy as np

def show_grayscale_image(image_array, index=0, scale=256):

    if len(image_array.shape) == 4:
        image = image_array[index, :, :, 0]
    else:
        image = image_array[index]

    display_img = (image * 255).astype(np.uint8)
    enlarged_img = cv2.resize(display_img, (scale, scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('Grayscale Image', enlarged_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()