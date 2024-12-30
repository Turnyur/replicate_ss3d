import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm


training = True
if training:
    active_path = "train_data"
else:
    active_path = "val_data"

base_path = "/home/hpc/iwi9/iwi9117h/dev"

dataset = "dataset/bird"

category_path_rgb = f"{base_path}/{dataset}/{active_path}/rgb_images"
category_path_masks = f"{base_path}/{dataset}/{active_path}/mask_images"
output_path = f"{base_path}/{dataset}"

csv_file_path = f"{output_path}/{active_path}.csv"



mask_images = []
for i, image in enumerate(os.listdir(category_path_masks)):
    mask_images.append(image)

rgb_images = []
for i, image in enumerate(os.listdir(category_path_rgb)):
    rgb_images.append(image)

mask_images.sort()
rgb_images.sort()

tqdm_counter = len(mask_images)
progress_bar = tqdm(range(tqdm_counter), desc="Rendering...", total=tqdm_counter)



data = []
for i in range(len(mask_images)):
    mask = cv2.imread(os.path.join(category_path_masks, mask_images[i]), cv2.IMREAD_GRAYSCALE)
    #mSource_Gray = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)

    binary_mask = mask / 255.0

    _, mThreshold = cv2.threshold(binary_mask, 0.1, 1.0, cv2.THRESH_BINARY)

    mThreshold = (mThreshold * 255).astype(np.uint8)
    Points = cv2.findNonZero(mThreshold)

    Min_Rect = cv2.boundingRect(Points)

    x, y, w, h = Min_Rect
    size = max(w, h)
    x_new = x + (w - size) // 2
    y_new = y + (h - size) // 2
    Min_Rect = (x_new, y_new, size, size)


    mSource_Bgr = cv2.imread(os.path.join(category_path_masks, mask_images[i]), cv2.IMREAD_COLOR)
    top_left = (Min_Rect[0], Min_Rect[1])
    bottom_right = (Min_Rect[0] + Min_Rect[2], Min_Rect[1] + Min_Rect[3])
    color = (0, 255, 0)
    cv2.rectangle(mSource_Bgr, top_left, bottom_right, color, 2)
    image = Image.fromarray(mSource_Bgr)
    #plt.imshow(image)
    #cv2.imwrite(os.path.join(output_path, "image_mask_{}.png".format(int(i))), mSource_Bgr)

    class_name = "bird_class"
    rgb_image_path = os.path.join(category_path_rgb, rgb_images[i])
    mask_image_path = os.path.join(category_path_masks, mask_images[i])

    Bounding_BOX_X_min = top_left[0]
    Bounding_BOX_Y_min = top_left[1]
    Bounding_BOX_X_max = bottom_right[0]
    Bounding_BOX_Y_max = bottom_right[1]

    data.append([
        class_name, 
        rgb_image_path, 
        mask_image_path, 
        Bounding_BOX_X_min, 
        Bounding_BOX_Y_min, 
        Bounding_BOX_X_max, 
        Bounding_BOX_Y_max])
    
    progress_bar.update(1)   
    progress_bar.set_description(f'Fiting bounding boxes...') 

csv_file_path

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    # writer.writerow([
    #     "class_name", 
    #     "rgb_image_path", 
    #     "mask_image_path", 
    #     "Bounding_BOX_X_min",
    #     "Bounding_BOX_Y_min", 
    #     "Bounding_BOX_X_max",
    #     "Bounding_BOX_Y_max"])
    
    writer.writerows(data)



