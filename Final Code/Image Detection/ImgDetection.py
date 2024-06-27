from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.layers import Dense

import numpy as np
import cv2

import json

def analyze_images(json_file, img_path, vgg_weights_path):

    # json_file - Path to JSON file containing the OCR output
    # img_path - Path to the input image
    # vgg_weights_path - Path to the VGG weights file

    img_dict = {} #  Function returns dict - > image : [[xmin, ymin, xmax, ymax], label] 

    # Detection of figures

    with open(json_file) as f:
        data = json.load(f)

    dimensions = data["pages"][0]["dimensions"]

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (dimensions[1], dimensions[0]))

    noise_removal = cv2.bilateralFilter(img, 9, 75, 75)
    binary_image = cv2.adaptiveThreshold(noise_removal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    for block in data["pages"][0]["blocks"]:
        for word in block["lines"]:
            for word_data in word["words"]:

                xmin = int(word_data["geometry"][0][0] * img.shape[1])
                ymin = int(word_data["geometry"][0][1] * img.shape[0])
                xmax = int(word_data["geometry"][1][0] * img.shape[1])
                ymax = int(word_data["geometry"][1][1] * img.shape[0])

                binary_image[ymin:ymax, xmin:xmax] = 255

    binary_image = cv2.bitwise_not(binary_image)

    closing_kernel = np.ones((10, 10), np.uint8)
    closing_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, closing_kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image, connectivity=8)

    img_area = img.shape[0] * img.shape[1]

    import os
    if not os.path.exists('crops'):
        os.makedirs('crops')

    for label in range(1, num_labels): 
        
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = width * height
        
        if img_area*0.2> area > img_area*0.001 and  not (width / height > 4 or height / width > 4):
            print(f"Component {label} - Area: {area}, Width: {width}, Height: {height}")

            mask = np.uint8(labels == label)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Padding
            x -= 10
            y -= 10
            w += 20
            h += 20
            component_crop = img[y:y+h, x:x+w]

            cv2.imwrite(f'crops/component_{label}.png', component_crop)

            xmin = x / img.shape[1]
            ymin = y / img.shape[0]
            xmax = (x + w) / img.shape[1]
            ymax = (y + h) / img.shape[0]

            img_dict[f'crops/component_{label}.png'] = [[xmin, ymin, xmax, ymax], ""]


    # Classify cropped images
    
    class_to_label = {'BarCodes': 0, 'Graphs': 1, 'LogosAndStamps': 2, 'Photographs': 3, 'QRcodes': 4, 'Signatures': 5}
    label_to_class = {v: k for k, v in class_to_label.items()}

    num_classes = len(class_to_label)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.load_weights(vgg_weights_path)

    img_folder = r"crops/"

    for file_name in os.listdir(img_folder):

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):

            file_path = os.path.join(img_folder, file_name)
            img = image.load_img(file_path, target_size=(224, 224))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            predictions = model.predict(x)
            predicted_labels = np.argmax(predictions, axis=1)

            labels = [label_to_class[label] for label in predicted_labels]

            img_dict[file_path][1] = labels[0]

    
    return img_dict
