
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.layers import Dense

import numpy as np
import cv2

import json

def analyze_image(model, json_file, img_path, page_number):

    # json_file - Path to JSON file containing the OCR output
    # img_path - Path to the input image
    # vgg_weights_path - Path to the VGG weights file
    img_dict = {} #  Function returns dict - > image : [[xmin, ymin, xmax, ymax], label] 
    
    working_dir = "/".join(img_path.split("/")[:-1])

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

                padding = 10
                ymin -= padding
                if ymin < 0:
                    ymin = 0

                ymax += padding
                if ymax >= img.shape[0]:
                    ymax = img.shape[0] - 1

                xmin -= padding
                if xmin < 0:
                    xmin = 0

                xmax += padding
                if xmax >= img.shape[1]:
                    xmax = img.shape[1] - 1

                binary_image[ymin:ymax, xmin:xmax] = 255

    binary_image = cv2.bitwise_not(binary_image)

    closing_kernel = np.ones((10, 10), np.uint8)
    closing_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, closing_kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image, connectivity=8)

    img_area = img.shape[0] * img.shape[1]

    import os
    if not os.path.exists(working_dir+'/crops'):
        os.makedirs(working_dir+'/crops')

    llista_comp = []

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
            if x < 0:
                x = 0            
            
            y -= 10
            if y < 0:
                y = 0

            w += 20
            h += 20

            y2 = y + h
            if y2 > img.shape[0]:
                y2 = img.shape[0] - 1
            
            x2 = x + w
            if x2 > img.shape[1]:
                x2 = img.shape[1] - 1
            
            component_crop = img[y:y+h, x:x+w]

            cv2.imwrite(f'{working_dir}/crops/component_{label}.png', component_crop)

            xmin = x / img.shape[1]
            ymin = y / img.shape[0]
            xmax = (x + w) / img.shape[1]
            ymax = (y + h) / img.shape[0]

            img_dict[f'{working_dir}/crops/component_{label}.png'] = [[xmin, ymin, xmax, ymax], ""]
            llista_comp.append(f'component_{label}.png')

            
            #img_dict[f'component_{label}.png'] = [[xmin, ymin, xmax, ymax], ""]


    # Classify cropped images
    
    class_to_label ={   0: 'LogosAndStamps',
                        1: 'Signatures',
                        2: 'QRcodes',
                        3: 'BarCodes',
                        4: 'Graphs',
                        5: 'Photographs',
                        6: 'BadCrop'
                    }
    label_to_class = {v: k for k, v in class_to_label.items()}


    img_folder = working_dir + "/crops"

    for file_name in llista_comp:

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):

            file_path = os.path.join(img_folder, file_name)
            img = image.load_img(file_path, target_size=(224, 224))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            predictions = model.predict(x)
            predicted_labels = np.argmax(predictions, axis=1)
            
            #check if predicted label is a random crop
            if predicted_labels[0] == 6:
                os.remove(file_path)
                del img_dict[file_path]
            else:
                img_dict[file_path][1] = class_to_label[predicted_labels[0]]


        with open(f"{working_dir}/document_images_{page_number}.json", "w") as outfile: 
            json.dump(img_dict, outfile)
        
    return img_dict


###Patch CLF model
import keras
model = keras.models.load_model('src/model.keras')
#print(model.summary())
import json 
f = open('src/working_paths.json') 
data = json.load(f) 

images_dict = analyze_image(model,data["ocr_path"],data["img_path"],page_number=data["page_number"])
   