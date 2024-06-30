import numpy as np
from doctr.models import ocr_predictor
import PIL.Image as Image
import cv2
import json
import matplotlib.pyplot as plt

def draw_all(image, ocr_results):
    for page in ocr_results['pages']:
        # Draw blocks
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    # Get the coordinates for words
                    x_min, y_min = int(word['geometry'][0][0] * image.shape[1]), int(word['geometry'][0][1] * image.shape[0])
                    x_max, y_max = int(word['geometry'][1][0] * image.shape[1]), int(word['geometry'][1][1] * image.shape[0])
                    
                    # Draw the bounding box for words
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    
                    # Put the text
                    #cv2.putText(image, word['value'], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw artifacts if they exist
        if 'artefacts' in page:
            for artifact in page['artefacts']:
                # Get the coordinates for artifacts
                
                x_min, y_min = int(artifact['geometry'][0][0] * image.shape[1]), int(artifact['geometry'][0][1] * image.shape[0])
                x_max, y_max = int(artifact['geometry'][1][0] * image.shape[1]), int(artifact['geometry'][1][1] * image.shape[0])
                
                # Draw the bounding box for artifacts
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
    return image


def img_run_doctr_pipeline(predictor ,image_path:str, parameter_bin_thr:float, parameter_box_thr:float):
    # Modify the binarization threshold and the box threshold
    predictor.det_predictor.model.postprocessor.bin_thresh = parameter_bin_thr
    predictor.det_predictor.model.postprocessor.box_thresh = parameter_box_thr

    image_path = image_path
    image = np.array(Image.open(image_path))
    out = predictor([image])

    json_output = out.export()

    # Draw the OCR results on the image
    output_image = draw_all(image, json_output)

    # Save the result
    output_image_path = f'{image_path}_bboxesdrawn.jpg'
    cv2.imwrite(output_image_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    # Save the OCR results in a JSON file
    with open(f'{image_path}.json', 'w') as f:
        json.dump(json_output, f)
