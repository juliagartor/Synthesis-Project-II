import PIL.Image as Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
import json
import os

def pageout2pdf(json_pth:str, crops_json_pth:str, out_pth:str, size_font:int):
    import json 
    f = open(json_pth) 
    data = json.load(f) 

    # Create a PDF document
    pdf_output_path = out_pth
    c = canvas.Canvas(pdf_output_path, pagesize=letter)

    # Set the font and size
    c.setFont("Helvetica", size_font)

    # Function to get color based on confidence
    def get_color(confidence):
        if confidence > 0.8:
            return Color(0, 0, 0)  # Green for high confidence
        elif confidence > 0.5:
            return Color(0, 0, 0)  # Yellow for medium confidence
        else:
            return Color(1, 0, 0)  # Red for low confidence

    # Extract text and positions from OCR output
    for block in data["pages"][0]["blocks"]:
        for line in block["lines"]:
            # Get y position of the first word in the line
            first_word_y1 = (1 - line["words"][0]["geometry"][0][1]) * 900

            for word in line["words"]:
                # Get word coordinates and text
                text = word["value"]
                (x1, y1), (x2, y2) = word["geometry"]  # top-left and bottom-right corners
                # Convert normalized coordinates to absolute pixel values (for a letter-size page)
                x_abs = x1 * 630  # letter width in points
                y_abs = first_word_y1  # Align y position based on the first word in the line
                # Set the color based on confidence
                color = get_color(word["confidence"])
                c.setFillColor(color)
                # Draw the text at the specified position
                c.drawString(x_abs, y_abs, text)

    
    page_dims_x = data["pages"][0]["dimensions"][1]
    page_dims_y = data["pages"][0]["dimensions"][0]

    ##Now add crops if there are
    if os.path.isfile(crops_json_pth):
        f = open(crops_json_pth) 
        crops_dict = json.load(f)
        # Draw rectangles for the crops
        for crop_info in crops_dict.values():
            coordinates, label = crop_info
            x0, y0, x1, y1 = coordinates
            # Convert normalized coordinates to absolute pixel values (for a letter-size page)
            x0_abs = x0 * page_dims_x
            y0_abs = (1 - y0) * page_dims_y
            x1_abs = x1 * page_dims_x
            y1_abs = (1 - y1) * page_dims_y

            # Draw the rectangle
            c.setStrokeColor(Color(R0, 0, 0, alpha=0.5))  # Black border with 50% transparency
            c.setFillColor(Color(0, 0, 1, alpha=0.2))  # Blue fill with 20% transparency
            c.rect(x0_abs, y1_abs, x1_abs - x0_abs, y0_abs - y1_abs)  # Note the y-coordinates are flipped
            
            # Calculate the center of the rectangle for the label
            center_x = (x0_abs + x1_abs) / 2
            center_y = (y1_abs + y0_abs) / 2

            # Draw the label text at the center of the rectangle
            c.drawCentredString(center_x, center_y, label)
    # Save the PDF
    c.save()

