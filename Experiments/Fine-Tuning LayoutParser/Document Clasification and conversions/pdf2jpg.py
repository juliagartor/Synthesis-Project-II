from pdf2image import convert_from_path
import sys

def convert_pdf_to_jpg(pdf_path, output_folder, poppler_path):
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    # Save each image as JPG
    for i, image in enumerate(images):
        image_path = f'{output_folder}/page_{i + 1}.jpg'
        image.save(image_path, 'JPEG')
        print(f'Saved {image_path}')

# Poseu el svostres paths si volewu probar i sobretot el poppler que es una cosa que s'ha de descarregar a part.
pdf_path = "C:/Users/alexx/OneDrive/Escriptori/uni/3rd year/synthesis/Sample documents/Sample documents/Bank account statements/Estado de cuenta completo BBVA.pdf"
output_folder = "C:/Users/alexx/OneDrive/Escriptori/uni/3rd year/synthesis/jpg_conversion"
poppler_path = "C:/Program Files/poppler-24.02.0/Library/bin" # Si us dona problemes el popper digueume que te el seu lio.

convert_pdf_to_jpg(pdf_path, output_folder, poppler_path)