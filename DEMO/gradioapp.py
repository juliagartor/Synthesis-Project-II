import gradio as gr
import base64
import os
import shutil
import fitz  # import the bindings
import json

#### OCR MODEL
from src.doctr_script import  img_run_doctr_pipeline
from doctr.models import ocr_predictor
predictor= ocr_predictor(pretrained=True)
spanish_vocab = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~áéíóúñ¿¡'

# Modify the vocabulary
predictor.reco_predictor.model.cfg['vocab'] = spanish_vocab


###PDF GENERATION
from src.doctr2pdf import pageout2pdf
from pypdf import PdfMerger

###WORD GENERATION
#from spire.pdf.common import *
#from spire.pdf import *

def pdf2imgs(pth:str, savedir:str, filename:str):
    doc = fitz.open(pth)  # open document
    for i, page in enumerate(doc):  # iterate through the pages
        pix = page.get_pixmap()  # render page to an image
        pix.save(f"{savedir}/{filename}_{i}.png")  # store image as a PNG

def greet(bin_thr:float, box_thr:float,font_size,file):
    if file is not None:
        with open(file, "rb") as file:
            pdf_data = file.read()
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # Create an HTML element to embed the PDF (for visualization)
        pdf_html = f'<iframe width="100%" height="500px" src="data:application/pdf;base64,{pdf_base64}"></iframe>'
        
        ##If document is a pdf then save each page as image and put into folder otherwise put single image into folder
        
        filename = str(file).split("/")[-1][:-2]
        filename_no_ext = "".join(filename.split(".")[:-1])

        if os.path.isdir(filename_no_ext):
            shutil.rmtree(filename_no_ext)

        os.mkdir(filename_no_ext)
        if ".pdf" in filename:
            pdf2imgs(file,filename_no_ext,filename)
        else:
            shutil.move(str(file), filename_no_ext + "/" + filename) 


        ##Run the model and show outputs
        
        ##Get output images:
        image_dir = filename_no_ext
        images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg', 'gif'))]
        
        #final pdf single page files, useful for concatenation
        pdf_pages_pths = []
        for img in images:
            img_run_doctr_pipeline(predictor,img, bin_thr, box_thr)

            #save to file so that the other process on the other python env can acces the paths we want to ptocess
            paths = {
                "page_number":img.split(".")[-2].split("_")[-1],
                "ocr_path" : img+".json",
                "img_path" : img
            }
            with open("src/working_paths.json", "w") as outfile: 
                json.dump(paths, outfile)
            
            #run process on another python env
            os.system("sh src/run_img_detection.sh")

            #generate page pdf
            pageout2pdf(img+".json",filename_no_ext+"/document_images_"+paths["page_number"]+".json",img+".pdf",size_font=font_size)
            pdf_pages_pths.append(img+".pdf")
        

        #concatenate all pdf pages into a single file
        merger = PdfMerger()

        for pdf in pdf_pages_pths:
            merger.append(pdf)
        
        merger.write(f"{filename_no_ext}/output.pdf")
        merger.close()

        '''
        #Create editable word file from pdf
        pdf = PdfDocument()
        # Load a PDF file
        pdf.LoadFromFile(f"{filename_no_ext}/output.pdf")

        # Convert the PDF file to a Word DOCX file
        pdf.SaveToFile(f"{filename_no_ext}/output.docx", FileFormat.DOCX)
        # Close the PdfDocument object
        pdf.Close()
        '''

        bounding_boxes_imgs = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if "bboxesdrawn" in img]
        

        #output pdf visualization
        with open(f"{filename_no_ext}/output.pdf", "rb") as file:
            pdf_data = file.read()
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # Create an HTML element to embed the PDF (for visualization)
        out_pdf_html= f'<iframe width="100%" height="500px" src="data:application/pdf;base64,{pdf_base64}"></iframe>'
        
    
        return pdf_html, bounding_boxes_imgs, out_pdf_html
    else:
        return "No PDF uploaded.", []

demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.Slider(label="Binary threshold", minimum=0, maximum=1,value=0.5),
        gr.Slider(label="Box threshold", minimum=0, maximum=1,value=0.2),
        gr.Slider(label="Font size", minimum=5, maximum=20,value=8),
        gr.File(label="Upload PDF", type="filepath")
    ],
    outputs=[
        gr.HTML(label="Input PDF Viewer"),
        gr.Gallery(label="Recognized Bounding Boxes"),
        gr.HTML(label="Generated PDF Viewer")
    ],
    title="Document digitalization"
)

demo.launch()



