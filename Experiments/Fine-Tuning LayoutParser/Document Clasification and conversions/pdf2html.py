import fitz  # PyMuPDF
from bs4 import BeautifulSoup as BS

def pdf_to_html(pdf_path, html_path):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Start an empty HTML document
    html = BS('<html><body></body></html>', 'html.parser')

    # Get the body tag
    body = html.body

    # Loop through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text("text")
        
        # Create a new paragraph for each page
        page_paragraph = html.new_tag("p")
        page_paragraph.string = text
        
        # Add the paragraph to the body
        body.append(page_paragraph)

    # Save the HTML to a file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(str(html))

# Path to the PDF file
pdf_path = r"C:\Users\alexx\OneDrive\Escriptori\uni\3rd year\Efficient-Recognition-of-Official-Documents\Tests Alex Sanchez\recognized_text2.pdf"

# Path where you want to save the HTML file
html_path = r"C:\Users\alexx\OneDrive\Escriptori\uni\3rd year\Efficient-Recognition-of-Official-Documents\Tests Alex Sanchez\output_document21.html"

# Convert PDF to HTML
pdf_to_html(pdf_path, html_path)
