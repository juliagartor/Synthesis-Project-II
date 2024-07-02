import json

def adjust_y_coordinates(input_file, output_file):
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Iterate through each page and each block to adjust the y-coordinates of words within each line
    for page in data['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                if line['words']:
                    # Get the y-coordinate of the first word in the line
                    first_word_y = line['words'][0]['geometry'][0][1]
                    # Adjust the y-coordinate of all words in the line to match the first word's y-coordinate
                    for word in line['words']:
                        word['geometry'][0][1] = first_word_y
                        word['geometry'][1][1] = first_word_y
    
    # Save the modified JSON data back to a file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# Example usage
input_file_path = r"C:\Users\alexx\OneDrive\Escriptori\uni\3rd year\Efficient-Recognition-of-Official-Documents\Tests Alex Sanchez\output_roldan.json"
output_file_path = r"C:\Users\alexx\OneDrive\Escriptori\uni\3rd year\Efficient-Recognition-of-Official-Documents\Tests Alex Sanchez\same_y1.json"
adjust_y_coordinates(input_file_path, output_file_path)
