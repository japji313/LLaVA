import json
import os
''' For second approch to seapate each feature , not currently using '''
def extract_text_from_json_files(json_files):
    extracted_data = []

    for json_file_path in json_files:
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)

        for item in json_data:
            filename = item["filename"]
            text = item["outputs"]

            keywords1 = ["Shape of the Image", "Texture of the Image", "Text of the Image", "Colors of the Image"]
            keywords2 = ["Shape:", "Texture:", "Text:", "Colors:"]

            def extract_text(text, keywords):
                split_text = text.split("\n\n")
                formatted_text = {}
                current_category = None
                for part in split_text:
                    for keyword in keywords:
                        if keyword in part:
                            current_category = keyword
                            formatted_text[current_category] = []
                            break
                    if current_category:
                        if current_category == "Shape of the Image" and "Shape of the Image" not in part:
                            formatted_text[current_category].extend(part.split())
                        else:
                            formatted_text[current_category].append(part)
                return formatted_text

            formatted_text1 = extract_text(text, keywords1)
            formatted_text2 = extract_text(text, keywords2)

            # If output from keywords1 is not generated, use the formatted text from keywords2
            if not formatted_text1:
                formatted_text = formatted_text2
            else:
                formatted_text = formatted_text1

            # Append the extracted data to the list
            extracted_data.append({"filename": filename, "extracted_text": formatted_text})

    # Write the extracted data to a new JSON file
    with open("merged_file.json", 'w') as outfile:
        json.dump(extracted_data, outfile, indent=4)

# Example usage
json_files = ["journal_no_2151.json", "journal_no_2153.json", "journal_no_2154.json"]  # Add more file paths as needed
extract_text_from_json_files(json_files)
