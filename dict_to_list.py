'''Dicts to List jsons done after description generation'''


import json

def read_json_file(file_path):
    """
    Read a JSON file and return its contents as a list of dictionaries.
    """
    try:
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print("Error decoding JSON on line:", e)
        return data
    except FileNotFoundError:
        print("File not found.")
        return []

def write_json_file(data, file_path):
    """
    Write data to a JSON file.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        print("Data written to", file_path)
    except IOError:
        print("Error writing to file.")

def main():
    # Replace 'your_json_file.json' with the path to your JSON file
    json_file_path = 'journal_no_2156.json'

    # Read the JSON file
    json_data = read_json_file(json_file_path)

    # Write the JSON data to a new file named output.json
    write_json_file(json_data, json_file_path)

if __name__ == "__main__":
    main()
