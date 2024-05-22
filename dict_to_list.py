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

def write_json_file(file_path):

    data = read_json_file(file_path)

    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        print("Data written to", file_path)
    except IOError:
        print("Error writing to file.")
