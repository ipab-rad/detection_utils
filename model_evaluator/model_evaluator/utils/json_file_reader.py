import json
def read_json(file_path:str):
    with open(file_path) as f:
        result = json.load(f)

    return result

def write_json(file_path:str, data: dict | list):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)