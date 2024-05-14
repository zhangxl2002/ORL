import os

def count_lines(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    total_lines += len(lines)
    return total_lines
import json

def count_code_lines(notebook_file):
    with open(notebook_file, 'r') as f:
        notebook = json.load(f)
        code_lines = 0
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                code_lines += len(cell['source'])
        return code_lines

if __name__ == "__main__":
    project_directory = "."
    python_lines = count_lines(project_directory)
    notebook_file = 'data_analysis.ipynb'
    jupyter_lines = count_code_lines(notebook_file)
    print("Total lines in the project:", python_lines + jupyter_lines)
