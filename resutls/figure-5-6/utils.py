import os

def create_non_exist_path(path_to_check):
    if not os.path.exists(path_to_check):
        os.makedirs(path_to_check, exist_ok=True)
        print(f"Created result directory: {path_to_check}")
    else:
        print(f"Result directory already exists: {path_to_check}")
