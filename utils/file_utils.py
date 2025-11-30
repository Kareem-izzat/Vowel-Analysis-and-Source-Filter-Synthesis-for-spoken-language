import os

def scan_data_folder(base_path):
    """
    Returns list of vowel folders inside data directory.
    """
    folders = []
    for f in os.listdir(base_path):
        full = os.path.join(base_path, f)
        if os.path.isdir(full):
            folders.append(f)
    return folders
