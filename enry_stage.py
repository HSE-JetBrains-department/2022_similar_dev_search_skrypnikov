from enry import get_languages
import os

def run_enry_by_file(CWD, base_dir_path):
    enry_dict_full_paths = {}

    for root, dirs, files in os.walk(base_dir_path):
        for f in files:
            to_enrify = os.path.normpath(os.path.join(CWD, root, f))
            with open(to_enrify, 'rb') as file_bytes:
                enrified = get_languages(to_enrify, file_bytes.read())
                if enrified:
                    enry_dict_full_paths[to_enrify] = enrified
    
    return enry_dict_full_paths
