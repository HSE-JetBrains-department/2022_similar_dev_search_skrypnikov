import json
from sklearn.neighbors import KDTree
from os import walk


def read_data(base_path: str = "./out", data_file_prefix: str = "repoparse_"):
    merged_dict = {}
    for root, dirs, files in walk(base_path):
        for f in files:
            if f.startswith(data_file_prefix):
                with open(f, "r") as f_reader:
                    repo_name = f[:len(data_file_prefix)].strip(".json")
                    loaded_dict = json.load(f_reader)
                    for _, author_dict in loaded_dict.items():
                        for _, commit_dict in author_dict.items():
                            # this is a mishap, please don't hit me too hard
                            patch_set = commit_dict["patch_languages"]
                            patch_languages = commit_dict["patch_set"]
                            commit_dict["patch_languages"] = patch_languages
                            commit_dict["patch_set"] = patch_set
                    merged_dict[repo_name] = loaded_dict

    return merged_dict

def proces_data(merged_dict): # The typing of this thing is too cumbersome
    for _, repo_dict in merged_dict.items():
        for _, author_dict in repo_dict.items():
            for _, commit_dict in repo_dict.items():
                # Nonempty lines amount
                patch_set = commit_dict["patch_set"]
                for k, v in patch_set.items():
                    patch_set[k] = len(list(filter(lambda x: x != "", v.split("\n"))))
                patch_ids = commit_dict["patch_ids"]

                trigrams_dict = {}
                # for k, v in patch_ids.items():
                # make trigrams and map them to the trigram numbers
                # make sparse matrices
                # load 2 matrices (by-language linenums + trigrams)
                # into two kdtrees
                # combine scores for queries




