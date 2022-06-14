import json
from sklearn.neighbors import KDTree
from os import walk
from typing import Dict, List


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


def make_trigrams_cnt(identifier: str) -> Dict[str, int]:
    res = {}
    for trigram in map(lambda tup: ''.join(tup), zip(*[identifier[i:] for i in range(3)])):
        if trigram not in res:
            res[trigram] = 1
        else:
            res[trigram] += 1
    return res

# TODO generics
def merge_cnts(left_dict, right_dict) -> dict:
    result = right_dict.copy()
    for k, v in left_dict:
        if k in result:
            result[k] += v
        else:
            result[k] = v


def proces_data(merged_dict): # The typing of this thing is too cumbersome
    # An object that maps the author into two sparse matrices that are
    # a) the amounts of lines by-language
    # b) the trigram counts
    final_author_dict: Dict[str, Dict[str, int]] = {}

    for _, repo_dict in merged_dict.items():
        for author_name, author_dict in repo_dict.items():
            languages_subdict = {}
            trigrams_subdict = {}

            for _, commit_dict in repo_dict.items():
                # Nonempty lines amount
                patch_set = commit_dict["patch_set"]
                for k, v in patch_set.items():
                    patch_set[k] = len(list(filter(lambda x: x != "", v.split("\n"))))

                patch_languages = commit_dict["patch_languages"]
                # Slap those together
                for k, v in patch_languages:
                    if v not in languages_subdict:
                        languages_subdict[v] = patch_set[k]
                    else:
                        languages_subdict[v] += patch_set[k]

                patch_ids = commit_dict["patch_ids"]

                for k, v in patch_ids.items():
                    for identifier in v:
                        trigrams_subdict = trigrams_subdict.union(make_trigrams_cnt(identifier))

                # make sparse matrices
                # load 2 matrices (by-language linenums + trigrams)
                # into two kdtrees
                # combine scores for queries

            if author_name not in final_author_dict:
                final_author_dict[author_dict] = {
                    "languages": languages_subdict,
                    "trigrams": trigrams_subdict
                }
            else:
                final_author_dict[author_dict] = {
                    "languages": merge_cnts(final_author_dict["languages"], languages_subdict),
                    "trigrams": merge_cnts(final_author_dict["trigrams"], trigrams_subdict)
                }
# WRITE A QUERY REPL

