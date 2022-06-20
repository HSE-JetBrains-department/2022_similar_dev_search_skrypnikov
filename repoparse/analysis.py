import json

from os import walk
from typing import Dict, TypeVar, List, Tuple
from collections import Counter

import numpy as np
from sklearn.neighbors import KDTree
from scipy.sparse import coo_array, csr_array
from repoparse.repoparse import global_languages


T = TypeVar("T")


def read_data(base_path: str = "./out", data_file_prefix: str = "repoparse_") -> dict:
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
    identifier = identifier.lower()
    for trigram in map(lambda tup: ''.join(tup), zip(*[identifier[i:] for i in range(3)])):
        if trigram not in res:
            res[trigram] = 1
        else:
            res[trigram] += 1
    return res


def merge_counts(left_dict: Dict[str, int], right_dict: Dict[str, int]) -> Dict[str, int]:
    cntr = Counter()
    cntr.update(left_dict)
    cntr.update(right_dict)
    return cntr


def process_data(merged_dict: dict) -> Dict[str, Dict[str, Dict[str, int]]]: # The typing of this thing is too cumbersome
    # An object that maps the author into two sparse matrices that are
    # a) the amounts of lines by-language
    # b) the trigram counts
    final_author_dict: Dict[str, Dict[str, Dict[str, int]]] = {}

    for _, repo_dict in merged_dict.items():
        for author_name, author_dict in repo_dict.items():
            languages_subdict: Dict[str, int] = {}
            trigrams_subdict: Dict[str, int] = {}

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
                final_author_dict[author_name] = {
                    "languages": languages_subdict,
                    "trigrams": trigrams_subdict
                }
            else:
                final_author_dict[author_name] = {
                    "languages": merge_counts(final_author_dict[author_name]["languages"], languages_subdict),
                    "trigrams": merge_counts(final_author_dict[author_name]["trigrams"], trigrams_subdict)
                }
    return final_author_dict


LANGUAGES_INDEX = {}
for i, k in enumerate(global_languages):
    LANGUAGES_INDEX[k] = i

TRIGRAMS_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
TRIGRAMS_CHARS_LEN = len(TRIGRAMS_CHARS)
TRIGRAMS_CHARS_POS = {}
for i, ch in enumerate(TRIGRAMS_CHARS):
    TRIGRAMS_CHARS_POS[ch] = i

TOTAL_TRIGRAM_COUNT = TRIGRAMS_CHARS_LEN ** 3


def get_trigram_number(trigram: str) -> int:
    # trigram number = pos * 1 + pos * len(trigrams_chars) + pos * len(trigrams_chars) ** 2
    return TRIGRAMS_CHARS_POS[trigram[0]] \
           + TRIGRAMS_CHARS_POS[trigram[1]] * TRIGRAMS_CHARS_LEN \
           + TRIGRAMS_CHARS_POS[trigram[2]] * TRIGRAMS_CHARS_LEN * TRIGRAMS_CHARS_LEN


def get_trigram_by_number(number: int) -> str:
    trigram = []
    while number != 0:
        trigram.append(TRIGRAMS_CHARS[number % TRIGRAMS_CHARS_LEN])
        number //= TRIGRAMS_CHARS_LEN
    return "".join(reversed(trigram))


def make_vectors(final_author_dict: Dict[str, Dict[str, Dict[str, int]]], author_index: Dict[str, int]) -> Tuple[np.array, csr_array]:
    language_matrix = np.zeros((len(author_index), len(LANGUAGES_INDEX)))
    trigram_coo_array = coo_array(len(author_index), TOTAL_TRIGRAM_COUNT, dtype=int)
    for author_name, author_dict in final_author_dict.items():
        for lang, cnt in author_dict["languages"]:
            language_matrix[author_index[author_name], LANGUAGES_INDEX[lang]] += cnt
        for trigram, cnt in author_dict["trigrams"]:
            trigram_coo_array[author_index[author_name], get_trigram_number(trigram)] += cnt

    return language_matrix, csr_array(trigram_coo_array)


if __name__ == "__main__":
    raw_data = process_data(read_data())
    author_index = {}
    for i, author_name in enumerate(sorted(raw_data.keys())):
        print(author_name)
        author_index[author_name] = i

    lang_matrix, trigram_matrix = make_vectors(raw_data, author_index)

    lang_tree = KDTree(lang_matrix)
    trigram_tree = KDTree(trigram_matrix)

    print("Welcome to the analysis repl! Enter a developer name to get the stats or `quit` to quit!")
    while (inp := input(">> ")) != "quit":
        if inp not in author_index:
            print("Can't find the specified author! Try again!")
        index = author_index[inp]
        dist, closest = lang_tree.query(lang_matrix[index, :], 3)
        print(f"Closest authors by languages: {closest}, respective distances: {dist}")

        dist, closest = trigram_tree.query(trigram_matrix[index, :], 3)
        print(f"Closest authors by trigrams: {closest}, respective distances: {dist}")


# WRITE A QUERY REPL

