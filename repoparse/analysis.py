from collections import Counter, defaultdict
import json
from os import path, walk
from typing import Dict, Tuple, TypeVar

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.neighbors import KDTree as skKdTree

from .repoparse import GLOBAL_LANGUAGES

T = TypeVar("T")


def read_data(base_path: str = "./out", data_file_prefix: str = "repoparse_") -> dict:
    """Reads raw data from files"""
    merged_dict = {}
    for root, _, files in walk(base_path):

        for f in files:
            if f.startswith(data_file_prefix):
                with open(path.join(root, f), "r") as f_reader:
                    repo_name = f[len(data_file_prefix):].strip(".json")
                    file_str = f_reader.read()
                    if len(file_str) == 0:
                        continue
                    loaded_list = json.loads(file_str)
                    processed_author_dict = {}
                    for author_dict in loaded_list:
                        for author, commit_dict in author_dict.items():
                            if not commit_dict:
                                continue
                            # this is a mishap, please don't hit me too hard
                            processed_commit_dict = {}
                            for _, contents in commit_dict.items():
                                patch_set = contents["patch_languages"]
                                patch_languages = contents["patch_set"]
                                patch_ids = contents["patch_ids"]

                                processed_commit_dict["patch_set"] = patch_set
                                processed_commit_dict["patch_languages"] = patch_languages
                                processed_commit_dict["patch_ids"] = patch_ids

                            if processed_commit_dict:
                                processed_author_dict[author] = processed_commit_dict

                    if processed_author_dict:
                        merged_dict[repo_name] = processed_author_dict
                        break

    return merged_dict


def tri_iter(string):
    idx = 0
    yield string[idx:idx+3]
    idx = idx + 3


def make_trigrams_cnt(identifier: str) -> Dict[str, int]:
    res = defaultdict(int)
    identifier = identifier.lower()
    for trigram in tri_iter(identifier):
        res[trigram] += 1
    return res


def merge_counts(left_dict: Dict[str, int], right_dict: Dict[str, int]) -> Dict[str, int]:
    cntr = Counter()
    cntr.update(left_dict)
    cntr.update(right_dict)
    return cntr


def process_data(merged_dict: dict) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Preprocesses data into a final structure.
     The typing of this thing is too cumbersome
     Returns an object that maps the author into two dictionaries that are
     a) the amounts of lines by-language
     b) the trigram counts
    """
    final_author_dict: Dict[str, Dict[str, Dict[str, int]]] = {}
    for _, repo_dict in merged_dict.items():
        print(f"\rMerged dict item {_}", end="")
        for author_name, _ in repo_dict.items():
            languages_subdict: Dict[str, int] = {}
            trigrams_subdict: Dict[str, int] = {}

            for _, commit_dict in repo_dict.items():
                # Nonempty lines amount
                patch_set = commit_dict["patch_set"]
                for k, v in patch_set.items():
                    if isinstance(v, str):
                        patch_set[k] = len(list(filter(lambda x: x != "", v.split("\n"))))

                patch_languages = commit_dict["patch_languages"]
                # Slap those together
                for k, v in patch_languages.items():
                    if v not in languages_subdict:
                        languages_subdict[v] = patch_set[k]
                    else:
                        languages_subdict[v] += patch_set[k]

                patch_ids = commit_dict["patch_ids"]

                for k, v in patch_ids.items():
                    for identifier in v:
                        trigrams_subdict = merge_counts(trigrams_subdict, make_trigrams_cnt(identifier))

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
for i, k in enumerate(GLOBAL_LANGUAGES):
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


def make_vectors(final_author_dict: Dict[str, Dict[str, Dict[str, int]]], author_index: Dict[str, int]) -> Tuple[np.ndarray, csr_matrix]:
    """Packs data into matrices"""
    language_matrix = np.zeros((len(author_index), len(LANGUAGES_INDEX)))
    trigram_dok_array = dok_matrix((len(author_index), TOTAL_TRIGRAM_COUNT), dtype=int)
    for author_name, author_dict in final_author_dict.items():
        for lang, cnt in author_dict["languages"].items():
            language_matrix[author_index[author_name], LANGUAGES_INDEX[lang]] += cnt
        for trigram, cnt in author_dict["trigrams"].items():
            trigram_dok_array[author_index[author_name], get_trigram_number(trigram)] += cnt

    return language_matrix, csr_matrix(trigram_dok_array)


if __name__ == "__main__":
    print("reading data...")
    data_from_files = read_data()
    print("processing data...")
    raw_data = process_data(data_from_files)
    author_index = {}
    for i, author_name in enumerate(sorted(raw_data.keys())):
        author_index[author_name] = i

    lang_matrix, trigram_matrix = make_vectors(raw_data, author_index)

    print(author_index)

    lang_tree = skKdTree(lang_matrix)
    trigram_tree = skKdTree(trigram_matrix.toarray())

    print("Welcome to the analysis repl! Enter a developer name to get the stats or `quit` to quit!")
    while (inp := input(">> ")) != "quit":
        if inp not in author_index:
            print("Can't find the specified author! Try again!")
            continue
        index = author_index[inp]
        dist, closest = lang_tree.query(lang_matrix[index, :].reshape(1, -1), min(len(author_index), 3))
        print(f"Closest authors by languages: {closest}, respective distances: {dist}")

        dist, closest = trigram_tree.query(trigram_matrix[index, :].reshape(1, -1).toarray(), min(len(author_index), 3))
        print(f"Closest authors by trigrams: {closest}, respective distances: {dist}")
