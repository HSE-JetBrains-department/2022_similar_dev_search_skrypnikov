from itertools import islice, chain
from os import walk, path, makedirs, system
from sys import stderr
from typing import List, Iterable, TypeVar, Dict

from dulwich.objects import S_ISGITLINK, Blob
from enry import get_languages
from tree_sitter import Language

T = TypeVar('T')


def group(iterable: Iterable[T], n: int) -> Iterable[Iterable[T]]:
    it = iter(iterable)
    while True:
        chunk_it = islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield chain((first_el,), chunk_it)


# Helper functions
# s means safe.

def get_content(repo, mode, hexsha):
    if hexsha is None:
        return Blob.from_string(b"")
    elif S_ISGITLINK(mode):
        return Blob.from_string(b"Subproject commit " + hexsha + b"\n")
    else:
        return repo.object_store[hexsha]


def get_lines(content: str) -> List[str]: return content.splitlines() if content else []


# Python is a readable multi-purpose language
def unwrap_bytes_gen_to_str(gen: Iterable[bytes], enc: str = "utf-8") -> str:
    return ''.join([it.decode(enc) if not isinstance(it, str) else it for it in gen])


def cleanup_diff(diff: List[str]) -> str: return '\n'.join(map(lambda s: s[1:] if s[0] == '+' else s, diff))


def path_base_norm(some_path: str) -> str: return path.normpath(path.basename(some_path))


def run_enry_by_file(base_path: str, repo_path: str) -> Dict[str, List[str]]:
    enry_dict_full_paths = {}

    for root, dirs, files in walk(repo_path):
        for f in files:
            to_enrify = path.normpath(path.join(base_path, root, f))
            with open(to_enrify, 'rb') as file_bytes:
                enrified = get_languages(to_enrify, file_bytes.read())
                if enrified:
                    enry_dict_full_paths[to_enrify] = enrified

    return enry_dict_full_paths


TREE_SITTER_SO = 'tree_sitter_collected.so'


def load_tree_sitter(languages: List[str], build_dir_path: str) -> Dict[str, Language]:
    if not path.exists(build_dir_path):
        makedirs(build_dir_path)

    monikers = list(map(lambda s: f'tree-sitter-{s.lower()}', languages))

    # How threadsafe is cloning?

    for ts_moniker in monikers:
        moniker_repo_path = path.join(build_dir_path, ts_moniker)
        if not path.exists(moniker_repo_path):
            exit_code = system(f"git clone https://github.com/tree-sitter/{ts_moniker} {moniker_repo_path}")
            if exit_code:
                stderr.write(f"ERROR: Couldn't clone the git repo for tree sitter language {ts_moniker}\n")
                stderr.write(f"Aborting...\n")
                exit(1)

    tree_sitter_so_path = path.join(build_dir_path, TREE_SITTER_SO)

    Language.build_library(
        tree_sitter_so_path,
        list(map(lambda m: path.join(build_dir_path, m), monikers))
    )

    return {lang: Language(tree_sitter_so_path, lang) for lang in languages}

