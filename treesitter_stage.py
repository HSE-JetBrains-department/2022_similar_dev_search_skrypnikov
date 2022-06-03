from tree_sitter import Language, Parser

from os import makedirs, system
from sys import stderr
import os.path as path

from typing import List, Dict

TREE_SITTER_SO = 'tree_sitter_collected.so'

def load_treesitter(languages: List[str]) -> Dict[str, Language]:
    if not path.exists('build'):
        makedirs('build')

    monikers = list(map(lambda s: f'tree-sitter-{s.lower()}', languages))

    # How threadsafe is cloning?

    for ts_moniker in monikers:
        if not path.exists(f'build/{ts_moniker}'):
            exit_code = system(f'git clone https://github.com/tree-sitter/{ts_moniker} build/{ts_moniker}')
            if exit_code:
                stderr.write(f'ERROR: Couldn\'t clone the git repo for tree sitter language {ts_moniker}\n')
                stderr.write(f'Aborting...\n')
                exit(1)

    tree_sitter_so_path = f'build/{TREE_SITTER_SO}'

    Language.build_library(
        tree_sitter_so_path,
        list(map(lambda m: f'build/{m}', monikers))
    )

    return {lang: Language(tree_sitter_so_path, lang) for lang in languages}