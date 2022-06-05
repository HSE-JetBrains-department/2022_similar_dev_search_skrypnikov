import json
import multiprocessing
from argparse import ArgumentParser
from os import makedirs, system, getcwd, path
from shutil import rmtree
from typing import Dict, List, Tuple
from dulwich.objects import Commit, ShaFile
from dulwich.repo import Repo
from tree_sitter import Parser
from unidiff import PatchSet
from repoparse.internal.utils import group, run_enry_by_file, load_tree_sitter
from repoparse.internal.classes import ParsedRepo, RepoParseContext

global_languages = ("python",)


def patch_set_insertions_to_dict(repo_path: str, patch_set: PatchSet) -> Dict[str, List[str]]:
    dict_obj = {}
    for patched_file in patch_set:
        changes_list = []
        dict_obj[path.join(repo_path, patched_file.path)] = changes_list
        for hunk in patched_file:
            changes_list.extend(map(str, hunk.target_lines()))

    return dict_obj


def _parse_repo_commits_chunked(repo_path: str, enry_dict: Dict[str, List[str]], chunk_size: int):
    print('Starting parallel map...')
    repo = Repo(repo_path)
    with multiprocessing.Pool(5) as p:
        return p.map(
            _parse_commit_chunk,
            [(i, repo_path, enry_dict, list(chunk)) for i, chunk in enumerate(group(repo.object_store, chunk_size))]
        )


def _parse_commit_chunk(chunk_tuple: Tuple[int, str, Dict[str, List[str]], List[ShaFile]]) -> ParsedRepo:
    chunk_n, repo_path, enry_dict, chunk = chunk_tuple
    print(f'Chunk {chunk_n} started...')

    # Have to bootstrap i/o objects for each chunk locally,
    # since they can't be copied via pickling from the parent process.
    repo = Repo(repo_path)
    parsed_repo = ParsedRepo({})
    chunk_local_parser = Parser()
    chunk_local_lang_objs = load_tree_sitter(global_languages.copy())

    ctx = RepoParseContext(
        enry_dict,
        chunk_local_lang_objs,
        chunk_local_parser,
        chunk_number=chunk_n,
        curr_item_number=0,
        is_quiet=False,
        parsed_repo=parsed_repo
    )

    for sha in chunk:
        ctx.curr_item_number += 1
        if isinstance(repo[sha], Commit) and len(repo[sha].parents) == 1:
            ctx.parse_commit_with_single_parent(repo, repo[sha])

    return ctx.parsed_repo


def _parse_repo_commits_not_chunked(repo_path: str, enry_dict: Dict[str, List[str]], quiet: bool) -> ParsedRepo:
    repo = Repo(repo_path)
    parsed_repo = ParsedRepo({})
    parser = Parser()
    lang_objs = load_tree_sitter(global_languages.copy())

    ctx = RepoParseContext(
        enry_dict,
        lang_objs,
        parser,
        chunk_number=0,  # The single chunk has id 0
        curr_item_number=0,
        is_quiet=quiet,
        parsed_repo=parsed_repo
    )

    for sha in repo.object_store:
        ctx.curr_item_number += 1
        if isinstance(repo[sha], Commit) and len(repo[sha].parents) == 1:
            ctx.parse_commit_with_single_parent(repo, repo[sha])

    return ctx.parsed_repo


def parse_repo_commits(repo_path: str, enry_dict: dict, quiet: bool, chunked: bool = False, chunk_size: int = 10000):
    if not chunked:
        return _parse_repo_commits_not_chunked(repo_path, enry_dict, quiet)
    else:
        return _parse_repo_commits_chunked(repo_path, enry_dict, chunk_size)


def clone_and_detect_path(repo_path: str, force_reclone: bool):
    if not path.exists('../build'):
        makedirs('../build')

    dirname = path.basename(path.normpath(repo_path))  # repo_path.rsplit("/", 1)[1]
    if dirname.endswith('.git'):
        dirname = dirname.rsplit(".", 1)[0]

    cloned_path = f'./build/{dirname}'

    if path.exists(cloned_path) and force_reclone:
        rmtree(cloned_path)

    if path.exists(cloned_path):
        print(f'Reusing repo already cloned from `{cloned_path}`...')
    else:
        print(f'Cloning `{repo_path}`...')
        system(f'git clone {repo_path} {cloned_path}')

    return cloned_path


def parse_repo(base_path: str, repo_path: str, do_force_reclone: bool, is_quiet: bool, is_chunked: bool,
               chunk_size: int, out_dir_name="out", build_dir_name="build") -> ParsedRepo:
    absolute_out_path = path.join(base_path, out_dir_name)
    absolute_build_path = path.join(base_path, build_dir_name)

    if not path.exists(absolute_out_path):
        makedirs(absolute_out_path)
    if not path.exists(absolute_build_path):
        makedirs(absolute_build_path)

    is_remote_repo = repo_path.startswith('http') or repo_path.startswith('git@')

    if not path.exists(repo_path) and not is_remote_repo:
        raise ValueError('Unknown repo...')

    if is_remote_repo:
        repo_path = clone_and_detect_path(repo_path, do_force_reclone)

    repo_path = path.join(base_path, repo_path)

    # TODO discover more repos and put them into parsing with separate instances of this script
    # TODO refactor into neat stages?

    print('Doing the enry part...')
    enry_on_latest_revision = run_enry_by_file(base_path, repo_path)

    print('Parsing the repo...')
    return parse_repo_commits(repo_path, enry_on_latest_revision, is_quiet, chunked=is_chunked, chunk_size=chunk_size)


if __name__ == '__main__':
    CWD = getcwd()

    arg_parser = ArgumentParser(description='Mine the developer data from a repo.')
    arg_parser.add_argument('--repo', '-r', nargs=1, required=True)
    arg_parser.add_argument('--force-reclone', action='store_true')
    arg_parser.add_argument('--quiet', '-q', action='store_true')
    arg_parser.add_argument('--chunked', '-ch', action='store_true')
    arg_parser.add_argument('--chunk-size', '-sz', nargs=1)
    args = arg_parser.parse_args()

    r = parse_repo(CWD, args.repo[0], args.force_reclone, args.quiet, args.chunked,
                   int(args.chunk_size if args.chunk_size else 0))

    with open(f"out/repoparse_{path.basename(path.normpath(args.repo[0]))}.json", "w") as json_out:
        json.dump(r, json_out, default=lambda x: x.to_dict())
