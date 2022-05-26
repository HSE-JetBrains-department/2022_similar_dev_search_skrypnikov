from dataclasses import dataclass
from typing import Dict, List

import sys

from dulwich.objects import Commit, Blob
from dulwich.patch import patch_filename, is_binary, gen_diff_header, unified_diff
from dulwich.repo import Repo
from os import getcwd, path

from os import makedirs, system
from shutil import rmtree

from unidiff import PatchSet, UnidiffParseError

from shared import flush_progress_bar, progress, finish_progress, group, s_get_content, s_get_lines, unwrap_bytes_gen_to_str

@dataclass(init=True)
class ParsedCommitInfo:

    # Only single parent is allowed
    parent_sha: str
    patch_set: List[str]


@dataclass
class ParsedCommits:

    commit_data_by_sha: Dict[str, ParsedCommitInfo]

    def __getitem__(self, item: str) -> ParsedCommitInfo:
        return self.commit_data_by_sha.get(item)

    def __setitem__(self, key: str, value: ParsedCommitInfo):
        self.commit_data_by_sha[key] = value


@dataclass
class ParsedRepo:

    commits_by_author: Dict[str, ParsedCommits]

    def __getitem__(self, item: str):
        return self.commits_by_author.get(item)

    def __setitem__(self, key: str, value: ParsedCommits):
        self.commits_by_author[key] = value


def patch_set_to_json(patchSet):
    dict_obj = {}
    for patched_file in patchSet:
        changes_list = []
        dict_obj[patched_file.path] = changes_list
        for hunk in patched_file:
            changes_list.extend(map(str, hunk.target_lines()))
    
    return dict_obj
    

def repo_part_to_json(obj):
    # Python can't pattern match the argument to a method parameter
    # by type if we need to pass a method into another function (?)
    if isinstance(obj, ParsedCommitInfo):
        return {
            'parent_sha': obj.parent_sha, 
            'patch_set': obj.patch_set
        }
    if isinstance(obj, ParsedCommits):
        return {k if isinstance(k, str) else k.decode(): v for k, v in obj.commit_data_by_sha.items()}
    if isinstance(obj, ParsedRepo):
        return {k if isinstance(k, str) else k.decode(): v for k, v in obj.commits_by_author.items()}
  
    return obj

def _map_chunk_to_repo_part(chunk):
    repo = Repo(repo_path)
    parsed_repo = ParsedRepo({})

    for sha in chunk:
        if isinstance(repo[sha], Commit):
            if not _try_do_parse_commit(repo, repo[sha], parsed_repo, quiet=True): continue
    
    return parsed_repo

def _parse_repo_commits_chunked(repo_path, quiet, chunk_size):
    return map(group(repo.object_store, chunk_size))

def _parse_repo_commits_unchunked(repo_path: str, enry_dict: dict, quiet: bool):
    # List all commits in object_store
    # If a commit has a single parent:
    #  Get a diff with the parent
    #  Parse the diff into the proper representation
    repo = Repo(repo_path)
    parsed_repo = ParsedRepo({})

    curr_item = 0

    for sha in repo.object_store:
        curr_item += 1
        if isinstance(repo[sha], Commit):
            if not _try_do_parse_commit(repo, repo[sha], parsed_repo, enry_dict, curr_item=curr_item, quiet=quiet): continue
    
    finish_progress()
    return parsed_repo

def _try_do_parse_commit(repo, commit, parsed_repo, enry_dict, *, quiet, curr_item):
    if len(commit.parents) != 1:
        # Do not parse merge commits, those contain barely any relevant info
        return False

    parent = repo[commit.parents[0]]

    diff_str = ''

    # Had to reverse-engineer (partially borrow) it from dulwich sources
    # ¯\_(ツ)_/¯
    changes = repo.object_store.tree_changes(commit.tree, parent.tree)
    for (old_path, new_path), (old_mode, new_mode), (old_sha, new_sha) in changes:
        p_old_path = patch_filename(old_path, b'a')
        p_new_path = patch_filename(new_path, b'b')
        
        diff_str += unwrap_bytes_gen_to_str(
            gen_diff_header((old_path, new_path), (old_mode, new_mode), (old_sha, new_sha))
        )

        old_content = s_get_content(repo, old_mode, old_sha)
        new_content = s_get_content(repo, new_mode, new_sha)

        if is_binary(old_content.data) or is_binary(new_content.data):
            diff_str += f"Binary files {p_old_path.decode()} and {p_new_path.decode()} differ\n"
        else:
            try:
                diff_str += unwrap_bytes_gen_to_str(
                    unified_diff(
                        s_get_lines(old_content),
                        s_get_lines(new_content),
                        p_old_path,
                        p_new_path
                    )
                )
            except UnicodeDecodeError:
                continue

    parsed_commits = parsed_repo[commit.author]
    if parsed_commits is None:
        parsed_commits = parsed_repo[commit.author] = ParsedCommits({})

    try:
        patch_languages_and_files = {}

        for f in PatchSet(diff_str):
            file_path = path.join(repo.path, f.path)
            if file_path in enry_dict:
                patch_languages_and_files[file_path] = enry_dict[file_path]

        if patch_languages_and_files:
            parsed_commits[commit.sha().hexdigest()] = ParsedCommitInfo(
                parent.sha().hexdigest(),
                patch_languages_and_files
            )
    except UnidiffParseError as e:
        with open(f'out/error{curr_item}.log', 'w') as f:
            f.write(diff_str)
        sys.stderr.write(f'\n!!! ERROR !!!: Invalid (?) unidiff at item {curr_item}, hex: {commit.sha().hexdigest()}\n')
        sys.stderr.write(f'\nText:\n\t{e}\nSee error{curr_item}.log for details...\n')

    if not quiet:
        progress(commit.sha().hexdigest(), curr_item, 'unk')
    
    return True

def parse_repo_commits(repo_path: str, enry_dict: dict, quiet: bool, chunked: bool = False, chunk_size: int = 10000):
    """Parse a repo.
    :repo_path: A path to the repo on the current os's FS.
    """
    if not chunked:
        return _parse_repo_commits_unchunked(repo_path, enry_dict, quiet)
    else:
        return _parse_repo_commits_chunked(repo_path, quiet, chunk_size)

def clone_and_detect_path(repo_path, force_reclone):

    if not path.exists('build'):
        makedirs('build')
    
    dirname = repo_path.rsplit("/", 1)[1]
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

if __name__ == '__main__':
    from argparse import ArgumentParser
    import json, pickle

    CWD = getcwd()

    parser = ArgumentParser(description='Mine the developer data from a repo.')
    parser.add_argument('--repo', '-r', nargs=1, required=True)
    parser.add_argument('--force-reclone', action='store_true')
    parser.add_argument('--clean', '-c', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    repo_path = args.repo[0]
    is_remote_repo = repo_path.startswith('http') or repo_path.startswith('git@')

    if not path.exists(repo_path) and not is_remote_repo:
        raise ValueError('Unknown repo...')

    from enry_stage import run_enry_by_file

    if is_remote_repo:
        repo_path = clone_and_detect_path(repo_path, args.force_reclone)

    repo_path = path.join(CWD, repo_path)

    # TODO extract the variable/function names with treesitter (or other parser software)
    # TODO discover more repos and put them into parsing with separate instances of this script 
    # TODO refactor into neat stages?

    if not path.exists('./out'):
        makedirs('out')

    print('Doing the enry part...')
    enry_dict = run_enry_by_file(CWD, repo_path)

    print('Parsing the repo...')
    r = parse_repo_commits(repo_path, enry_dict, args.quiet)

    with open(f'out/enry_{path.basename(path.normpath(repo_path))}.json', 'w') as json_out:
        json.dump(enry_dict, json_out, default=repo_part_to_json)

    with open(f'out/repoparse_{path.basename(path.normpath(repo_path))}.json', 'w') as json_out:
        json.dump(r, json_out, default=repo_part_to_json)

