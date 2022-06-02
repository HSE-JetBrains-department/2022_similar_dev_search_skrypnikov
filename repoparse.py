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

import multiprocessing

from treesitter_stage import load_treesitter
from tree_sitter import Language, Parser

@dataclass(init=True)
class ParsedCommitInfo:

    # Only single parent is allowed
    parent_sha: str
    patch_set: Dict[str, str]
    langs: Dict[str, str]
    ids: Dict[str, str]


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

def cleanup_diff(diff):
    return '\n'.join(map(lambda s: s[1:] if s[0] == '+' else s, diff))

def patch_set_to_json(repopath, patchSet):
    dict_obj = {}
    for patched_file in patchSet:
        changes_list = []
        dict_obj[path.join(repopath, patched_file.path)] = changes_list
        for hunk in patched_file:
            changes_list.extend(map(str, hunk.target_lines()))
    
    return dict_obj
    

def repo_part_to_json(obj):
    # Python can't pattern match the argument to a method parameter
    # by type if we need to pass a method into another function (?)
    if isinstance(obj, ParsedCommitInfo):
        return {
            'parent_sha': obj.parent_sha, 
            'patch_set': obj.patch_set,
            'langs': obj.langs,
            'ids': obj.ids
        }
    if isinstance(obj, ParsedCommits):
        return {k if isinstance(k, str) else k.decode(): v for k, v in obj.commit_data_by_sha.items()}
    if isinstance(obj, ParsedRepo):
        return {k if isinstance(k, str) else k.decode(): v for k, v in obj.commits_by_author.items()}
  
    return obj

def _parse_repo_commits_chunked(repo_path, enry_dict, quiet, chunk_size):
    print('Starting parallel map...')
    repo = Repo(repo_path)
    with multiprocessing.Pool(5) as p:
        return p.map(_parse_commit_chunk, [(i, repo_path, enry_dict, list(chunk)) for i, chunk in enumerate(group(repo.object_store, chunk_size))])

def _parse_commit_chunk(chunk_tuple): # chunk is an iterable of whatever the repo.object store yields as an iterable
    i, repo_path, enry_dict, chunk = chunk_tuple
    print(f'Chunk {i} started...')
    
    repo = Repo(repo_path)
    parsed_repo = ParsedRepo({})

    parser = Parser()
    LANG_LIST = ['python']
    lang_objs = load_treesitter(LANG_LIST)

    curr_item = 0
    for sha in chunk:
        curr_item += 1
        if isinstance(repo[sha], Commit):
            if not _try_do_parse_commit(repo, repo[sha], parsed_repo, enry_dict, lang_objs, parser, quiet=True, curr_item=curr_item, chunk_n=i): continue
    
    return parsed_repo


def _parse_repo_commits_unchunked(repo_path: str, enry_dict: Dict[str, str], lang_objs: Dict[str, Language], quiet: bool):
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

def _try_do_parse_commit(repo, commit, parsed_repo, enry_dict, lang_objs, parser, *, quiet, curr_item, chunk_n=0):
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
        patch_languages = {}
        patch_set_dict  = {}
        patch_ids_dict  = {}

        # Splice out the parsed files
        patch_files = patch_set_to_json(repo.path, PatchSet(diff_str))

        for f in patch_files:
            file_path = path.join(repo.path, f)
            if file_path in enry_dict:
                languages = enry_dict[file_path]
                if (ll := languages[0].lower()) in lang_objs:
                    patch_text = patch_files[file_path]

                    parser.set_language(lang_objs[ll])
                    treesitter_query = lang_objs[ll].query('((identifier) @id)')
                    patch_string = cleanup_diff(patch_text)

                    src_bytes = bytes(patch_string, 'utf8')
                    captures = treesitter_query.captures(parser.parse(src_bytes).root_node)
                    if captures:
                        patch_set_dict[file_path] = patch_string
                        patch_languages[file_path] = ll
                        patch_ids_dict[file_path] = list(map(lambda capture: src_bytes[capture[0].start_byte:capture[0].end_byte].decode('utf8'), captures))

        if patch_languages and patch_set_dict and patch_ids_dict:
            parsed_commits[commit.sha().hexdigest()] = ParsedCommitInfo(
                parent.sha().hexdigest(),
                patch_set_dict,
                patch_languages,
                patch_ids_dict
            )

    except UnidiffParseError as e:
        with open(f'out/error{chunk_n}:{curr_item}.log', 'w') as f:
            f.write(diff_str)
        sys.stderr.write(f'\n!!! ERROR !!!: Invalid (?) unidiff at item {chunk_n}:{curr_item}, hex: {commit.sha().hexdigest()}\n')
        sys.stderr.write(f'\nText:\n\t{e}\nSee error{chunk_n}:{curr_item}.log for details...\n')

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
        return _parse_repo_commits_chunked(repo_path, enry_dict, quiet, chunk_size)

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
    parser.add_argument('--chunked', '-ch', action='store_true')
    args = parser.parse_args()

    repo_path = args.repo[0]
    is_remote_repo = repo_path.startswith('http') or repo_path.startswith('git@')

    if not path.exists(repo_path) and not is_remote_repo:
        raise ValueError('Unknown repo...')

    from enry_stage import run_enry_by_file

    if is_remote_repo:
        repo_path = clone_and_detect_path(repo_path, args.force_reclone)

    repo_path = path.join(CWD, repo_path)

    # TODO discover more repos and put them into parsing with separate instances of this script 
    # TODO refactor into neat stages?

    if not path.exists('./out'):
        makedirs('out')

    print('Doing the enry part...')
    enry_dict = run_enry_by_file(CWD, repo_path)
    print('enry part DONE')

    print('Parsing the repo...')
    r = parse_repo_commits(repo_path, enry_dict, args.quiet, chunked=args.chunked, chunk_size=50000)

    with open(f'out/enry_{path.basename(path.normpath(repo_path))}.json', 'w') as json_out:
        json.dump(enry_dict, json_out, default=repo_part_to_json)

    with open(f'out/repoparse_{path.basename(path.normpath(repo_path))}.json', 'w') as json_out:
        json.dump(r, json_out, default=repo_part_to_json)

