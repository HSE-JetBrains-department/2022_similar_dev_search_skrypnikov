from dataclasses import dataclass
from os import path
import sys
from typing import Dict, List, Union, cast

from dulwich.patch import gen_diff_header, is_binary, patch_filename, unified_diff
from internal.utils import cleanup_diff, get_content, get_lines, unwrap_bytes_gen_to_str
from tree_sitter import Language, Parser
from unidiff import PatchSet, UnidiffParseError


# Helper dtos to better transfer data between functions..
@dataclass(init=True)
class CommitParseResult:
    patch_set: Dict[str, str]
    patch_languages: Dict[str, str]
    patch_ids: Dict[str, List[str]]

    @property
    def is_ok(self):
        return self.patch_set and self.patch_languages and self.patch_ids


@dataclass
class ParsedCommitInfo:
    # Only single parent is allowed
    parent_sha: str
    patch_set: Dict[str, str]
    patch_languages: Dict[str, str]
    patch_ids: Dict[str, List[str]]

    def __init__(self, parent_sha: str, commit_parse_result: CommitParseResult):
        self.parent_sha = parent_sha
        self.patch_set = commit_parse_result.patch_set
        self.patch_languages = commit_parse_result.patch_languages
        self.patch_ids = commit_parse_result.patch_ids

    def to_dict(self):
        return {
            'parent_sha': self.parent_sha,
            'patch_set': self.patch_set,
            'patch_languages': self.patch_languages,
            'patch_ids': self.patch_ids
        }


@dataclass
class ParsedCommits:
    commit_data_by_sha: Dict[str, ParsedCommitInfo]

    def __getitem__(self, item: str) -> Union[ParsedCommitInfo, None]:
        return self.commit_data_by_sha.get(item)

    def __setitem__(self, key: str, value: ParsedCommitInfo):
        self.commit_data_by_sha[key] = value

    def to_dict(self):
        """Shallow-dumps the object into a dictionary. Not the __dict__ way."""
        return {k: v for k, v in self.commit_data_by_sha.items()}


@dataclass
class ParsedRepo:
    commits_by_author: Dict[str, ParsedCommits]

    def __getitem__(self, item: str):
        return self.commits_by_author.get(item)

    def __setitem__(self, key: str, value: ParsedCommits):
        self.commits_by_author[key] = value

    def to_dict(self):
        return {k: v for k, v in self.commits_by_author.items()}


def patch_set_insertions_to_dict(repo_path: str, patch_set: PatchSet) -> Dict[str, List[str]]:
    dict_obj = {}
    for patched_file in patch_set:
        changes_list = []
        dict_obj[path.join(repo_path, patched_file.path)] = changes_list
        for hunk in patched_file:
            changes_list.extend(map(str, hunk.target_lines()))

    return dict_obj


@dataclass(init=True)
class RepoParseContext:
    """Context for parsing a (part) of a repo"""
    enry_on_latest_revision: Dict[str, List[str]]
    local_ts_lang_objs: Dict[str, Language]
    local_ts_parser: Parser

    chunk_number: Union[int, None]
    curr_item_number: int
    is_quiet: bool

    parsed_repo: ParsedRepo

    out_dir_path: str

    def parse_commit_with_single_parent(self, repo, commit):
        """Parses a commit"""
        parent = repo[commit.parents[0]]

        diff_str = ''

        # Had to reverse-engineer (partially borrow) it from dulwich sources
        # ¯\_(ツ)_/¯
        changes = repo.object_store.tree_changes(commit.tree, parent.tree)
        for (old_path, new_path), (old_mode, new_mode), (old_sha, new_sha) in changes:
            p_old_path = cast(bytes, patch_filename(old_path, b'a'))
            p_new_path = cast(bytes, patch_filename(new_path, b'b'))

            diff_str += unwrap_bytes_gen_to_str(
                gen_diff_header((old_path, new_path), (old_mode, new_mode), (old_sha, new_sha))
            )

            old_content = get_content(repo, old_mode, old_sha)
            new_content = get_content(repo, new_mode, new_sha)

            if is_binary(old_content.data) or is_binary(new_content.data):
                diff_str += f"Binary files {p_old_path.decode()} and {p_new_path.decode()} differ\n"
            else:
                try:
                    diff_str += unwrap_bytes_gen_to_str(
                        unified_diff(
                            get_lines(old_content.as_raw_string().decode()),
                            get_lines(new_content.as_raw_string().decode()),
                            p_old_path.decode(),
                            p_new_path.decode()
                        )
                    )
                except UnicodeDecodeError:
                    continue

        parsed_commits = self.parsed_repo[commit.author.decode()]
        if parsed_commits is None:
            parsed_commits = self.parsed_repo[commit.author.decode()] = ParsedCommits({})

        commit_parse_result = self._extract_commit_info(repo.path, diff_str)
        if not commit_parse_result.is_ok:
            return

        # Some repos are truly humongous
        # and CPython can't effectively reclaim memory while processing them
        # So, here comes the del!
        del diff_str

        parsed_commits[commit.sha().hexdigest()] = ParsedCommitInfo(
            parent.sha().hexdigest(),
            commit_parse_result
        )

    def _extract_commit_info(self, repo_path: str, diff_str: str) -> CommitParseResult:
        """Extracts languages, lines and identifiers from a diff and returns them as an object."""
        try:
            patch_languages = {}
            patch_set_dict = {}
            patch_ids_dict = {}

            # Splice out the parsed files
            patch_files = patch_set_insertions_to_dict(repo_path, PatchSet(diff_str))

            for f in patch_files:
                file_path = path.join(repo_path, f)
                if file_path in self.enry_on_latest_revision:
                    languages = self.enry_on_latest_revision[file_path]
                    if (language := languages[0].lower()) in self.local_ts_lang_objs:
                        patch_text = patch_files[file_path]

                        self.local_ts_parser.set_language(self.local_ts_lang_objs[language])
                        ts_query = self.local_ts_lang_objs[language].query("((identifier) @id)")
                        patch_string = cleanup_diff(patch_text)

                        src_bytes = bytes(patch_string, "utf8")
                        captures = ts_query.captures(self.local_ts_parser.parse(src_bytes).root_node)
                        if captures:
                            patch_set_dict[file_path] = patch_string
                            patch_languages[file_path] = language
                            patch_ids_dict[file_path] = list(map(
                                lambda capture: src_bytes[capture[0].start_byte:capture[0].end_byte].decode('utf8'),
                                captures
                            ))

            return CommitParseResult(patch_languages, patch_set_dict, patch_ids_dict)

        except UnidiffParseError as e:
            repo_name = path.basename(path.normpath(repo_path))
            err_log_name = f"error{repo_name}:{self.chunk_number}:{self.curr_item_number}.log"
            parser_position = f"{self.chunk_number}:{self.curr_item_number}"

            with open(f"{self.out_dir_path}/error{repo_name}:{parser_position}.log", 'w') as f:
                f.write(diff_str)
            sys.stderr.write(f"\n!!! ERROR !!!: Invalid (?) unidiff at item {parser_position}\n")
            sys.stderr.write(f"\nText:\n\t{e}\nSee {err_log_name} for details...\n")

            return CommitParseResult({}, {}, {})  # not is_ok

