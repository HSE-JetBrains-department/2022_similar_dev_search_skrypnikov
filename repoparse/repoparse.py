from argparse import ArgumentParser
import json
import multiprocessing
from os import makedirs, system, getcwd, path
from shutil import rmtree
import time
from typing import Dict, List, Tuple
from typing import Union

from dulwich.objects import Commit, ShaFile
from dulwich.repo import Repo
from github import Github, GithubException
from tree_sitter import Parser

from internal.classes import ParsedRepo, RepoParseContext
from internal.utils import group, run_enry_by_file, load_tree_sitter

GLOBAL_LANGUAGES = ["python", "c", "java", "javascript", "php"]


def _parse_repo_commits_chunked(repo_path: str, enry_dict: Dict[str, List[str]], chunk_size: int, out_path: str,
                                build_path: str, pool_processes: int):
    print('Starting parallel map...')
    repo = Repo(repo_path)
    with multiprocessing.Pool(pool_processes) as p:
        return p.map(
            _parse_commit_chunk,
            [(i, repo_path, enry_dict, list(chunk), out_path, build_path) for i, chunk in
             enumerate(group(repo.object_store, chunk_size))]
        )


def _parse_commit_chunk(chunk_tuple: Tuple[int, str, Dict[str, List[str]], List[ShaFile], str, str]) -> ParsedRepo:
    chunk_n, repo_path, enry_dict, chunk, out_path, build_path = chunk_tuple
    print(f"Chunk {chunk_n} started...")

    # Have to bootstrap i/o objects for each chunk locally,
    # since they can't be copied via pickling from the parent process.
    repo = Repo(repo_path)
    parsed_repo = ParsedRepo({})
    chunk_local_parser = Parser()
    chunk_local_lang_objs = load_tree_sitter(GLOBAL_LANGUAGES, build_path)

    ctx = RepoParseContext(
        enry_dict,
        chunk_local_lang_objs,
        chunk_local_parser,
        chunk_number=chunk_n,
        curr_item_number=0,
        is_quiet=False,
        parsed_repo=parsed_repo,
        out_dir_path=out_path
    )

    for sha in chunk:
        ctx.curr_item_number += 1
        if isinstance(repo[sha], Commit) and len(repo[sha].parents) == 1:
            ctx.parse_commit_with_single_parent(repo, repo[sha])

    print(f"Chunk {chunk_n} exiting from {repo_path}...")
    return ctx.parsed_repo


def _parse_repo_commits_not_chunked(repo_path: str, enry_dict: Dict[str, List[str]], quiet: bool, out_path: str,
                                    build_path: str) -> ParsedRepo:
    repo = Repo(repo_path)
    parsed_repo = ParsedRepo({})
    parser = Parser()
    lang_objs = load_tree_sitter(GLOBAL_LANGUAGES, build_path)

    ctx = RepoParseContext(
        enry_dict,
        lang_objs,
        parser,
        chunk_number=0,  # The single chunk has id 0
        curr_item_number=0,
        is_quiet=quiet,
        parsed_repo=parsed_repo,
        out_dir_path=out_path
    )

    for sha in repo.object_store:
        ctx.curr_item_number += 1
        if isinstance(repo[sha], Commit) and len(repo[sha].parents) == 1:
            ctx.parse_commit_with_single_parent(repo, repo[sha])

    return ctx.parsed_repo


def parse_repo_commits(repo_path: str, enry_dict: dict, quiet: bool, out_path: str, build_path: str,
                       chunked: bool = False, chunk_size: int = 10000, pool_processes=5):
    if not chunked:
        return _parse_repo_commits_not_chunked(repo_path, enry_dict, quiet, out_path, build_path)
    else:
        return _parse_repo_commits_chunked(repo_path, enry_dict, chunk_size, out_path, build_path, pool_processes)


def clone_and_detect_path(repo_path: str, force_reclone: bool, build_dir_path: str) -> str:
    if not path.exists(build_dir_path):
        makedirs(build_dir_path)

    dirname = path.basename(path.normpath(repo_path))  # repo_path.rsplit("/", 1)[1]
    if dirname.endswith(".git"):
        dirname = dirname.rsplit(".", 1)[0]

    cloned_path = path.join(build_dir_path, dirname)

    if path.exists(cloned_path) and force_reclone:
        rmtree(cloned_path)

    if path.exists(cloned_path):
        print(f"Reusing repo already cloned from `{cloned_path}`...")
    else:
        print(f"Cloning `{repo_path}`...")
        system(f"git clone {repo_path} {cloned_path}")

    return cloned_path


def get_github_repo_id(repo_path: str) -> Union[str, None]:
    parsing_str: str

    if repo_path.startswith("git@"):
        return repo_path.replace("git@github.com:", "").replace(".git", "")
    elif repo_path.startswith("https://"):
        return repo_path.replace("https://github.com/", "").replace(".git", "")

    return None


def calculate_stars(repo_path: str, git_key_path: str, parse_depth: int, max_stargazers: int,
                    max_stargazers_starred: int, next_stage_repos: int) -> List[str]:
    repos_for_next_parse = []
    github_repo_id = get_github_repo_id(repo_path)

    with open(git_key_path, "r") as f:
        github_token = f.readline()

    if parse_depth > 0:
        github = Github(github_token)
        print(f"Getting github repo id {github_repo_id}")
        github_repo = github.get_repo(github_repo_id)
        print(f"Got repo {github_repo_id}!")

        print("Calculating stars...")
        starred_dict: Dict[str, int] = {}
        finished = False
        while not finished:
            try:
                stargazers = github_repo.get_stargazers()
                for i, stargazer in enumerate(stargazers):
                    if i == max_stargazers:
                        break

                    stargazers_starred = stargazer.get_starred()
                    for j, stargazers_star in enumerate(stargazers_starred):
                        if j == max_stargazers_starred:
                            break

                        if stargazers_star.full_name in starred_dict:
                            starred_dict[stargazers_star.full_name] += 1
                        else:
                            starred_dict[stargazers_star.full_name] = 1

                repos_for_next_parse = list(
                    [f"git@github.com:{k}.git" for k, _ in sorted(starred_dict.items(), key=lambda x: x[1])]
                )
                repos_for_next_parse = repos_for_next_parse[:min(next_stage_repos, len(repos_for_next_parse))]
                finished = True
            except GithubException as e:
                print(f"Github is behaving: {type(e)}")
                reset_time = github.get_rate_limit().rate.reset
                print("Retrying after a sleep...")
                time.sleep(reset_time - time.time())

        print("Finished calculating stars")

    return repos_for_next_parse


def parse_repo(
        base_path: str, repo_path: str, do_reclone: bool, is_quiet: bool, is_chunked: bool, chunk_size: int,
        out_dir_name="out", build_dir_name="build", git_key_path="build/github_key",
        parse_depth=3, max_stargazers=100, max_stargazers_starred=10, next_stage_repos=2, pool_processes=5
) -> Tuple[Union[None, List[str]], ParsedRepo]:
    absolute_out_path = path.join(base_path, out_dir_name)
    absolute_build_path = path.join(base_path, build_dir_name)
    repos_for_next_parse: Union[None, List[str]] = None

    if not path.exists(absolute_out_path):
        makedirs(absolute_out_path)
    if not path.exists(absolute_build_path):
        makedirs(absolute_build_path)

    is_remote_repo = repo_path.startswith("http") or repo_path.startswith("git@")

    if not path.exists(repo_path) and not is_remote_repo:
        raise ValueError("Unknown repo...")

    if is_remote_repo:
        repos_for_next_parse = calculate_stars(repo_path, git_key_path, parse_depth, max_stargazers,
                                               max_stargazers_starred, next_stage_repos)
        repo_path = clone_and_detect_path(repo_path, do_reclone, absolute_build_path)

    repo_path = path.join(base_path, repo_path)

    # TODO refactor into neat stages?

    print("Doing the enry part...")
    enry_on_latest_revision = run_enry_by_file(base_path, repo_path)

    print("Parsing the repo...")
    result = parse_repo_commits(repo_path, enry_on_latest_revision, is_quiet, absolute_out_path, absolute_build_path,
                                chunked=is_chunked, chunk_size=chunk_size, pool_processes=pool_processes)
    if repos_for_next_parse:
        return repos_for_next_parse, result
    else:
        return None, result


class RepoProcessWorker(multiprocessing.Process):

    def __init__(self, base_path: str, repo_path: str, do_reclone: bool, is_quiet: bool, is_chunked: bool,
                 chunk_size: int, visited_dict: Dict[str, int], visited_lock: multiprocessing.Lock,
                 out_dir_name="out", build_dir_name="build", git_key_path="build/github_key",
                 parse_depth=3, max_stargazers=1000, max_stargazers_starred=10, next_stage_repos=4, pool_processes=5):
        super().__init__()
        self.base_path = base_path
        self.repo_path = repo_path
        self.do_reclone = do_reclone
        self.is_quiet = is_quiet
        self.is_chunked = is_chunked
        self.chunk_size = chunk_size
        self.out_dir_name = out_dir_name
        self.build_dir_name = build_dir_name
        self.git_key_path = git_key_path
        self.parse_depth = parse_depth  # Next parse step...
        self.max_stargazers = max_stargazers
        self.max_stargazers_starred = max_stargazers_starred
        self.next_stage_repos = next_stage_repos
        self.visited_dict = visited_dict
        self.visited_lock = visited_lock
        self.pool_processes = pool_processes

    def run(self):
        print(f"PARSE DEPTH: {self.parse_depth}")

        try:
            next_repos, r = parse_repo(
                self.base_path, self.repo_path, self.do_reclone, self.is_quiet, self.is_chunked, self.chunk_size,
                self.out_dir_name, self.build_dir_name, self.git_key_path, self.parse_depth, self.max_stargazers,
                self.max_stargazers_starred, self.next_stage_repos
            )
        except Exception as e:
            print(f"Closing {self.repo_path} due to having encountered an error in the parser...")
            print(e)
            self.close()
            return

        with open(f"{self.out_dir_name}/repoparse_{path.basename(path.normpath(self.repo_path))}.json",
                  "w") as json_out:
            json.dump(r, json_out, default=lambda x: x.to_dict() if not isinstance(x, list) else x)

        # Some repos are truly humongous
        # and CPython can't effectively reclaim memory while processing them
        # So, here comes the del!
        del r

        if next_repos:
            try:
                self.visited_lock.acquire()
                next_repos = list(filter(lambda x: x not in self.visited_dict, next_repos))
                for repo in next_repos:
                    self.visited_dict[repo] = 1
            finally:
                self.visited_lock.release()

        print(f"Wrote {self.repo_path}")
        wait_list = []
        try:
            if not next_repos:
                self.close()
                return

            for repo_path in next_repos:
                worker = RepoProcessWorker(
                    self.base_path, repo_path, self.do_reclone, self.is_quiet, self.is_chunked, self.chunk_size,
                    self.visited_dict, self.visited_lock,
                    self.out_dir_name, self.build_dir_name, self.git_key_path, self.parse_depth - 1,
                    self.max_stargazers,
                    self.max_stargazers_starred, self.next_stage_repos
                )
                worker.start()
                wait_list.append(worker)

            for worker in wait_list:
                worker.join()
        finally:
            for w in wait_list:
                w.close()

        print(f"Process {self.repo_path} runned parent!!!")

    def close(self):
        print(f"Closing {self.repo_path}!!!")
        super().close()


if __name__ == '__main__':
    CWD = getcwd()

    arg_parser = ArgumentParser(description="Mine the developer data from a repo.")
    arg_parser.add_argument("--repo", "-r", nargs=1, required=True)
    arg_parser.add_argument("--force-reclone", action="store_true")
    arg_parser.add_argument("--quiet", "-q", action="store_true")
    arg_parser.add_argument("--chunked", "-ch", action="store_true")
    arg_parser.add_argument("--chunk-size", "-sz", nargs=1)
    arg_parser.add_argument("--build-path", "-b", nargs=1)
    arg_parser.add_argument("--out-path", "-o", nargs=1)
    arg_parser.add_argument("--process-amount", "-p", nargs=1)
    args = arg_parser.parse_args()

    # Preload tree sitter so there would be no races
    # (this function clones the git repos)
    _ = load_tree_sitter(GLOBAL_LANGUAGES, build_dir_path=args.build_path[0])

    with multiprocessing.Manager() as manager:
        proc = RepoProcessWorker(
            CWD, args.repo[0], args.force_reclone, args.quiet, args.chunked,
            int(args.chunk_size[0] if args.chunk_size else 0),
            manager.dict(), manager.Lock(),
            out_dir_name=args.out_path[0],
            build_dir_name=args.build_path[0],
            pool_processes=args.process_amount[0]
        )

        proc.start()
        proc.join()
        proc.close()
        print(f"Ending process {proc.repo_path}...")

    # die around here
