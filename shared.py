import sys

from itertools import islice, chain
from dulwich.objects import S_ISGITLINK, Blob, Commit

def group(iterable, n):
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

def s_get_content(repo, mode, hexsha):
    if hexsha is None:
        return Blob.from_string(b"")
    elif S_ISGITLINK(mode):
        return Blob.from_string(b"Subproject commit " + hexsha + b"\n")
    else:
        return repo.object_store[hexsha]

def s_get_lines(content):
    if not content:
        return []
    else:
        return content.splitlines()

# Python is a readable multi-purpose language
def unwrap_bytes_gen_to_str(gen, enc='utf-8'):
    return ''.join([it.decode(enc) if not isinstance(it, str) else it for it in gen])


def progress(sha_str, curr_prog, total_prog):
    sys.stdout.write('\r')
    sys.stdout.write(f'parsing... {curr_prog}/{total_prog} at {sha_str}')
    sys.stdout.flush()

def flush_progress_bar():
    sys.stdout.write('\r')

def finish_progress():
    sys.stdout.write('\n')
