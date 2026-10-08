from contextlib import contextmanager

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_state = {"sink": None}


class NullBar:
    def update(self, n=1):
        pass

    def set_postfix(self, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def log(msg):
    sink = _state["sink"]
    if sink is not None:
        sink.write(msg + "\n")
        sink.flush()
    elif tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


@contextmanager
def log_to(path):
    previous = _state["sink"]
    with open(path, "a") as handle:
        _state["sink"] = handle
        try:
            yield
        finally:
            _state["sink"] = previous


def bar(total, desc, unit="job", leave=False):
    if tqdm is None or _state["sink"] is not None:
        return NullBar()
    return tqdm(total=total, desc=desc, unit=unit, leave=leave, disable=None, dynamic_ncols=True)
