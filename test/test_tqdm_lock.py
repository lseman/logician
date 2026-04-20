import os

os.environ['TQDM_DISABLE'] = '1'

class DummyLock:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def acquire(self, *args, **kwargs): pass
    def release(self): pass

from tqdm.std import tqdm

tqdm._lock = DummyLock()
tqdm.get_lock = lambda: tqdm._lock

from tqdm.asyncio import tqdm_asyncio

try:
    with tqdm_asyncio() as pbar:
        print("Success!")
except Exception:
    import traceback
    traceback.print_exc()
