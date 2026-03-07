try:
    from tqdm.std import tqdm
    _orig_close = tqdm.close
    def _safe_close(self):
        if hasattr(self, "disable") is False:
            self.disable = True
        return _orig_close(self)
    tqdm.close = _safe_close
except Exception:
    pass

import asyncio
from tqdm.asyncio import tqdm_asyncio

async def main():
    async for i in tqdm_asyncio(range(10)):
        pass
        
    # deliberately make an invalid object to test GC
    obj = tqdm_asyncio.__new__(tqdm_asyncio)
    # let it get garbage collected
    del obj

if __name__ == "__main__":
    asyncio.run(main())
