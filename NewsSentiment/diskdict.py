from pathlib import Path
import pickle
from typing import Union

from NewsSentiment.fxlogger import get_logger

logger = get_logger()


class DiskDict(dict):
    def __init__(self, sync_path: Union[Path, str]):
        if type(sync_path) == str:
            sync_path = Path(sync_path)
        self.path = sync_path

        if self.path.exists():
            with open(self.path, "rb") as file:
                tmp_dct = pickle.load(file)
                super().update(tmp_dct)
                logger.info(
                    "loaded DiskDict with %s items from %s", len(tmp_dct), self.path
                )

    def sync_to_disk(self):
        with open(self.path, "wb") as file:
            tmp_dct = super().copy()
            pickle.dump(tmp_dct, file)
            # print(f"saved DiskDict with {len(tmp_dct)} items to {self.path}")
