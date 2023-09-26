from pathlib import Path

from juxtorpus.checks import Check


class FileSizeCheck(Check):
    REASON: str = "Exceeded the maximum size of {} bytes."

    def __init__(self, max_bytes: int):
        self.max_size = max_bytes

    def __call__(self, path: Path) -> bool:
        # print(f"File: {path} Size: {path.stat().st_size} bytes")
        return path.stat().st_size < self.max_size

    def reason(self):
        return self.REASON.format(self.max_size)
