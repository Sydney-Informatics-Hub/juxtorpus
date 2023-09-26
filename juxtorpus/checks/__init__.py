from abc import ABCMeta, abstractmethod
from typing import Iterable, Callable, Generator
import pandas as pd
from pathlib import Path


class FlaggedPath(object):
    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason

    def __repr__(self):
        return f"Path: {self.path}\t [REASON: {self.reason}]"


class Check(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, path):
        raise NotImplementedError()


class FileCheckers(object):
    def __init__(self, checks: list[Callable]):
        self._checks = checks
        self._flagged = dict()
        self._passed = list()

    def run(self, paths: Iterable[Path]):
        if isinstance(paths, Generator):
            paths = list(paths)
        paths = self._to_paths(paths)
        flagged = dict()
        path: Path
        for path in paths:
            for check in self.checks:
                passed = check(path)
                if not passed:
                    if isinstance(check, Check):
                        reason = check.reason()
                    else:
                        reason = check.__name__

                    key: str = str(path)
                    reasons = flagged.get(key, list())
                    reasons.append(reason)
                    flagged[key] = reasons
        self._passed = [str(p) for p in paths if str(p) not in flagged.keys()]
        self._flagged = flagged
        return flagged

    @property
    def checks(self):
        return self._checks

    def flagged(self):
        return self._flagged

    def passed(self):
        return self._passed

    def summary(self):
        num_flagged = len(self._flagged.keys())
        num_passed = len(self._passed)
        return pd.Series([num_flagged, num_passed, num_flagged + num_passed],
                         index=['Flagged', 'Passed', 'Total'],
                         name='File Check Summary',
                         dtype='UInt16')  # up to ~65k files

    def _to_paths(self, paths) -> list[Path]:
        for i, p in enumerate(paths):
            if not isinstance(p, Path):
                paths[i] = Path(p)
        return paths


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing your files to run checkers.")
    args = parser.parse_args()

    dir_ = Path(args.dir)


    def check_file_lang(file: Path):
        print(f"file_lang: checking file - {file}")
        return True


    checks = [
        check_file_lang,
    ]
    file_checks = FileCheckers(checks)

    paths = dir_.glob("*")

    flagged = file_checks.run(paths)

    print("+++ Results +++")
    print("FLAGGED ", file_checks.flagged())
    print("PASSED ", file_checks.passed())
    print(file_checks.summary())
