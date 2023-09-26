import codecs
from pathlib import Path
from chardet.universaldetector import UniversalDetector

from juxtorpus.checks import Check

class EncodingCheck(Check):
    """
    This Encoding Check depends on well known chardet.

    There seems to be better libraries on Github.
    TODO: https://github.com/Ousret/charset_normalizer
    """

    REASON: str = "File encoding inferred to be {} with confidence {}. Expected {}."

    expected_but_got_still_fine = {
        'UTF-8': {'ASCII'}
    }

    def __init__(self, expected: str, min_rows_to_check: int):
        try:
            codecs.lookup(expected)
        except LookupError:
            raise LookupError(f"Encoding {expected} is not supported.")
        self._expected = expected.upper()
        self._min_rows_to_check = min_rows_to_check

        self._detector = UniversalDetector()

        self._current_inferred = ''
        self._current_inferred_conf = -1

    @property
    def expected(self):
        return self._expected

    def __call__(self, path: Path) -> bool:
        detector = self._detector
        with open(path, 'rb') as h:
            for i, line in enumerate(h):
                detector.feed(line)
                if detector.done and i > self._min_rows_to_check:
                    break
        detector.close()
        self._current_inferred = detector.result.get('encoding')
        self._current_inferred_conf = detector.result.get('confidence')
        if self.expected.upper() == self._current_inferred.upper():
            return True
        else:
            return self._current_inferred.upper() in self.expected_but_got_still_fine.get(self.expected.upper())

    def reason(self):
        return self.REASON.format(self._current_inferred, self._current_inferred_conf, self.expected)
