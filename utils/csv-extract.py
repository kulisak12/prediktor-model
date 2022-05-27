#!/usr/bin/env python3
from typing import List
import pandas as pd
import re
import sys


def tokenize(line: str) -> List[str]:
    """Split a line into tokens.

    Keep only words.
    """
    return re.findall(r"\w+", line)

    # splits = re.split(r"[\b\s]", line)
    # splits = map(lambda x: x.strip(), splits)
    # splits = filter(lambda x: x, splits)
    # return list(splits)


def main():
    if len(sys.argv) <= 1:
        print("No CSV file specified.")
        sys.exit(1)
    filename = sys.argv[1]

    csv = pd.read_csv(filename)
    with open("output.txt", "w") as f:
        for line in csv["text"]:
            tokens = tokenize(line)
            f.write(" ".join(tokens) + "\n")


if __name__ == '__main__':
    main()
