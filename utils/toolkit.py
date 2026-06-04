from datetime import datetime
from enum import Enum
from bisect import bisect_left


import numpy as np


class Counter:
    index: int = 0

    def add(self, step: int = 1) -> int:
        self.index += step
        return self.index


class Converter:
    @staticmethod
    def hour2sec(hours: float) -> float:
        return hours * 60 * 60

    @staticmethod
    def sec2hour(seconds: float) -> float:
        return seconds / 60 / 60

    @staticmethod
    def enum2dict(enum: Enum) -> dict:
        return {entry.name: entry.value for entry in enum}

def log_txt(path: str, tag_file: str = "unknown process", parameter_dict: dict = None):
    parameter_string = ''
    if parameter_dict is not None:
        i = 0
        for key, value in parameter_dict:
            parameter_string = parameter_string + f"{f' | ' if i == 0 else ', '}{key}: {value}"
            i += 1
    message = input("type in message to log")
    with open(path, 'a') as file:
        log_string = f"---\n" \
                     f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')} | {tag_file}\n" \
                     f"{parameter_string}\n" \
                     f"{message}"
        file.write(log_string)

def flatten_list_of_list(l):
    return [item for sublist in l for item in sublist]


# count elements in a dictionary with arbitrary number of nested levels
def count_entries_in_dict(my_dict, c=0):
    for my_key in my_dict:
        if isinstance(my_dict[my_key], dict):
            c = count_entries_in_dict(my_dict[my_key], c)
        elif isinstance(my_dict[my_key], np.ndarray) or isinstance(my_dict[my_key], list):
            c += len(my_dict[my_key])
        else:
            c += 1
    return c

def get_all_pair_combination_from_list(l):
    return [(a, b) for idx, a in enumerate(l) for b in l[idx + 1:]]

def take_closest(myList, myNumber, return_index=False):  # adapted from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        if return_index:
            result = pos
        else:
            result = after
    else:
        if return_index:
            result = pos -1
        else:
            result = before

    return result

def center_bins(bin_list):
    bin_list = np.array(bin_list)
    return (bin_list[:-1] + bin_list[1:]) / 2

def pad_to_length(x, m):
    return np.pad(x,((0, 0), (0, m - x.shape[1])), mode = 'constant')