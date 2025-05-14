import copy
from abc import ABC

from analysis.personal_dirs.Roberto.style import Style
from analysis.personal_dirs.Roberto.utils.palette import Palette


class BehavioralModelStyle(Style):
    def __init__(self):
        Style.__init__(self)

        self.add_palette("stimulus", Palette.green_short)
        self.palette["stimulus"].append("#000000")


    palette = {"default": Palette.arlecchino,
               "correct_incorrect": Palette.correct_incorrect}

    font_size_label = 8
    font_size_text = 6

    padding = 2
    plot_height = 1

    plot_width = 1
    plot_width_large = 3

    xpos_start = 2
    ypos_start = 27

    page_tight = False

    def add_palette(self, label: str, palette: list):
        self.palette[label] = copy.deepcopy(palette)