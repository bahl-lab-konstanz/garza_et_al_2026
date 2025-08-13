import copy
from abc import ABC

from analysis.personal_dirs.Roberto.style import Style
from analysis.personal_dirs.Roberto.utils.palette import Palette


class BehavioralModelStyle(Style):
    def __init__(self, plot_label_i=0, stimulus_palette=Palette.green_short):
        Style.__init__(self)
        self.plot_label_i = plot_label_i
        self.add_palette("stimulus", stimulus_palette)


    palette = {"default": Palette.arlecchino,
               "correct_incorrect": Palette.correct_incorrect,
               "neutral": [Palette.color_neutral],
               "green": Palette.green_short,
               "fish_code": ["#73489C", "#753B51", "#103882", "#7F0C0C"]
               }

    font_size_label = 8
    font_size_text = 6

    padding = 2
    plot_size = 1

    plot_height = 1

    plot_width = 1
    plot_width_large = 3

    xpos_start = 2
    ypos_start = 27

    page_tight = False

    plot_label_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q" ,"r", "s", "t", "u", "v", "w", "x" ,"y", "z"]

    def add_palette(self, label: str, palette: list):
        self.palette[label] = copy.deepcopy(palette)

    def get_plot_label(self):
        label_to_show = self.plot_label_list[plot_label_i]
        self.plot_label_i += 1
        return label_to_show