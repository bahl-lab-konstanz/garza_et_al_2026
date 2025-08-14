from enum import Enum


class Keyword(Enum):
    INPUT = 'Input'
    OUTPUT = 'Output'
    RIGHT = 90
    LEFT = -90


marker = ["x", "1", "h", "*", "p", "P", "v", "s", "^", "2", "<", "X", ">", "+", "3", "8", "o", "H", "4", "d", "|", "_"]
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q" ,"r", "s", "t", "u", "v", "w", "x" ,"y", "z"]
color_list = ["deepskyblue", "deeppink", "orange", "springgreen", "tomato", "silver", "mediumorchid", "mediumaquamarine", "firebrick", "blue", "red", "green", "brown"]
palette_0 = ["#FF03C8", "#B37DE4", "#66F7FF", "#85FF0A", "#FBFF0A", "#000000"]
palette_1 = ["#CB6BFF", "#9951C0", "#663781", "#331D42", "#000000"]
palette_2 = ["#00FFFF", "#00C0C0", "#008080", "#004040", "#000000"]
palette_3 = ["#FF03C8", "#CB6BFF", "#9951C0", "#663781", "#008080", "#00C0C0", "#00FFFF", "#66F7FF"]
palette_long = ["deepskyblue", "deeppink", "orange", "springgreen", "tomato", "silver", "mediumorchid", "mediumaquamarine", "firebrick", "blue", "red", "green", "brown", "#FF03C8", "#CB6BFF", "#9951C0", "#663781", "#008080", "#00C0C0", "#00FFFF", "#66F7FF", "rosybrown", "slategrey", "darkorchid", "gold", "darkgreen", "peru"]

class BaseColor(Enum):
    BLUE = 'tag:blue'
    ORANGE = 'tag:orange'
    GREEN = 'tag:green'
    RED = 'tag:red'
    PURPLE = 'tag:purple'
    BROWN = 'tag:brown'
    PINK = 'tag:pink'
    GRAY = 'tag:gray'
    OLIVE = 'tag:olive'
    CYAN = 'tag:cyan'


class StimulusParameterLabel(Enum):
    SEPARATOR = '_'

    DIRECTION = "dir"
    DIRECTION_0 = "dir0"
    DIRECTION_1 = "dir1"
    COHERENCE = "coh"
    DENSITY = "den"
    SPEED = "speed"
    LIFETIME = "lifetime"
    BRIGHTNESS = "brig"
    SIZE = "size"
    CORRECT = "corr"
    TIME_FORGET = "tf"
    PERIOD = "per"

class StimulusParameterLabel2Dir(Enum):
    SEPARATOR = '_'

    DIRECTION_0 = "dir0"
    DIRECTION_1 = "dir1"
    COHERENCE = "coh"
    SPEED = "speed"
    LIFETIME = "lifetime"
    CORRECT = "corr"
    PERIOD = "per"


class ExperimentalParameterRange():
    coh = [
    {
        "label": "0%-20%",
        "min": 0,
        "max": 20
    },
    {
        "label": "20%-40%",
        "min": 20,
        "max": 40
    },
    {
        "label": "40%-60%",
        "min": 40,
        "max": 60
    },
    {
        "label": "60%-80%",
        "min": 60,
        "max": 80
    },
    {
        "label": "80%-100%",
        "min": 80,
        "max": 100
    }
]

    den = [
    {
        "label": "100-750",
        "min": 100,
        "max": 750
    },
    {
        "label": "750-1500",
        "min": 750,
        "max": 1500
    },
    {
        "label": "1500-2200",
        "min": 1500,
        "max": 2200
    },
    {
        "label": "2200-3000",
        "min": 2200,
        "max": 3000
    }
]


class RestState(Enum):
    MOTION_COHERENCE_0 = 'motion_coherence_0'
    STATIC = 'static'
    BLACK = 'black'


RestStateValue = -1
StraightBout = -1
CorrectBout = 1
IncorrectBout = 0
CorrectBoutColumn = 'correct_bout'
ResponseTimeColumn = 'interbout_interval'
ResponseTimeColumnAlternative = 'response_time'
AngleChangeColumn = 'estimated_orientation_change'


class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    STRAIGHT = -1

class StimulusArchive():

    # important to have the attributes defined in __init__(), so that I can get stimulus dict using:
    # stim_dict = StimulusArchive().get_dict()

    def __init__(self):
        self.grating_motion_right = {
            "dir": 90,
            "coh": 100,
        }

        self.grating_motion_left = {
            "dir": -90,
            "coh": 100,
        }

        self.grating_motion_forward = {
            "dir": 0,
            "coh": 100,
        }

        self.baseline_gray = {
            "dir": 90,
            "coh": 0,
        }

    def get_dict(self):
        return self.__dict__
