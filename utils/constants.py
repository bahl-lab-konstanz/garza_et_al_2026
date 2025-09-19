from enum import Enum


class Keyword(Enum):
    INPUT = 'Input'
    OUTPUT = 'Output'
    RIGHT = 90
    LEFT = -90

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

RestStateValue = -1
StraightBout = -1
CorrectBout = 1
IncorrectBout = 0
CorrectBoutColumn = 'correct_bout'
ResponseTimeColumn = 'interbout_interval'
AngleChangeColumn = 'estimated_orientation_change'


class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    STRAIGHT = -1
