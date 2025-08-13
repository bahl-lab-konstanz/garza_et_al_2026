from enum import Enum

time_start_stimulus = 10  # seconds
time_end_stimulus = 40  # seconds
time_experimental_trial = 50  # seconds

ResponseTimeColumn = "interbout_interval"
CorrectBoutColumn = "correct_bout"

class StimulusParameterLabel(Enum):
    SEPARATOR = '_'

    DIRECTION = "dir"
    COHERENCE = "coh"
    DENSITY = "den"
    SPEED = "speed"
    LIFETIME = "lifetime"
    BRIGHTNESS = "brig"
    SIZE = "size"

class Keyword(Enum):
    INPUT = 'Input'
    OUTPUT = 'Output'
    RIGHT = 90
    LEFT = -90