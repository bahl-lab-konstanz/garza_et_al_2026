from enum import Enum

class ConfigurationExperiment():
    time_start_stimulus = 10  # (s)
    time_end_stimulus = 40  # (s)
    time_experimental_trial = 50  # (s)

    coherence_list = [0, 25, 50, 100]  # (%)

    ResponseTimeColumn = "interbout_interval"
    CorrectBoutColumn = "correct_bout"

    coherence_label = "Coh (%)"
    all_fish_label = "all"
    example_fish_list = ["001", "002", "003"]  # identifier of three example fish

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
    RIGHT = 90  # (deg)
    LEFT = -90  # (deg)