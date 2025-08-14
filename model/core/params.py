class Parameter():
    def __init__(self, min=None, max=None, value=None, fittable=False, fluctuation_law=None):
        self.min = min
        self.max = max
        self.value = value
        self.fittable = fittable
        self.fluctuation_law = fluctuation_law

    def fluctuate(self, *attr):
        pass

class ParameterList():

    def __init__(self, parameter_name_list: object = None) -> object:
        if parameter_name_list is not None:
            for parameter_name in parameter_name_list:
                self.add_parameter(parameter_name, Parameter())

    def add_parameter(self, label, value):
        setattr(self, label, value)

    def __iter__(self):
        attribute_list = [{"key": attr_label, "value": getattr(self, attr_label)} for attr_label in dir(self) if not (attr_label.startswith('__') or attr_label == "add_parameter")]
        for attribute in attribute_list:
            yield attribute["key"], attribute["value"]
