
# Creates the configuration to pass with the payload of an onshape request
from enum import Enum
import numpy as np


class Units(Enum):
    DEGREE="deg"
    METER="m"
    RADIAN="rad"


class ValueWithUnit:
    def __init__(self, value, unit: Units):
        self.unit = unit.value
        self.value = value


class ConfigurationEncoder:
    def __init__(self):
        self.encoding = ""
        self.numpyParameters = np.array([])

    def clear_encoding(self):
        self.encoding = ""

    def add_configuration(self, name : str, value : ValueWithUnit):
        self.encoding += name + "="
        self.encoding += str(value.value) + " "
        self.encoding += value.unit + ";"
        self.numpyParameters = np.append(self.numpyParameters, value.value)

    def get_encoding(self):
        return self.encoding


class KinematicSampleConfigurationEncoder(ConfigurationEncoder):
    def __init__(self, unitsList: list = None, numpyParameters: np.ndarray = None):
        super(KinematicSampleConfigurationEncoder, self).__init__()
        self.parameterIndex = 0
        self.unitsList = unitsList
        if unitsList is not None:
            self.numDim = len(unitsList)
        if numpyParameters is not None:
            self.add_parameters(numpyParameters)

    def add_parameter(self, value : ValueWithUnit):
        self.parameterIndex += 1
        self.add_configuration("Parameter" + str(self.parameterIndex), value)

    def add_parameters(self, numpyParameters: np.ndarray):
        if self.unitsList is not None:
            for i in range(self.numDim):
                self.add_parameter(ValueWithUnit(numpyParameters[i], self.unitsList[i]))
        else:
            print("Configuration encoder: unitsList is None")