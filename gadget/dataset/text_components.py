from dataclasses import dataclass
import functools

from ..utils.dataclasses import field
from ._abstract_bases import ComponentBase
from .text_functions import (
    toLowerCase, 
    replaceEmailToken, 
)


@dataclass
class ToLowerCase(ComponentBase):

    def get_function(self):
        return toLowerCase


@dataclass
class ReplaceEmailToken(ComponentBase):

    def get_function(self):
        return replaceEmailToken
