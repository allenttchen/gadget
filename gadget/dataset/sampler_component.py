from dataclasses import dataclass
import functools

from ..utils.dataclasses import field
from ._abstract_bases import ComponentBase


@dataclass
class SamplerBase(ComponentBase):
    sample_rate: float = field(required=True)
    seed: int = field(default=0)
    additional_sample_columns: list = field(
        default=None, 
        help=(
            "The additional list of columns to be used for partitioning before subsampling. "
            "The subsmapling makes sure the same proportion of data points are chosen from each partition. "
            "By default the fold column and lable column are used"
        )
    )
        
        
@dataclass
class AblationSampler(SamplerBase):
    
    def get_function(self):
        pass
    

@dataclass
class Subsampler(SamplerBase):
    
    def get_function(self):
        pass
