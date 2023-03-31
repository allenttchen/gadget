import abc
from typing import Callable
from dataclasses import dataclass

from ..utils.dataclasses import field


class AbstractDatasetTransformer(abc.ABC):
    
    @abc.abstractmethod
    def transform(self, dataset, col="text"):
        pass
    

class DatasetTransformerBase(AbstractDatasetTransformer):
    
    @abc.abstractmethod
    def _transform(self, item, meta_dict=dict()):
        pass
    
    def transform(self, dataset, col="text"):
        return dataset.apply(
            lambda x: self._transform(dataset[col]), 
            axis=1, 
        )
    
    def __call__(self, dataset):
        return self.transform(dataset)
    

@dataclass
class ComponentBase(DatasetTransformerBase):
    transform_func: Callable = field(init=False, serializes=False, default=None, repr=False)
        
    def __post_init__(self):
        self.transform_func = self.get_function()
        
    @abc.abstractmethod
    def get_function(self):
        pass
    
    def _transform(self, item, meta_dict=dict()):
        return self.transform_func(item, **meta_dict)
