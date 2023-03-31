from dataclasses import dataclass

from ._abstract_bases import AbstractDatasetTransformer
from ..utils.dataclasses import field


@dataclass
class Pipeline:
    """
    Example:
        pipeline = Pipeline([...])
        df = pipeline.transform(df)
    """
    transformers: list[str, AbstractDatasetTransformer] = field(default_factory=list)

    def transform(self, df):
        df_copy = df.copy(deep=True)
        for transformer in self.transformers:
            if isinstance(transformer, str):
                transformer = eval(transformer)
            df_copy = transformer(df_copy)
        return df_copy
