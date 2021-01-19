import abc

_DATASET_REGISTRY = {}
_CONFIG_REGISTRY = {}
_SCHEMA_REGISTRY = {}
_SPLIT_HANDLER_REGISTRY = {}
_TRAINER_REGISTRY = {}

class ABCWithRegistry(abc.ABC):
	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		REGISTRY = cls._get_registry()
		REGISTRY[cls.__name__] = cls

	@classmethod
	def get_class(_, class_name):
		REGISTRY = _._get_registry()
		if class_name not in REGISTRY:
			raise Exception(
				f"{class_name} not a known subclass. "
				f"Known subclasses: {tuple(REGISTRY.keys())}"
			)
		return REGISTRY[class_name]

	@classmethod
	def _registry_contains(_, class_name):
		REGISTRY = _._get_registry()
		return class_name in REGISTRY

	@abc.abstractstaticmethod
	def _get_registry():
		raise NotImplemented


class AbstractDataset(ABCWithRegistry):
	@staticmethod
	def _get_registry():
		return _DATASET_REGISTRY

	@abc.abstractmethod
	def get_data(self):
		pass

	@abc.abstractmethod
	def set_split(self, split):
		pass


class AbstractDatasetConfig(ABCWithRegistry):
	@staticmethod
	def _get_registry():
		return _CONFIG_REGISTRY

	@abc.abstractmethod
	def load_dataframe(self):
		pass

	@abc.abstractmethod
	def validate(self, **validation_kwargs):
		pass


class AbstractSchema(ABCWithRegistry):
	@staticmethod
	def _get_registry():
		return _SCHEMA_REGISTRY


class AbstractSplitHandler(ABCWithRegistry):
	@staticmethod
	def _get_registry():
		return _SPLIT_HANDLER_REGISTRY

	@abc.abstractclassmethod
	def from_dataframe(cls, df, partition_column):
		return cls.from_raw(df[partition_column].values)

	@abc.abstractclassmethod
	def from_raw(cls, raw_partition_vector):
		pass

	@abc.abstractmethod
	def items(self):
		pass


class AbstractTrainer(ABCWithRegistry):
	@staticmethod
	def _get_registry():
		return _TRAINER_REGISTRY

	@abc.abstractclassmethod
	def get_spec(_):
		pass

	@abc.abstractmethod
	def validate(self):
		pass

	@abc.abstractclassmethod
	def from_broadcast(cls, hparams, broadcast_dict):
		pass

	@abc.abstractmethod
	def make_tracker(self):
		pass

	@abc.abstractmethod
	def get_results(self):
		pass

	@abc.abstractmethod
	def set_global_seeds(self):
		pass

	@abc.abstractmethod
	def reset(self, **reset_kwargs):
		pass

	@abc.abstractmethod
	def run_training(self, train_bar=None):
		pass

	@abc.abstractmethod
	def run_eval(self, split="val", split_log_name="val", log_model_output=False, val_bar=None):
		pass

	@abc.abstractmethod
	def run(self, reset=True):
		pass

	@abc.abstractmethod
	def _run(self, reset=True):
		pass

	@abc.abstractmethod
	def make_dataset(self):
		pass

	@abc.abstractmethod
	def make_model(self):
		pass

	@abc.abstractmethod
	def compute_model_output(self, data_dict):
		pass

	@abc.abstractmethod
	def compute_metrics(self, data_dict, model_output):
		pass



































