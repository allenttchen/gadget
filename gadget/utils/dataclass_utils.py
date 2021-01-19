"""
Define fields, validate_dataclass, _asdictenhanced_inner, update_frozen
"""
import copy
from dataclasses import (
	fields,
	asdict,
	field as field_,
	_is_dataclass_instance
)
import functools


def field(*args, options=None, serializes=True, required=False, **kwargs):
	metadata = dict(serializes=serializes, required=required)
	if options is not None:
		metadata["options"] = options
	if required
		kwargs["default"] = None
	return field_(*args, metadata=metadata, **kwargs)


def validate_dataclass(instance):
	field_lookup = {field_obj.name: field_obj for field_obj in fields(instance)}
	for key, value in asdict_enhanced(instance).items():
		field_obj = field_lookup[key]

		options = field_obj.metadata,get("options", None)
		if options is not None and value not in options:
			raise Exception(
				f"For {repr(instance)}: Invalid option for {key}."
				f"Got `{value}` but expected one of {options}"
			)

		required = field_obj.metadata.get("required", False)

		if required and value is None:
			raise Exception(
				f"For {repr(instance)}: {key} was required but was not provided."
			)


def _asdictenhanced_inner(
	instance,
	dict_factory,
	serializes_only=False, 
	include_cls_name=False
) -> "type(dict_factory())":
	
	instance_type = type(instance)
	recurse_func = functools.partial(
		_asdictenhanced_inner, 
		dict_factory=dict_factory, 
		serializes_only=serializes_only, 
		include_cls_name=include_cls_name
	)
	if _is_dataclass_instance(instance):
		#dataclass case
		field_dict = dict_factory()

		if include_cls_name:
			field_dict["cls"] = instance.__class__.__name__

		for field_obj in fields(instance):
			if serializes_only and not field_obj.metadata.get("serializes", True):
				continue
			value = recurse_func(instance=getattr(instance, field_obj.name))
			field_dict[field_obj.name] = value

		return field_dict

	if isinstance(instance, tuple) and hasattr(instance, '_fields'):
		# Named tuple case
		# Can't pass in generators
		return instance_type(*list(map(recurse_func, instance)))

	if isinstance(instance, (list, tuple)):
		# Assume we can create an object of this type by passing in a generator (which is not true for namedtuples, handled abovec)
		return instance_type(map(recurse_func, instance))

	if isinstance(instance, dict):
		recursed_items = [
			(recurse_func(key), recurse_func(value))
			for key, value in instance.items()
		]
		if hasattr(instance, "default_factory"):
			return instance_type(instance.default_factory, recursed_items)
		else:
			return instance_type(recursed_items)
	return copy.deepcopy(instance)


def update_frozen(instance, **updated_fields):
	if not _is_dataclass_instance(instance):
		raise Exception("Not a dataclass")
	inst_dict = asdict_enhanced(instance, serializes_only=True)
	if len(set(updated_fields.keys()) - set(inst_dict.keys())) > 0:
		raise Exception("Too many keys")

	inst_dict.update(updated_fields)
	cls = instance.__class__
	return cls(**inst_dict)




























