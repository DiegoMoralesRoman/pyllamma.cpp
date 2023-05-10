from dataclasses import dataclass
import os.path
import json
import logging
from typing import Dict

log = logging.getLogger('pyllamacpp')


@dataclass
class Model:
    """
    A dataclass that represents a model.
    """
    module: str
    path_required: bool


def validate_json(json_data):
    """
    Validates JSON data against a specific structure.
    @param json_data: The JSON data to validate.
    @return: True if the JSON data matches the expected structure.
    @raises ValueError: If the JSON data does not match the expected structure.
    """
    required_keys = {"module", "path_required"}

    if not isinstance(json_data, dict):
        raise ValueError("Invalid JSON data: not a dictionary")

    for key, value in json_data.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid JSON data: '{key}' value is not a dictionary")

        if not required_keys.issubset(value.keys()):
            missing_keys = required_keys - value.keys()
            raise ValueError(f"Invalid JSON data: missing keys {missing_keys} in '{key}'")

        if not isinstance(value["module"], str):
            raise ValueError(f"Invalid JSON data: 'module' in '{key}' is not a string")

        if not isinstance(value["path_required"], bool):
            raise ValueError(f"Invalid JSON data: 'path_required' in '{key}' is not a boolean")

    return True


def load_models_from_file(path: str) -> Dict[str, Model]:
    """
    Loads a JSON file and validates its structure against an expected schema.
    @param path: The path to the JSON file to load.
    @return: A dictionary of Model objects if the structure is valid.
    @raises FileNotFoundError: If the JSON file cannot be found.
    @raises json.decoder.JSONDecodeError: If the JSON data is not valid.
    @raises RuntimeError: If the JSON data does not match the expected structure.
    """
    path_to_open = os.path.join(os.path.dirname(__file__), path)
    try:
        with open(path_to_open) as json_file:
            objects = json.load(json_file)
            try:
                validate_json(objects)
            except ValueError as e:
                msg = f'''
    Model list read:
    {objects}

    Error: {e}

    Expected JSON structure:
    {{
        "module_name": {{
            "module": "module_string",
            "path_required": boolean
        }}
    }}
                '''
                log.error(msg)
                raise RuntimeError("Invalid JSON structure") from e

            model_dict = {}
            for key, value in objects.items():
                model_dict[key] = Model(**value)

            return model_dict

    except FileNotFoundError as e:
        msg = f'''
    Couldn't open file "{path_to_open}"
        '''
        log.critical(msg)
        log.critical(e)
        raise e
    except json.decoder.JSONDecodeError as e:
        msg = f'''
    Failed to decode file
        {path_to_open}
    Error:
        {e}
        '''
        log.error(msg)
        raise e
