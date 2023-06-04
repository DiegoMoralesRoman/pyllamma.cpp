from types import ModuleType
import importlib
from typing import Optional
import logging

log = logging.getLogger('pyllamacpp')

mod : Optional[ModuleType] = None

def init_interface(modname: str):
    """
    Loads a module with the specified name and stores it in a global variable.
    @param modname: The name of the module to load.
    @raises ModuleNotFoundError: If the specified module cannot be found.
    """
    global mod
    full_modname = f'.model_interfaces.{modname}'
    try:
        mod = importlib.import_module(full_modname, package=__package__)
    except ModuleNotFoundError as e:
        if mod == None:
            msg = f'''
        Failed to load module {full_modname}
            '''
            log.error(msg)
            raise ModuleNotFoundError(msg)


def check_initialized():
    """
    Checks if the global module variable has been initialized.
    @raises ImportError: If the module variable has not been initialized.
    """
    global mod
    if mod == None:
        raise ImportError(f'''
    An interface module has not been provided so no models are available
        ''')

# Add bindings from loaded module (this has to be done manually to get the LSPs to work)
def ensure_initialized(func):
    """
    A decorator that checks if the global module variable has been initialized before calling the decorated function.
    @param func: The function to decorate.
    """
    def wrapper(*args, **kwargs):
        global mod
        check_initialized()
        return func(*args, **kwargs)
    return wrapper

@ensure_initialized
def load_model(path: str):
    """
    Loads a model using the loaded module.
    @param path: The path to the model file.
    @return: The loaded model.
    """
    return mod.load_model(path)

@ensure_initialized
def tokenize(string: str):
    """
    Tokenizes a string using the loaded module.
    @param string: The string to tokenize.
    @return: The tokenized string.
    """
    return mod.tokenize(string)

@ensure_initialized
def generate_embeddings(prompt: str):
    """
    Generates embeddings from prompt with loaded model
    @param prompt Text to generate the emebddings from
    @return List[Float] of embeddings
    """
    return mod.generate_embeddings(prompt)
