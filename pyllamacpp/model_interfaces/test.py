# test.py
#
# A test module for the pyllamacpp package. Implements the `load_model()` and `tokenize()` functions for testing purposes.

def load_model(path: str):
    """
    Loads a model from the specified path.
    @param path: The path to the model file.
    """
    print(f'Loading model from path: {path}')

def tokenize(string: str):
    """
    Tokenizes a string.
    @param string: The string to tokenize.
    @return: A list of integer ordinals representing the characters in the string.
    """
    return [ord(c) for c in string]
