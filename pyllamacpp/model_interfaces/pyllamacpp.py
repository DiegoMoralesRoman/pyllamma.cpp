import importlib.util
import os.path
import logging
import inspect

log = logging.getLogger('pyllamacpp')

SO_LOCATION = os.path.join(os.path.dirname(__file__), 'libpyllamacpp.so')

spec = importlib.util.spec_from_file_location('pyllamacpp', SO_LOCATION)

if not spec:
    log.error(f'''
    Failed to import library located at {SO_LOCATION}
    This library is necessary for the adapter module to work
    ''')
    exit(1)
else:
    try:
        mod = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(mod)

        # Define all available functions explicitly
        def load_model(path: str):
            return mod.load_model(path)

        def tokenize(string: str):
            return mod.tokenize(string)

        def generate_embeddings(string: str):
            return mod.generate_embeddings(string)

    except ImportError as e:
        log.error(f'''
        Failed to load "{SO_LOCATION}"
            Exception: {e}
        ''')
        exit(1)

# Add the defined functions to the module namespace
globals()['load_model'] = load_model
globals()['tokenize'] = tokenize
globals()['generate_embeddings'] = generate_embeddings
