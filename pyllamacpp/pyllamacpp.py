import importlib.util
import inspect
import os.path
import logging

log = logging.getLogger('pyllamacpp')

SO_LOCATION = os.path.join(os.path.dirname(__file__), 'libpyllamacpp.so')

spec = importlib.util.spec_from_file_location('pyllamacpp', SO_LOCATION)

if not spec:
    log.error(f'''
    Failed to import library located at {SO_LOCATION}
    This library is neccessary for the adapter module to work
    ''')
    exit(1)
else:
    try:
        mod = importlib.util.module_from_spec(spec)
    except ImportError as e:
        log.error(f'''
        Failed to load "{SO_LOCATION}"
            File not found
        ''')
        exit(1)

    spec.loader.exec_module(mod)

def load_model(path: str):
    mod.load_model(path)
