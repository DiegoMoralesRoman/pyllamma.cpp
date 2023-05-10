from pyllamacpp.splash.splash import splash
from . import model_inteface
import argparse
import logging
from . import models

log = logging.getLogger('pyllamacpp')

MODEL_INTERFACES = {}
try:
    MODEL_INTERFACES = models.load_models_from_file('models.json')
except Exception as e:
    exit(1)

if len(MODEL_INTERFACES) == 0:
    log.critical(f'''
    There are no model interafces defined.
    To create a new model interface add a python module to the folder model_interfaces inside the module and
    and add the selectable (in flag) name for it to the MODEL_INTERFACES dictionary
    TODO: load this dictionary from a file
    ''')

DEFAULT_INTERFACE = list(MODEL_INTERFACES.keys())[0]

parser = argparse.ArgumentParser(description='''
    This CLI application allows interfacing with the llama.cpp family models easily from the terminal
''')

parser.add_argument(
    '--model-interface', '-i',
    action='store',
    help=f'Specified the model interface to use. This interfaces are included in the model_interfaces directory. Default: {DEFAULT_INTERFACE}',
    choices=list(MODEL_INTERFACES.keys()),
    default=DEFAULT_INTERFACE
)

parser.add_argument(
    '--model-path', '-m',
    action='store',
    help='Specifies the LLaMA model\'s path'
)

parser.add_argument(
    '--prompt', '-p',
    action='store',
    help='Prompt that\'s passed to the model'
)

parser.add_argument(
    '--no-splash',
    action='store_true',
    help='Disables the initial splash banner (useful for scripts)'
)

args = parser.parse_args()

SELECTED_MODEL = MODEL_INTERFACES[args.model_interface]

if not args.model_path and SELECTED_MODEL.path_required:
    log.error(f'''
    Model path was not specified.
    To specifiy a path use the --model-path or -p flags.
    For more information use --help
    ''')
    exit(1)

# Load model interface
log.info(f'Loading model interface "{args.model_interface}"')
model_inteface.init_interface(SELECTED_MODEL.module)

if not args.no_splash:
    splash()

print('')
log.info(f'Loading LLaMA model from "{args.model_path}"...')
model_inteface.load_model(args.model_path)

# Try embedding
tokens = model_inteface.tokenize(args.prompt)
print([tok.token for tok in tokens])
