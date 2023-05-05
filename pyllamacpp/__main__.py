from pyllamacpp.splash.splash import splash
from . import pyllamacpp
import argparse
import logging

log = logging.getLogger('pyllamacpp')

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-path', '-p',
    action='store',
    help='Specifies the LLaMA model\'s path'
)

args = parser.parse_args()

if not args.model_path:
    log.error(f'''
    Model path was not specified.
    To specifiy a path use the --model-path or -p flags.
    For more information use --help
    ''')
    exit(1)

splash()
print('')
log.info(f'Loading LLaMA model from "{args.model_path}"...')
pyllamacpp.load_model(args.model_path)
