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
# Text embeddings generation
prompts = [
    'Poetry',
    'Literature',
    'Calculus',
    'Engineering',
    'Maths'
]

# master_prompt = 'Engineering'
# print(f'Generating embeddings for: {master_prompt}')
# base_embedding = list(model_inteface.generate_embeddings(master_prompt))
# comparison_embeddings = []
# for prompt in prompts:
#     print(f'Generating embeddings for: {prompt}')
#     comparison_embeddings.append((prompt, list(model_inteface.generate_embeddings(prompt))))
#
# # Compare results
# import numpy as np
#
# def cosine_similarity(vec1, vec2):
#     # Compute the dot product of vec1 and vec2
#     dot_product = np.dot(vec1, vec2)
#
#     # Compute the L2 norms (or Euclidean norms) of vec1 and vec2
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
#
#     # Compute the cosine similarity
#     cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
#
#     return cosine_similarity
#
# print(f'Similarities with {master_prompt}')
# for prompt, embeddings in comparison_embeddings:
#     print(f'{prompt}: {cosine_similarity(np.array(base_embedding), np.array(embeddings))}')
