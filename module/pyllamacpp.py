import ctypes
import os.path

SO_LOCATION = os.path.join(os.path.dirname(__file__), 'libpyllamacpp.so')
mod = ctypes.cdll.LibraryLoader(SO_LOCATION)

print(mod.say_hello)
