import sys
import sysconfig
import importlib.util

lib_path = sysconfig.get_path("purelib")

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

file_path = f"{lib_path}/torch/version.py"
module = import_module_from_path("version", file_path)

if sys.argv[1] == "torch":
    print(module.__version__.split("+")[0])
elif sys.argv[1] == "cuda":
    print(module.cuda)
