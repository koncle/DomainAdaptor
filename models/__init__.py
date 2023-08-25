import os
import importlib


def import_all_modules_in_current_folders():
    imported_modules = []
    for module in os.listdir(os.path.dirname(__file__)):
        if module == '__init__.py' or module[-3:] != '.py':
            continue
        importlib.import_module('.' + module[:-3], __package__)  # '.' before module_name is required
        imported_modules.append(module)
    del module
    print('Successfully imported modules : ', imported_modules)

import_all_modules_in_current_folders()