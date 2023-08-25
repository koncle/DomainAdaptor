"""
Copy from maskrcnn-benchmark
"""
import functools
import inspect


def _register_generic(module_dict, func_name, func, parse_name):
    assert func_name not in module_dict, 'Key of "{}" from "{}" already defined in "{}"'.format(
        func_name, inspect.getsourcefile(func), module_dict.get_src_file(func_name))
    if parse_name:
        func = functools.partial(func, registed_name=func_name)
    module_dict[func_name.lower()] = func


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''

    def __init__(self, name, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)
        self.name = name

    def register(self, module_name, module=None, parse_name=False):
        # '-' is reserved as extra param for calling the function
        assert '-' not in module_name, "Function name should not contain '-'"

        # used as function call
        if module is not None:
            _register_generic(self, module_name, module, parse_name)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn, parse_name)
            return fn

        return register_fn

    def __getitem__(self, item):
        """
        if function name is used with 'func-xx',
        the item after the first '-' will be passed to the function as param
        """
        splits = item.split('-', 1)
        item = splits[0].lower()
        func = super(Registry, self).__getitem__(item)
        if len(splits) == 1:
            return func
        else:
            return functools.partial(func, param=splits[1])

    def __repr__(self):
        return "Registry-{}".format(self.name)

    def get_src_file(self, module_name):
        module = self[module_name]
        src_file = inspect.getsourcefile(module)
        return src_file


Models = Registry('Models')
Datasets = Registry('Datasets')
LossFuncs = Registry('LossFuncs')
AccFuncs = Registry('AccFuncs')
EvalFuncs = Registry('EvalFuncs')
TrainFuncs = Registry('TrainFuncs')
Schedulers = Registry('Schedulers')
Backbones = Registry('Backbones')

Entries = [Models, Datasets, LossFuncs, AccFuncs, EvalFuncs, TrainFuncs, Schedulers, Backbones]


def show_entries_and_files():
    for entry in Entries:
        print(f'\n{entry.name} : [')
        for key in entry.keys():
            print(f'\t"{key}" from "{entry.get_src_file(key)}"')
        print(']')

