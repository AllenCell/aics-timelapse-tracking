from timelapsetracking.dir_to_nodes import dir_to_nodes
from timelapsetracking.version import MODULE_VERSION


__all__ = ['dir_to_nodes', 'MODULE_VERSION']
__version__ = MODULE_VERSION


def get_module_version():
    return MODULE_VERSION
