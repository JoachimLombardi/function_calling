import inspect
from django.core.files.storage import default_storage



def save_file(save_path, image_file):
    if default_storage.exists(save_path):
        default_storage.delete(save_path)
    default_storage.save(save_path, image_file)


def detect_func_signature(func, params):
    """
    Detects the signature of the given function and calls it with the given parameters accordingly.

    If the function takes only one argument, it will be called with the given parameters as a single argument.
    If the function takes more than one argument, it will be called with the given parameters as keyword arguments.

    Parameters:
        func (function): The function to call.
        params (dict): The parameters to pass to the function.

    Returns:
        The result of the function call.
    """
    sig = inspect.signature(func)
    param_count = len(sig.parameters)

    if param_count == 1:
        # Function is called with a single argument 
        return func(params)
    else:
        # Function is called with keyword arguments
        return func(**params)
