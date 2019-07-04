import platform

system = platform.system().lower()

if system == 'darwin':
    from .MacOSController import MacOSController as Controller
elif system == 'windows':
    from .WindowsController import WindowsController as Controller
else:
    raise NotImplementedError(
        "No controller implemented for platform '{system}'".format(system)
    )
