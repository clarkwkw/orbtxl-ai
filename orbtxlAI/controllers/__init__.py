import platform


def configure_tesseract_path(path):
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = path


system = platform.system().lower()

if system == 'darwin':
    from .MacOSController import MacOSController as Controller
elif system == 'windows':
    from .WindowsController import WindowsController as Controller
    configure_tesseract_path(
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    )
else:
    raise NotImplementedError(
        "No controller implemented for platform '{system}'".format(system)
    )
