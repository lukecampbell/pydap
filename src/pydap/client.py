from pydap.handlers.dap import DAPHandler


def open_url(url):
    return DAPHandler(url).dataset
