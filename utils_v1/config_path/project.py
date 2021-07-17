def get_root_file(file):
    import pathlib
    return str(pathlib.Path(file).parent)
