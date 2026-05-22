import toml


def _read_config(configpath):
    with open(configpath, "r", encoding="utf-8") as f:
        return toml.load(f)
