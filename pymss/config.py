import re

import yaml


class ConfigLoader(yaml.FullLoader):
    pass


ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^[-+]?(
            ([0-9][0-9_]*)?\.[0-9_]+([eE][-+]?[0-9]+)?
            |[0-9][0-9_]*[eE][-+]?[0-9]+
            |\.(inf|Inf|INF)
            |\.(nan|NaN|NAN)
        )$""",
        re.X,
    ),
    list("-+0123456789."),
)


class AttrDict(dict):
    def __init__(self, data=None, **kwargs):
        super().__init__()
        for key, value in dict(data or {}, **kwargs).items():
            self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, to_attrdict(value))


def to_attrdict(value):
    if isinstance(value, AttrDict):
        return value
    if isinstance(value, dict):
        return AttrDict(value)
    if isinstance(value, list):
        return [to_attrdict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_attrdict(item) for item in value)
    return value


def to_plain(value):
    if isinstance(value, AttrDict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain(item) for item in value)
    return value


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return to_attrdict(yaml.load(f, Loader=ConfigLoader))
