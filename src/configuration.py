from frozendict import frozendict
from collections.abc import Mapping

class Configuration(Mapping):
    """
    Sort of inspired by here: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    __frozen = False
    def __init__(self, d):
        # Oh boy recursion...
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = Configuration(value)
        self.config = frozendict(d)
        self.__frozen = True

    def __iter__(self):
        for k in self.config:
            yield k

    def __len__(self):
        return self.config.__len__()

    def __getitem__(self, key):
        return self.config.__getitem__(key)

    def __getattr__(self, attr):
        return self.config.get(attr)

    def __setattr__(self, name, value):
        if name != '__frozen' and self.__frozen:
            raise TypeError('Configuration cannot be changed after construction')
        super().__setattr__(name, value)

    def __contains__(self, key):
        return key in self.config

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def to_dict(self):
        # Oh boy, here we go again with this recursion
        d = {**self.config}
        for key, value in d.items():
            if isinstance(value, self.__class__):
                d[key] = value.to_dict()
        return d
