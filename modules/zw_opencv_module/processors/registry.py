_REGISTRY: dict[str, type] = {}

def register_processor(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def get_processor(name: str) -> type:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown processor: {name!r}, available: {list(_REGISTRY)}")
    return _REGISTRY[name]
