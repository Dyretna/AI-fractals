from .tile_search import BaseTileSearch, TileSearchBasic, TileSearchJittered

STRATEGIES = {
    "tile_basic": TileSearchBasic,
    "tile_jittered": TileSearchJittered,
}


def format_available_strategies():
    rows = ["Available strategies: "]
    for str_key, cls in STRATEGIES.items():
        rows.append(f"{str_key:10} : {cls.__name__}")
    return "    \n".join(rows)


def create_search_strategy(strategy: str) -> BaseTileSearch:
    try:
        return STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"Unregistered strategy '{strategy}', {format_available_strategies()}"
        )
