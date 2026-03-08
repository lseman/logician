_SQLITE_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;",
    "PRAGMA foreign_keys=ON;",
    "PRAGMA cache_size=-65536;",  # ~64MB
)
