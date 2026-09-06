import sqlite3


def get_connection(db_path: str = "tickets.db", check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    return conn
