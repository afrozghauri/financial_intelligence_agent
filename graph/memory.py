from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

def get_memory():
    conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
    return SqliteSaver(conn)