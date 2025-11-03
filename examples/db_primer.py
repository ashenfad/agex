PRIMER = """
# SQLite Quick Guide

You work directly with an sqlite3 connection.
- Use parameterized queries (`?` placeholders) for safety.
- For simple queries, call `db.execute(...).fetchone()` / `.fetchall()`.
- For updates/inserts, use a context manager so commits happen automatically:
  ```python
  with db as conn:
      conn.execute("INSERT INTO users (name, email, age) VALUES (?, ?, ?)", (name, email, age))
  ```
- Catch `ValueError` to handle constraint violations.

Example:
```python
# Create table
with db as conn:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            age INTEGER
        )
    ''')

# Read rows
top_users = db.execute("SELECT name, age FROM users ORDER BY age DESC LIMIT 5").fetchall()
```
"""
