PRIMER = """
# Database Operations Primer

Essential patterns for working with SQLite databases in natural language database tasks.

## Recommended Patterns

### Chained Method Calls (Preferred)

For simple queries, chain operations for clean, one-line execution:

```python
# ✅ Get all rows
all_users = db.execute("SELECT * FROM users").fetchall()

# ✅ Get one row
first_user = db.execute("SELECT * FROM users LIMIT 1").fetchone()

# ✅ Count records
count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
```

### Transactions with Context Managers

For INSERT/UPDATE/DELETE operations, use `with` statements for safe transactions:

```python
# ✅ Safe transaction - automatically commits on success
with db as connection:
    connection.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("John", "john@example.com"))
    connection.execute("UPDATE users SET age = ? WHERE name = ?", (25, "John"))
```

### Direct Assignment (Works in Single Turn)

You can assign cursors to variables for single-turn operations:

```python
# ✅ Works fine in a single turn
cursor = db.execute("SELECT * FROM users")
results = cursor.fetchall()
```

Note: If you try to reuse `cursor` in a later turn, you'll get a helpful error with solutions.

## Error Handling

Handle constraint violations (mapped to ValueError):

```python
try:
    with db as connection:
        connection.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Bob", "existing@example.com"))
except ValueError as e:
    print(f"Database constraint violation: {e}")
```

## Quick Reference

- ✅ **Chained queries**: `db.execute("SELECT ...").fetchall()`
- ✅ **Transactions**: `with db as conn: conn.execute("INSERT ...")`
- ✅ **Direct assignment** (single-turn): `cursor = db.execute(...)`
- ✅ **Parameterized queries**: Always use `?` placeholders for safety
"""
