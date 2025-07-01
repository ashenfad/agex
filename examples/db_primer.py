PRIMER = """
# Database Cursor Primer

Essential concepts for working with SQLite cursors in natural language database tasks.

## CRITICAL: Cursors Cannot Be Assigned to Variables

When you call `db.execute()`, it returns a cursor object that CANNOT be stored in a variable:

```python
# ❌ THIS WILL FAIL - Cannot assign unpickleable cursor objects
cursor = db.execute("SELECT * FROM users")  # ERROR!
results = cursor.fetchall()  # This won't work
```

## ✅ Correct Approach: Chained Method Calls

You MUST chain the fetch methods directly to the execute call:

```python
# ✅ Get all rows - chain .fetchall() directly
all_users = db.execute("SELECT * FROM users").fetchall()

# ✅ Get one row - chain .fetchone() directly  
first_user = db.execute("SELECT * FROM users LIMIT 1").fetchone()

# ✅ Count records - chain .fetchone() and access result
count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

# ✅ Get limited rows - chain .fetchmany() directly
some_users = db.execute("SELECT * FROM users LIMIT 5").fetchmany(5)
```

## Using `with` Statements

### For Transactions (INSERT/UPDATE/DELETE)

For INSERT/UPDATE/DELETE operations, use `with` statements:

```python
# ✅ Safe transaction - use 'with' for modifications
with db as connection:
    connection.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("John", "john@example.com"))
    connection.execute("UPDATE users SET age = ? WHERE name = ?", (25, "John"))
    # Transaction automatically commits on exit
```

### For Cursor Iteration

For iterating through large result sets, you can use cursors with `with` statements:

```python
# ✅ Iterate through cursor - transient variables handle unpickleable cursors
with db.execute("SELECT * FROM users WHERE age > ?", (25,)) as cursor:
    for row in cursor:
        print(f"User: {row[1]}, Email: {row[2]}, Age: {row[3]}")
        
# ✅ Process cursor in chunks
with db.execute("SELECT * FROM large_table") as cursor:
    while True:
        rows = cursor.fetchmany(100)  # Process 100 rows at a time
        if not rows:
            break
        for row in rows:
            # Process each row
            pass
```

## Error Handling

Handle constraint violations which are mapped to ValueError:

```python
try:
    with db as connection:
        connection.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Bob", "existing@example.com"))
except ValueError as e:
    print(f"Database constraint violation: {e}")
```

## Quick Reference - What Works

- ✅ **Chained queries**: `db.execute("SELECT ...").fetchall()`
- ✅ **Chained single row**: `db.execute("SELECT ...").fetchone()`
- ✅ **Chained count**: `db.execute("SELECT COUNT(*) FROM table").fetchone()[0]`
- ✅ **Safe updates**: `with db as conn: conn.execute("INSERT ...")`
- ✅ **Cursor iteration**: `with db.execute("SELECT ...") as cursor: for row in cursor:`
- ✅ **Parameterized queries**: Use `?` placeholders for safety

## What Doesn't Work

- ❌ **Storing cursors**: `cursor = db.execute(...)` - Will cause assignment errors
- ❌ **Two-step process**: Must chain fetch methods immediately
- ❌ **Direct modifications**: Always use `with` statements for INSERT/UPDATE/DELETE

Remember: The cursor restriction is a safety feature to prevent unpickleable objects in agent state!
"""
