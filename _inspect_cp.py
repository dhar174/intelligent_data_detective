import sqlite3, json, sys
sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect(r'C:\Users\darf3\Documents\intelligent_data_detective\checkpoints.sqlite')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', tables)
for t in tables:
    tname = t[0]
    cols = conn.execute(f'PRAGMA table_info({tname})').fetchall()
    print(f'Table {tname}: {[c[1] for c in cols]}')
    cnt = conn.execute(f'SELECT COUNT(*) FROM {tname}').fetchone()
    print(f'  Rows: {cnt[0]}')
    # Show latest few rows 
    rows = conn.execute(f'SELECT * FROM {tname} ORDER BY rowid DESC LIMIT 3').fetchall()
    for row in rows:
        print(f'  row: {str(row)[:200]}')
