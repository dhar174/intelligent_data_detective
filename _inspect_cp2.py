import sqlite3, json, sys, msgpack
sys.stdout.reconfigure(encoding="utf-8")
conn = sqlite3.connect(r"C:\Users\darf3\Documents\intelligent_data_detective\checkpoints.sqlite")

# Get the latest checkpoint id
latest = conn.execute("SELECT thread_id, checkpoint_id FROM checkpoints ORDER BY rowid DESC LIMIT 1").fetchone()
print(f"Latest checkpoint: {latest}")
tid, cid = latest

# Get all pending writes for this checkpoint
writes = conn.execute("SELECT task_id, idx, channel, type, value FROM writes WHERE thread_id=? AND checkpoint_id=? ORDER BY task_id, idx", (tid, cid)).fetchall()
print(f"\nPending writes ({len(writes)} total):")
for w in writes:
    task_id, idx, channel, wtype, value = w
    if wtype == "msgpack" and value:
        try:
            decoded = msgpack.unpackb(value, raw=False)
        except Exception as e:
            decoded = f"<decode error: {e}>"
        val_repr = str(decoded)[:100]
    else:
        val_repr = str(value)[:100]
    print(f"  task={task_id[:8]} idx={idx} channel={channel} type={wtype} value={val_repr}")
