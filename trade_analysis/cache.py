import json, time, pathlib, hashlib
CACHE_DIR = pathlib.Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
TTL = 60*60*24  # 24 h
def _key(name): return CACHE_DIR / f"{hashlib.md5(name.encode()).hexdigest()}.json"

def put(n, obj):
    _key(n).write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')

def get(n):
    fp = _key(n)
    if not fp.exists():
        return None
    try:
        data = fp.read_text(encoding='utf-8')
        return json.loads(data)
    except Exception as e:
        print(f"Cache error: {e}")
        return None