import json, time, pathlib, hashlib
CACHE_DIR = pathlib.Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
TTL = 60*60*24  # 24 h
def _key(name): return CACHE_DIR / f"{hashlib.md5(name.encode()).hexdigest()}.json"
def get(n): fp=_key(n); return None if not fp.exists() or time.time()-fp.stat().st_mtime>TTL else json.loads(fp.read_text())
def put(n,obj): _key(n).write_text(json.dumps(obj,ensure_ascii=False))
