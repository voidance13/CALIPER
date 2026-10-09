import json, pathlib, hashlib, statistics, datetime
root = pathlib.Path('/mnt/data/caliper/code/CALIPER')
run = root / 'retrieval_runs/routed_large_20260921'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
s = json.loads((run/'results.json').read_text())
rows = [json.loads(x) for p in sorted(run.glob('shard_*.jsonl')) for x in p.read_text().splitlines()]
assert len(rows) == len({(x['dataset'], x['row_id']) for x in rows}) == 24274
assert all(sha(pathlib.Path(p)) == h for p,h in s['shard_sha256'].items())
for d,b in s['datasets'].items():
    subset = [x for x in rows if x['dataset']==d]
    assert len(subset)==b['n']
    for k,v in b['new'].items():
        assert round(statistics.mean(x['recall'][k] for x in subset)*100,2)==v
ftpath = root/'retrieval_runs/hp_tw_large_20260921/results.json'
ft = json.loads(ftpath.read_text())
for b in ft.values():
    assert sha(pathlib.Path(b['cache']))==b['sha256']
receipt = {'status':'CURRENT_FROZEN_TABLE8_RESULT_UPDATE', 'frozen_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), 'verification':'PASS: routed count, unique IDs, metrics and shard hashes; FT-SEP cache hashes', 'scope':'New HotpotQA/2Wiki FT-SEP and all three routed cells; BASE, FT-MER and MuSiQue full-train FT-SEP retained from manuscript', 'routed':s, 'ftsep':ft, 'source_sha256':{str(run/'results.json'):sha(run/'results.json'),str(ftpath):sha(ftpath),str(run/'manifest.json'):sha(run/'manifest.json')}, 'historical_status_note':'Original candidate results and RUNNING manifest preserved; this adoption receipt supersedes their status only.', 'full_companion_archive_current':False}
print(json.dumps(receipt,indent=2))
