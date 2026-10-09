"""Seal a fresh MuSiQue sample before any new timing measurement."""
import hashlib,json,random,time
from pathlib import Path

ROOT=Path('/mnt/data/caliper/code/CALIPER')
OUT=ROOT/'latency_runs/musique_resample_seed20260922_v1'
TEST=ROOT/'data/musique/test.json'
EXCLUSIONS=ROOT/'table-result/table2_musique_paired2403_metrics.json'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    tests=json.loads(TEST.read_text())
    audit=json.loads(EXCLUSIONS.read_text())
    excluded={r['index'] for r in audit['excluded']}
    eligible=[i for i in range(len(tests)) if i not in excluded]
    assert len(tests)==2417 and len(excluded)==14 and len(eligible)==2403
    chosen=random.Random(20260922).sample(eligible,200)
    selected=[{'draw_position':j,'source_index':i,'id':tests[i]['id'],'question':tests[i]['question']} for j,i in enumerate(chosen)]
    assert len({r['source_index'] for r in selected})==200
    assert len({r['question'] for r in selected})==200
    result={'status':'SEALED_BEFORE_MEASUREMENT','sealed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'dataset':'musique','population':'Table 2 common 2403-question intersection','population_n':2403,
        'seed':20260922,'algorithm':'Python random.Random(seed).sample(eligible_indices, 200)',
        'sample_n':200,'test_path':str(TEST),'exclusions_path':str(EXCLUSIONS),
        'input_sha256':{str(TEST):sha(TEST),str(EXCLUSIONS):sha(EXCLUSIONS)},'selected':selected,
        'replacement_policy':'No outcome-based replacement. Failures and slow successful requests remain visible.',
        'paper_archive_modified':False}
    (OUT/'SEALED_SAMPLE.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(OUT/'SEALED_SAMPLE.json')

if __name__=='__main__':main()
