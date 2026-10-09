"""Remote-only archive refresh. Verify predecessor and every output before promotion."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/mnt/data/caliper/code/CALIPER')
DEST=Path('/mnt/data/caliper/archives/CALIPER_companion_archive_history')
NAME='CALIPER_companion_archive_20260922_v1_CURRENT_FROZEN'
OLD=DEST/'CALIPER_companion_archive_20260921_v3_CURRENT_FROZEN.tar.gz'
STAGE=DEST/(NAME+'.staging')
OUT=DEST/(NAME+'.tar.gz')
INPUT=DEST/'refresh_20260922_v1_inputs'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')

def main():
    assert not STAGE.exists() and not OUT.exists()
    assert sha(OLD)=='cf352db57f2f01f794fc985e9e8d6ec518476cf72fef17fb0f6f51ee9fdcb1de'
    STAGE.mkdir()
    print('PREDECESSOR_HASH_PASS',flush=True)
    with tarfile.open(OLD,'r|gz') as tar:
        for m in tar:
            if m.isdir():continue
            assert m.isfile()
            rel=Path(*Path(m.name).parts[1:]);target=STAGE/rel
            assert target.resolve().is_relative_to(STAGE.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    oldmanifest=read(STAGE/'manifest.json')
    for f in oldmanifest['files']:
        p=STAGE/f['path']; assert p.stat().st_size==f['size_bytes'] and sha(p)==f['sha256'],str(p)
    print('PREDECESSOR_CONTENTS_PASS',flush=True)
    hist=STAGE/'historical/pre_20260922_v1';hist.mkdir()
    for n in ['paper_snapshot','README.md','FREEZE_STATUS.json','CURRENT_EVIDENCE_INDEX.json','manifest.json']:
        shutil.move(str(STAGE/n),str(hist/n))
    (STAGE/'paper_snapshot').mkdir()
    with tarfile.open(INPUT/'paper_snapshot_20260922.tar') as tar:
        for m in tar:
            if m.isdir():continue
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            p=STAGE/'paper_snapshot'/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
    shutil.copy2(INPUT/'cas-dc-template.pdf',STAGE/'paper_snapshot/cas-dc-template.pdf')
    result=ROOT/'latency_runs/composed_table5_20260922_v1'
    current=STAGE/'results/table5_composed_20260922'
    shutil.copytree(result,current)
    for d in ('hotpotqa','musique','2wikimultihopqa'):
        v='v2' if d=='musique' else 'v1'
        source=ROOT/f'retrieval_runs/path_latency_per_question_20260922_{v}/{d}'
        shutil.copytree(source,current/'retrieval'/d)
    scripts=STAGE/'scripts/current_20260922';scripts.mkdir(parents=True)
    for n in ['compose_routed_table5_20260922.py','compose_table5_end_to_end_20260921.py',
              'measure_route_latency_20260922.py','test_compose_routed_table5_20260922.py',
              'test_measure_route_latency_20260922.py','archive_table5_20260922.py']:
        shutil.copy2(ROOT/'work/repair'/n,scripts/n)
    # All historical generation inputs are copied explicitly; no reliance on stale aggregates.
    raw=current/'raw_generation';raw.mkdir()
    for d in ('hotpotqa','musique','2wikimultihopqa'):
        for n in (f'table4_timing_{d}_seed42.json',f'table4_clf_actual_{d}_seed42.json',f'table4_cloud_deep_{d}_full_b256_seed42.json'):
            shutil.copy2(ROOT/'table-result'/n,raw/n)
    for p,h in read(current/'input_hashes.json').items():assert sha(Path(p))==h,p
    summary=read(current/'summary.json');paper=(STAGE/'paper_snapshot/cas-dc-template.tex').read_text()
    import statistics,numpy as np
    for d,b in summary['datasets'].items():
        records=read(current/f'{d}_paired_composition.json')
        assert len(records)==800
        for method,s in b['conditions'].items():
            rows=[r for r in records if r['method']==method and r['error'] is None]
            assert len(rows)==s['n']
            for r in rows:assert abs(r['seconds']-r['historical_generation_seconds']-r['retrieval_seconds_added'])<1e-10
            values=[r['seconds'] for r in rows]
            assert abs(statistics.mean(values)-s['mean'])<1e-10
            assert abs(float(np.percentile(values,50))-s['p50'])<1e-10
            assert abs(float(np.percentile(values,95))-s['p95'])<1e-10
            cell='\\shortstack{'+f'{s["mean"]:.2f}'+r'\\'+f'({s["p50"]:.2f}/{s["p95"]:.2f})'+'}'
            assert cell in paper,(d,method,cell)
    now=datetime.now(timezone.utc).isoformat()
    receipt={'status':'CURRENT_FROZEN_COMPOSED_ESTIMATE','frozen_at_utc':now,
        'definition':'New branch-matched retrieval + historical generation; NOT integrated live latency.',
        'generation_transfer_assumption':'Generation time reused despite changed contexts; not validated by this composition.',
        'classifier_rule':'Archived local branch uses fixed retrieval; collaborative branch uses routed_clf minus classifier_seconds.',
        'route_replay':'Two MuSiQue source assignments replay frozen ms although live classifier predicts hp.',
        'hardware':'retrieval timing on one A40 per dataset, workers=4; historical generation measured separately',
        'summary_sha256':sha(current/'summary.json'),'bootstrap_sha256':sha(current/'paired_bootstrap.json'),
        'context_matches':{d:b['context_matches'] for d,b in summary['datasets'].items()},
        'per_question_sha256':{d:sha(current/f'{d}_paired_composition.json') for d in summary['datasets']}}
    write(current/'FREEZE_RECEIPT.json',receipt)
    index=read(hist/'CURRENT_EVIDENCE_INDEX.json')
    index.update(Table5_summary='results/table5_composed_20260922/summary.json',
                 Table5_statistics='results/table5_composed_20260922/paired_bootstrap.json',
                 Table5_freeze='results/table5_composed_20260922/FREEZE_RECEIPT.json')
    index['scope_note']=index.get('scope_note','')+' Table5 20260921 aggregate is historical; current 20260922 branch-matched composition retains historical generation assumptions.'
    write(STAGE/'CURRENT_EVIDENCE_INDEX.json',index)
    freeze={'archive_version':'2026-09-22-v1','status':'CURRENT_FROZEN','is_current_frozen':True,
        'frozen_at_utc':now,'supersedes_archive_version':'2026-09-21-v3',
        'paper_git_commit':(INPUT/'paper_commit.txt').read_text().strip(),
        'paper_sha256':sha(STAGE/'paper_snapshot/cas-dc-template.tex'),
        'pdf_sha256':sha(STAGE/'paper_snapshot/cas-dc-template.pdf'),
        'scope':'Table5 new composed estimates and scoped manuscript corrections; Table2/4/8 quality unchanged.',
        'evidence_index':'CURRENT_EVIDENCE_INDEX.json'}
    write(STAGE/'FREEZE_STATUS.json',freeze)
    (STAGE/'README.md').write_text('# CALIPER companion archive — 2026-09-22-v1\n\nCURRENT_FROZEN at '+now+'\n\nStart with CURRENT_EVIDENCE_INDEX.json. Table5 has new branch-matched composed estimates, not live integrated timings. Read its FREEZE_RECEIPT.json for context mismatches, two frozen-route replays, historical generation settings and the unvalidated transfer assumption. The discarded/aborted live benchmark is NOT used.\n\nTable2/4/8 quality evidence is unchanged. Historical Table5 aggregates and manuscript snapshots are superseded only by the explicit current pointers. Original absolute paths document the producing environment; this package does not redistribute all benchmark/model files. The execution-classifier original training command/hyperparameter log remains unavailable.\n')
    files=[{'path':str(p.relative_to(STAGE)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(STAGE.rglob('*')) if p.is_file() and p!=STAGE/'manifest.json']
    write(STAGE/'manifest.json',{**freeze,'current_evidence':index,'files':files})
    print('NEW_EVIDENCE_AND_TABLE_PASS',len(files),flush=True)
    subprocess.run(['tar','--use-compress-program=pigz -p 4 -1','--transform',f's,^{STAGE.name},{NAME},','-cf',str(OUT),'-C',str(DEST),STAGE.name],check=True)
    expected={f['path']:f for f in files};seen=set()
    with tarfile.open(OUT,'r|gz') as tar:
        for m in tar:
            if m.isdir():continue
            assert m.isfile()
            rel='/'.join(Path(m.name).parts[1:]);h=hashlib.sha256();size=0
            with tar.extractfile(m) as f:
                for b in iter(lambda:f.read(8<<20),b''):h.update(b);size+=len(b)
            if rel=='manifest.json':assert h.hexdigest()==sha(STAGE/rel);continue
            assert rel in expected and rel not in seen
            assert h.hexdigest()==expected[rel]['sha256'] and size==expected[rel]['size_bytes'],rel
            seen.add(rel)
    assert seen==set(expected)
    pointer={**freeze,'remote_archive_path':'gpu01:'+str(OUT),'archive_path':str(OUT),
        'archive_sha256':sha(OUT),'archive_size_bytes':OUT.stat().st_size,'verification':'PASS',
        'verified_file_count':len(seen),'local_archive_present':False,
        'retention':'Remote current 20260922-v1 and predecessor 20260921-v3; no new large local download.'}
    write(DEST/'CURRENT_FROZEN_ARCHIVE.json',pointer)
    write(DEST/'VERIFICATION_20260922_v1.json',pointer)
    (DEST/(OUT.name+'.sha256')).write_text(pointer['archive_sha256']+'  '+OUT.name+'\n')
    # Only known superseded v1 package; v3 predecessor and new current both retained.
    retired=DEST/'CALIPER_companion_archive_20260921_v1_CURRENT_FROZEN.tar.gz'
    assert retired.parent.resolve()==DEST.resolve() and retired not in (OUT,OLD)
    if retired.exists():
        assert sha(retired)=='97653190a1c40c25d9d6d471deaccdd755a99f86e2abe0624ac47711efbf3713'
        retired.unlink()
        side=Path(str(retired)+'.sha256')
        if side.exists():side.unlink()
    assert STAGE.parent.resolve()==DEST.resolve() and STAGE.name==NAME+'.staging'
    shutil.rmtree(STAGE)
    print('VERIFIED_CURRENT',json.dumps(pointer),flush=True)

if __name__=='__main__':main()
