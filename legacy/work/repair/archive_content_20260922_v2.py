"""Version a documentation-only manuscript update, entirely on gpu01."""
import hashlib, json, shutil, subprocess, tarfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT=Path('/mnt/data/caliper/code/CALIPER')
DEST=Path('/mnt/data/caliper/archives/CALIPER_companion_archive_history')
NAME='CALIPER_companion_archive_20260922_v2_CURRENT_FROZEN'
OLD=DEST/'CALIPER_companion_archive_20260922_v1_CURRENT_FROZEN.tar.gz'
STAGE=DEST/(NAME+'.staging')
OUT=DEST/(NAME+'.tar.gz')
INPUT=DEST/'refresh_20260922_v2_inputs'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def extract(source,dest,strip=False):
    with tarfile.open(source,'r|*') as t:
        for m in t:
            if m.isdir():continue
            assert m.isfile()
            rel=Path(*Path(m.name).parts[1:]) if strip else Path(m.name)
            p=dest/rel
            assert not rel.is_absolute() and p.resolve().is_relative_to(dest.resolve())
            p.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(m) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)

def main():
    assert not STAGE.exists() and not OUT.exists()
    assert sha(OLD)=='f94bd37d3081b7608c806507cd8642306d36f0c794f00c0c5d04aed81ef6a89e'
    STAGE.mkdir(); extract(OLD,STAGE,True)
    for r in read(STAGE/'manifest.json')['files']:
        p=STAGE/r['path'];assert p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256']
    print('PREDECESSOR_VERIFIED',flush=True)
    hist=STAGE/'historical/pre_20260922_v2';hist.mkdir()
    for n in ('paper_snapshot','manifest.json','FREEZE_STATUS.json','FREEZE_RECEIPT.json','CURRENT_EVIDENCE_INDEX.json','README.md'):
        if (STAGE/n).exists():shutil.move(str(STAGE/n),str(hist/n))
    paper=STAGE/'paper_snapshot';paper.mkdir()
    extract(INPUT/'paper_snapshot.tar',paper)
    shutil.copy2(INPUT/'cas-dc-template.pdf',paper/'cas-dc-template.pdf')
    evidence=STAGE/'documentation_evidence_20260922';evidence.mkdir()
    sources=[ROOT/'logs'/n for n in ['ft_embed_hotpotqa.log','ft_embed_2wiki_retry.log','ft_mer_embedder.log','ft_mer_reranker.log','train_merged_v2.log','unified_bgebase_musique_embed_20260920.log','unified_large_ftsep_20260920_final.log','merged_cache_hotpotqa_ckpt5000.log','merged_cache_2wiki_ckpt5000.log']]
    sources += [Path('/tmp')/n for n in ['build_ms_first.py','build_fulltrain_jsonl.py','merged_cache_build_ckpt5000.py']]
    sources += [ROOT/'work'/n for n in ['router_e2e_rerun_20260918.py','router_e2e_v2v4_rerun_20260919.py','finalize_router_e2e_20260918.py','finalize_router_e2e_musique_unified_bgebase_20260920.py','rebuild_unified_musique_cache.py','rebuild_hp_tw_large_20260921.py','rebuild_routed_large_20260921.py']]
    sources += [ROOT/'retrieve_fine_tuning/models/merged_bge-reranker-ft/checkpoint-5000/trainer_state.json']
    mapping={}
    for p in sources:
        target=evidence/p.name;assert not target.exists()
        shutil.copy2(p,target);mapping[str(p)]={'artifact':str(target.relative_to(STAGE)),'sha256':sha(target)}
    write(evidence/'source_mapping.json',mapping)
    shutil.copy2(Path(__file__),evidence/Path(__file__).name)
    oldtext=(hist/'paper_snapshot/cas-dc-template.tex').read_text()
    newtext=(paper/'cas-dc-template.tex').read_text()
    import re
    tables=lambda s: [re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',x) for x in re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',s,re.S)]
    assert tables(oldtext)==tables(newtext),'Table numerical content changed'
    assert re.findall(r'coordinates \{[^}]+\}',oldtext)==re.findall(r'coordinates \{[^}]+\}',newtext),'Figure coordinates changed'
    timestamp=datetime.now(timezone.utc)
    status={'archive_version':'2026-09-22-v2','status':'CURRENT_FROZEN','is_current_frozen':True,'frozen_at_utc':timestamp.isoformat(),'frozen_at_local':timestamp.astimezone(timezone(timedelta(hours=8))).isoformat(),'supersedes_archive_version':'2026-09-22-v1','paper_git_commit':(INPUT/'paper_commit.txt').read_text().strip(),'paper_sha256':sha(paper/'cas-dc-template.tex'),'pdf_sha256':sha(paper/'cas-dc-template.pdf'),'scope':'Documentation and layout only; all table numbers and plotted coordinates unchanged. Supplementary configuration notes and source logs added.','verification':'Content hashes and numerical preservation verified; final archive hash is in external CURRENT_FROZEN_ARCHIVE.json to avoid self-reference.'}
    write(STAGE/'FREEZE_STATUS.json',status)
    finish(status,paper,hist)

def finish(status,paper,hist):
    receipt=read(hist/'FREEZE_RECEIPT.json') if (hist/'FREEZE_RECEIPT.json').exists() else {'scope':'Documentation-only receipt; prior result-specific freeze receipts remain at their original indexed paths.'}
    receipt['documentation_update']=status;write(STAGE/'FREEZE_RECEIPT.json',receipt)
    index=read(hist/'CURRENT_EVIDENCE_INDEX.json');index['documentation_update']={'version':'2026-09-22-v2','notes':'paper_snapshot/Supplementary_Configuration_Notes.md','source_map':'documentation_evidence_20260922/source_mapping.json','quality_and_latency_results_unchanged':True};write(STAGE/'CURRENT_EVIDENCE_INDEX.json',index)
    (STAGE/'README.md').write_text('# CALIPER companion archive 2026-09-22-v2\n\nCurrent paper_snapshot includes the bibliography and Figure 3 layout fixes, clarified V2/branch definitions, Supplementary Configuration Notes, and narrowed dual-stream wording. All result tables and plotted coordinates are unchanged. See CURRENT_EVIDENCE_INDEX.json, FREEZE_STATUS.json and documentation_evidence_20260922/source_mapping.json. The preceding README and paper snapshot are retained under historical/pre_20260922_v2/.\n\nTable 5 remains a composed estimate, not an integrated live run. Context-transfer assumptions remain unverified; no new experiments were run for this update. Missing historical difficulty-classifier training logs remain a limitation. Archive SHA-256 is recorded externally, not recursively inside the hashed archive.\n')
    files=[{'path':str(p.relative_to(STAGE)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(STAGE.rglob('*')) if p.is_file() and p!=STAGE/'manifest.json']
    write(STAGE/'manifest.json',{'files':files,'version':'2026-09-22-v2'})
    expected={r['path']:r for r in files};expected['manifest.json']={'size_bytes':(STAGE/'manifest.json').stat().st_size,'sha256':sha(STAGE/'manifest.json')}
    print('PACKAGING',len(files),flush=True)
    subprocess.run(['tar','-I','pigz -p 4 -1','-cf',str(OUT),'-C',str(DEST),STAGE.name],check=True)
    seen=set()
    with tarfile.open(OUT,'r|gz') as t:
        for m in t:
            if not m.isfile():continue
            rel=str(Path(*Path(m.name).parts[1:]));r=expected[rel];h=hashlib.sha256()
            with t.extractfile(m) as f:
                for b in iter(lambda:f.read(8<<20),b''):h.update(b)
            assert m.size==r['size_bytes'] and h.hexdigest()==r['sha256'];seen.add(rel)
    assert seen==set(expected)
    pointer=dict(status,archive_path=str(OUT),remote_archive_path='gpu01:'+str(OUT),archive_sha256=sha(OUT),archive_size_bytes=OUT.stat().st_size,verification='PASS',verified_file_count=len(files),local_archive_present=False)
    write(DEST/'CURRENT_FROZEN_ARCHIVE.json',pointer);write(DEST/'VERIFICATION_20260922_v2.json',pointer)
    (DEST/(OUT.name+'.sha256')).write_text(pointer['archive_sha256']+'  '+OUT.name+'\n')
    obsolete=DEST/'CALIPER_companion_archive_20260921_v3_CURRENT_FROZEN.tar.gz'
    assert obsolete.parent.resolve()==DEST.resolve()
    if obsolete.exists():
        assert sha(obsolete)=='cf352db57f2f01f794fc985e9e8d6ec518476cf72fef17fb0f6f51ee9fdcb1de';obsolete.unlink()
        side=DEST/(obsolete.name+'.sha256')
        if side.exists():side.unlink()
    assert STAGE.parent.resolve()==DEST.resolve() and STAGE.name==NAME+'.staging';shutil.rmtree(STAGE)
    print('VERIFIED_CURRENT',json.dumps(pointer),flush=True)
if __name__=='__main__':
    import sys
    if '--resume' in sys.argv:
        assert STAGE.exists() and not OUT.exists()
        status=read(STAGE/'FREEZE_STATUS.json');paper=STAGE/'paper_snapshot';hist=STAGE/'historical/pre_20260922_v2'
        assert status['paper_sha256']==sha(paper/'cas-dc-template.tex') and status['pdf_sha256']==sha(paper/'cas-dc-template.pdf')
        for source,r in read(STAGE/'documentation_evidence_20260922/source_mapping.json').items():assert sha(STAGE/r['artifact'])==r['sha256']
        shutil.copy2(Path(__file__),STAGE/'documentation_evidence_20260922'/Path(__file__).name)
        finish(status,paper,hist)
    else:main()
