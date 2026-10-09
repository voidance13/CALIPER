import importlib.metadata as md
import json
from pathlib import Path
from supplement_table5_generation_20260922 import ROOT,RUN,sha,dump
paths=[Path('/mnt/data/caliper/models/Qwen3.5-9B')]
files=[]
for p in paths:
    files+=list(p.glob('*.safetensors'))+[p/'config.json',p/'generation_config.json',p/'tokenizer_config.json']
files += [ROOT/'query_clf/train_new/9b_clf.bin',ROOT/'query_clf/stopwords.txt']
versions={}
for name in ('openai','fasttext','jieba','numpy'):
    try:versions[name]=md.version(name)
    except md.PackageNotFoundError:versions[name]='not recorded'
dump(RUN/'execution_provenance.json',{'file_sha256':{str(p):sha(p) for p in files if p.is_file()},'client_environment_versions':versions,
    'secrets_excluded':True,'historical_generation_seed_note':'Historical scripts use seed 42 for sample selection, not per-call local/cloud sampling. The refresh preserves omitted per-call seeds.',
    'historical_weights_equivalence':'Current file hashes recorded; historical generation result JSON itself does not establish weight hash identity.'})
print('PROVENANCE_RECORDED')
