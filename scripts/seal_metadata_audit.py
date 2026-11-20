"""Record the completed one-hour audit with artifact hashes and release checks."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ebnerd-metadata-audit'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
protocol=json.loads((OUT/'protocol.json').read_text());now=datetime.now(timezone.utc)
started=datetime.fromisoformat(protocol['started_at_utc']);until=datetime.fromisoformat(protocol['requested_work_until_utc'])
if now<until:raise SystemExit(f'Requested work window still active: {(until-now).total_seconds():.0f} seconds remain')
verify=json.loads((OUT/'verification.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
assert verify['models_verified']==summary['models_fitted']==153
for name,expected in protocol['previous_artifact_sha256'].items():assert digest(ROOT/name)==expected,name
for row in verify['runs']:assert digest(ROOT/'models/ebnerd-metadata-audit'/row['name']/'model.txt')==row['model_sha256']
for prefix in ['exposure','exposure_d3600','exposure_d86400']:
    record=json.loads((OUT/f'{prefix}_feature_verification.json').read_text());assert record['all_features_exactly_equal']
assert json.loads((OUT/'reference_verification.json').read_text())['all_counts_hashes_and_field_differences_reproduced']
stress=json.loads((OUT/'exposure_stress.json').read_text());assert stress['processed']==60000 and stress['exact_parity_passed']
registry={}
for path,model,version in [('runs/v2/mlflow.db','fairjob-v2',13),('runs/ebnerd/mlflow.db','ebnerd-v1',5)]:
    connection=sqlite3.connect('file:'+str((ROOT/path).resolve())+'?mode=ro',uri=True)
    result=connection.execute('select version from registered_model_aliases where name=? and alias=?',(model,'champion')).fetchone();connection.close()
    assert result and int(result[0])==version
    registry[model]=version
code=[p for p in (ROOT/'scripts').glob('*.py') if any(token in p.name for token in ['metadata','exposure','archiv','article_reference'])]
code += [ROOT/'src/fairjob_ranker/ebnerd/metadata_audit.py',ROOT/'src/fairjob_ranker/ebnerd/exposure_audit.py',
         ROOT/'tests/test_ebnerd_metadata_audit.py',ROOT/'tests/test_ebnerd_exposure_audit.py']
reports=[p for p in OUT.iterdir() if p.is_file() and p.name!='run_manifest.json']
docs=[ROOT/p for p in ['README.md','docs/PROJECT_STORY.md','docs/EBNERD_METADATA_AUDIT.md','docs/FEATURE_AVAILABILITY.md']]
files={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(code+reports+docs))}
result={'started_at_utc':started.isoformat(),'completed_at_utc':now.isoformat(),'elapsed_seconds':(now-started).total_seconds(),
    'requested_minutes':60,'required_work_window_completed':True,'audit_models':153,
    'recomputed_prediction_rows':verify['prediction_rows_recomputed'],'independent_metric_queries':verify['independent_metric_queries'],
    'unit_tests':{'observed_passed':7,'command':'python3 scripts/run.py -m pytest tests/test_ebnerd_metadata_audit.py tests/test_ebnerd_exposure_audit.py -q'},
    'original_artifact_hashes_unchanged':True,'registry_champions_unchanged':registry,
    'sustained_in_memory_replay':{'requests':stress['processed'],'duration_seconds':stress['elapsed_seconds'],'exact_parity':True,
        'deadline_misses':stress['deadline_misses'],'scope':stress['scope']},
    'conclusion_scope':'Post-hoc audit; original metadata leakage contribution is unidentified. New exposure models retain explicit log-arrival assumptions. No new champion.',
    'sha256':files}
(OUT/'run_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sha256'},indent=2))
