"""Schema, temporal/history and cold-item feasibility audit, without fitting models."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data/raw/ebnerd-small/extracted'
OUTPUT = ROOT / 'reports/ebnerd-audit'
FIT_END = pd.Timestamp('2023-05-23 07:00:00')
TEST_START = pd.Timestamp('2023-05-25 07:00:00')
BEHAVIOR = ['impression_id', 'user_id', 'impression_time', 'article_ids_inview', 'article_ids_clicked']
HISTORY = ['article_id_fixed', 'impression_time_fixed', 'read_time_fixed', 'scroll_percentage_fixed']


def partition(times, fit_end=FIT_END, test_start=TEST_START):
    if not fit_end < test_start or times.isna().any():
        raise ValueError('Invalid temporal boundary or missing request time')
    return np.where(times < fit_end, 'train', np.where(times < test_start, 'validation', 'test'))


def past_events(ids, times, request_time):
    """Strict past only, stable sort; equal timestamps must not enter features."""
    if len(ids) != len(times):
        raise ValueError('Unaligned history lists')
    ts = np.asarray(times, dtype='datetime64[us]')
    if np.isnat(ts).any():
        raise ValueError('Missing history timestamp')
    keep = ts < np.datetime64(request_time, 'us')
    ids, ts = np.asarray(ids)[keep], ts[keep]
    order = np.argsort(ts, kind='stable')
    return ids[order], ts[order]


def ids_in(frame, column):
    return set(int(i) for row in frame[column] for i in row)


def history_audit(history, behaviors, article_ids):
    result = {'rows': len(history), 'users': int(history.user_id.nunique()),
              'duplicate_user_rows': int(history.user_id.duplicated().sum()),
              'unaligned_rows': 0, 'unsorted_rows': 0, 'null_timestamp_events': 0,
              'not_strictly_before_first_request': 0, 'events': 0,
              'unknown_catalog_events': 0, 'empty_histories': 0}
    first = behaviors.groupby('user_id').impression_time.min().to_dict()
    result['request_users_without_history'] = len(set(first)-set(history.user_id))
    minimum, maximum = None, None
    lengths = []
    for row in history.itertuples(index=False):
        values = [getattr(row, c) for c in HISTORY]
        if any(v is None for v in values) or len({len(v) for v in values}) != 1:
            result['unaligned_rows'] += 1
            continue
        ids, ts = row.article_id_fixed, np.asarray(row.impression_time_fixed, dtype='datetime64[us]')
        lengths.append(len(ids))
        result['events'] += len(ids)
        result['empty_histories'] += int(not len(ids))
        result['null_timestamp_events'] += int(np.isnat(ts).sum())
        result['unknown_catalog_events'] += sum(int(i) not in article_ids for i in ids)
        valid = ts[~np.isnat(ts)]
        if len(valid):
            lo, hi = valid.min(), valid.max()
            minimum = lo if minimum is None else min(lo, minimum)
            maximum = hi if maximum is None else max(hi, maximum)
            result['unsorted_rows'] += int((valid[1:] < valid[:-1]).any())
            if row.user_id in first:
                result['not_strictly_before_first_request'] += int((valid >= np.datetime64(first[row.user_id], 'us')).sum())
    result.update(time_min=str(minimum), time_max=str(maximum),
                  history_length_quantiles={str(q): float(np.quantile(lengths,q)) for q in [0,.5,.9,.99,1]} if lengths else {})
    return result


def candidate_audit(behaviors, articles, known_candidates=None, known_inputs=None, cutoff=None,
                    history_known=None):
    """Full recorded candidate lists; count cold clicks only on cold candidates."""
    pub = articles.set_index('article_id').published_time.to_dict()
    modified = articles.set_index('article_id').last_modified_time.to_dict()
    catalog = set(pub)
    counters = {k: 0 for k in ['requests', 'candidate_rows', 'clicked_candidate_rows',
        'empty_candidate_requests', 'duplicate_candidate_requests', 'duplicate_clicked_requests',
        'clicks_outside_candidates_requests', 'no_click_requests', 'multi_click_requests',
        'informative_requests', 'unknown_catalog_candidates', 'future_published_candidates',
        'missing_published_candidates', 'modified_after_request_candidates',
        'missing_modified_candidates', 'snapshot_available_candidates',
        'snapshot_available_clicked_candidates', 'requests_over_64_candidates']}
    informative_users, all_items, sizes, clicked_requests_users = set(), set(), [], set()
    cohorts = {k: {'candidate_rows': 0, 'clicked_candidate_rows': 0, 'requests': 0,
                   'clicked_requests': 0, 'all_candidates_cold_requests': 0,
                   'all_candidates_cold_clicked_requests': 0,
                   'snapshot_available_candidates': 0, 'snapshot_available_clicked_candidates': 0,
                   '_items': set(), '_users': set(), '_clicked_users': set()}
               for k in ['unseen_candidates', 'unseen_training_inputs', 'unseen_inputs_and_history', 'published_since_cutoff']}
    for row in behaviors.itertuples(index=False):
        items = list(map(int, row.article_ids_inview)); clicked = set(map(int, row.article_ids_clicked))
        unique = set(items); n = len(items); nclick = len(unique & clicked)
        counters['requests'] += 1; counters['candidate_rows'] += n
        counters['clicked_candidate_rows'] += nclick
        counters['empty_candidate_requests'] += int(n == 0)
        counters['duplicate_candidate_requests'] += int(n != len(unique))
        counters['duplicate_clicked_requests'] += int(len(clicked) != len(row.article_ids_clicked))
        counters['clicks_outside_candidates_requests'] += int(bool(clicked - unique))
        counters['no_click_requests'] += int(nclick == 0)
        counters['multi_click_requests'] += int(nclick > 1)
        counters['requests_over_64_candidates'] += int(n > 64)
        if 0 < nclick < len(unique):
            counters['informative_requests'] += 1; informative_users.add(row.user_id)
        if nclick: clicked_requests_users.add(row.user_id)
        sizes.append(n); all_items.update(items)
        safe = set()
        for i in items:
            if i not in catalog:
                counters['unknown_catalog_candidates'] += 1; continue
            p, m = pub[i], modified[i]
            counters['missing_published_candidates'] += int(pd.isna(p))
            counters['missing_modified_candidates'] += int(pd.isna(m))
            counters['future_published_candidates'] += int(pd.notna(p) and p > row.impression_time)
            counters['modified_after_request_candidates'] += int(pd.notna(m) and m > row.impression_time)
            if pd.notna(p) and pd.notna(m) and p <= row.impression_time and m <= row.impression_time:
                safe.add(i)
        counters['snapshot_available_candidates'] += len(safe)
        counters['snapshot_available_clicked_candidates'] += len(safe & clicked)
        if known_inputs is None:
            continue
        flags = {'unseen_candidates': unique - known_candidates,
                 'unseen_training_inputs': unique - known_inputs,
                 'unseen_inputs_and_history': unique - known_inputs - history_known,
                 'published_since_cutoff': {i for i in unique if i in pub and pd.notna(pub[i]) and cutoff <= pub[i] <= row.impression_time}}
        for name, cold in flags.items():
            c = cohorts[name]; cc = cold & clicked
            c['candidate_rows'] += len(cold); c['clicked_candidate_rows'] += len(cc)
            c['requests'] += int(bool(cold)); c['clicked_requests'] += int(bool(cc))
            c['all_candidates_cold_requests'] += int(bool(unique) and cold == unique)
            c['all_candidates_cold_clicked_requests'] += int(bool(unique) and cold == unique and bool(cc))
            c['snapshot_available_candidates'] += len(cold & safe)
            c['snapshot_available_clicked_candidates'] += len(cc & safe)
            c['_items'].update(cold)
            if cold: c['_users'].add(row.user_id)
            if cc: c['_clicked_users'].add(row.user_id)
    counters.update(users=int(behaviors.user_id.nunique()), items=len(all_items),
                    informative_users=len(informative_users), clicked_users=len(clicked_requests_users),
                    duplicate_impression_ids=int(behaviors.impression_id.duplicated().sum()),
                    time_min=str(behaviors.impression_time.min()), time_max=str(behaviors.impression_time.max()),
                    candidate_count_quantiles={str(q): float(np.quantile(sizes,q)) for q in [0,.5,.9,.99,1]})
    if known_inputs is not None:
        for c in cohorts.values():
            for field in ['items','users','clicked_users']:
                c[field] = len(c.pop('_'+field))
        counters['cold_cohorts'] = cohorts
    return counters


def run():
    from scripts.download_ebnerd import digest
    articles = pd.read_parquet(RAW/'articles.parquet', columns=['article_id','published_time','last_modified_time'])
    b = {s: pd.read_parquet(RAW/s/'behaviors.parquet', columns=BEHAVIOR) for s in ['train','validation']}
    h = {s: pd.read_parquet(RAW/s/'history.parquet') for s in b}
    catalog = set(articles.article_id)
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'actual_parquet_audit',
        'no_model_trained': True, 'official_hidden_test_downloaded': False,
        'timezone_note': 'Parquet timestamps have no timezone; preserve source clock, do not assume UTC.',
        'files': {}, 'articles': {'rows': len(articles), 'duplicate_ids': int(articles.article_id.duplicated().sum())},
        'official_splits': {}, 'histories': {}, 'temporal_splits': {}}
    for path in sorted(RAW.rglob('*.parquet')):
        table = pq.read_table(path)
        result['files'][str(path.relative_to(RAW))] = {'rows': len(table), 'sha256': digest(path),
            'schema': {f.name: str(f.type) for f in table.schema},
            'nulls': {c: table[c].null_count for c in table.column_names}}
        del table
    for s in b:
        result['official_splits'][s] = candidate_audit(b[s], articles)
        result['histories'][s] = history_audit(h[s],b[s],catalog)
    result['official_request_id_overlap'] = len(set(b['train'].impression_id)&set(b['validation'].impression_id))
    result['official_user_overlap'] = len(set(b['train'].user_id)&set(b['validation'].user_id))
    result['official_time_strictly_ordered'] = bool(b['train'].impression_time.max() < b['validation'].impression_time.min())
    frame = pd.concat([b[s].assign(source=s) for s in b], ignore_index=True)
    frame['split'] = partition(frame.impression_time)
    train_history_ids = ids_in(h['train'],'article_id_fixed')
    fit = frame.loc[frame.split.eq('train')]
    fit_candidates = ids_in(fit,'article_ids_inview')
    fit_inputs = fit_candidates | ids_in(fit,'article_ids_clicked') | train_history_ids
    refit_candidates = ids_in(b['train'],'article_ids_inview')
    refit_inputs = refit_candidates | ids_in(b['train'],'article_ids_clicked') | train_history_ids
    result['split_policy'] = {'fit_end_exclusive': str(FIT_END), 'test_start_inclusive': str(TEST_START),
        'validation': 'Last two days of official train', 'test': 'Official labeled validation, repurposed local holdout',
        'refit': 'Select on inner validation, then fit official train; holdout has no model results yet',
        'fit_input_items': len(fit_inputs), 'refit_input_items': len(refit_inputs),
        'unseen_training_inputs': 'Absent from fit candidate/click IDs AND histories used by that fit; article catalog membership does not make an item warm.'}
    for s in ['train','validation','test']:
        part = frame.loc[frame.split.eq(s)]
        if s == 'train':
            result['temporal_splits'][s] = candidate_audit(part, articles)
        else:
            is_test = s == 'test'
            result['temporal_splits'][s] = candidate_audit(part, articles,
                refit_candidates if is_test else fit_candidates, refit_inputs if is_test else fit_inputs,
                TEST_START if is_test else FIT_END,
                ids_in(h['validation' if is_test else 'train'],'article_id_fixed'))
    processed = ROOT/'data/processed/ebnerd-audit-v1'
    processed.mkdir(parents=True,exist_ok=True)
    assignment = processed/'request_splits.parquet'
    frame[['impression_id','user_id','impression_time','source','split']].to_parquet(assignment,index=False)
    result['assignment_sha256'] = digest(assignment)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({s:{k:v for k,v in x.items() if k in ['requests','users','candidate_rows','clicked_candidate_rows','time_min','time_max']} for s,x in result['temporal_splits'].items()},indent=2))
    return result
