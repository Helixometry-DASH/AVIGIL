"""Source-only manifests, original-family holdouts and test access guards."""
import csv
import hashlib
import json
from pathlib import Path
from .io import atomic_json, sha256


ALLOWED = {'audio_source': {'Real', 'A-only'}, 'visual_source': {'Real', 'V-only'},
           'mixed_source': {'Real', 'A-only', 'V-only', 'AV'}}


def relative(path):
    parts = str(path).replace('\\', '/').split('/')
    if '..' in parts:
        raise ValueError('Parent traversal in media path')
    for i, value in enumerate(parts):
        if value in ('train', 'dev', 'val', 'test'):
            if i + 1 == len(parts) or not parts[-1]:
                raise ValueError('Missing media filename')
            return '/'.join(parts[i:]).replace('val/', 'dev/', 1)
    raise ValueError('Video path lacks an official split: ' + str(path))


def group(record):
    flags = (bool(record['modify_audio']), bool(record['modify_video']))
    return {(False, False): 'Real', (True, False): 'A-only', (False, True): 'V-only', (True, True): 'AV'}[flags]


def family(record):
    return Path(str(record.get('original') or record['file'])).stem


def partition(key):
    number = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
    return 'calibration' if number == 0 else 'reference' if number == 1 else 'fit'


def selected_rows(config, source, split, *, target=None, final_evaluation=False):
    if source not in ALLOWED:
        raise ValueError('Training source must be audio_source, visual_source or mixed_source')
    if split == 'test' and not final_evaluation:
        raise ValueError('Training/preparation cannot open final test manifests')
    if split == 'test':
        if source == 'mixed_source':
            if target != 'MIXED':
                raise ValueError('Mixed-source evaluation requires target MIXED')
            manifest = Path(config['legacy_manifests']) / 'test_mixed.csv'
            permitted = ALLOWED['mixed_source']
        else:
            if target not in ('A', 'V'):
                raise ValueError('Strict-source final evaluation needs an explicit A/V target')
            manifest = Path(config['legacy_manifests']) / ('test_' + target + '_only.csv')
            permitted = ALLOWED['audio_source' if target == 'A' else 'visual_source']
    else:
        if split not in ('train', 'dev'):
            raise ValueError('Unsupported split')
        manifest = Path(config['legacy_manifests']) / (split + '_' + source + '.csv')
        permitted = ALLOWED[source]
    with manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    requested = {relative(row['video_path']) for row in rows}
    records = json.loads(Path(config['metadata']).read_text())
    metadata = {relative(r['file']): r for r in records if relative(r['file']) in requested}
    result, seen = [], set()
    for row in rows:
        key = relative(row['video_path'])
        record = metadata[key]
        if key in seen or row['split'] != split or row['group'] not in permitted:
            raise ValueError('Duplicate, wrong-split or forbidden-group record: ' + key)
        if record['split'].replace('val', 'dev') != split or group(record) != row['group']:
            raise ValueError('Metadata disagrees with source manifest')
        if (int(row['modify_audio']), int(row['modify_video'])) != (int(record['modify_audio']), int(record['modify_video'])):
            raise ValueError('Manipulation flag mismatch')
        seen.add(key)
        result.append({**record, 'file': key, 'group': group(record), 'family': family(record),
                       'partition': partition(family(record)), 'avhubert_path': row['features_path'],
                       'video_path': row['video_path']})
    if not result:
        raise ValueError('No permitted examples')
    return result


def source_training_rows(rows):
    return [r for r in rows if r['partition'] != 'calibration']


def genuine_rows(rows, part='fit'):
    return [r for r in rows if r['group'] == 'Real' and (part is None or r['partition'] == part)]


def view_periods(row, modality='joint'):
    if modality == 'audio' and not row['modify_audio']:
        return []
    if modality == 'video' and not row['modify_video']:
        return []
    return [(float(s), float(e)) for s, e in row['fake_periods']]


def rebase_periods(periods, start, duration):
    # Preserve endpoints beyond the crop for boundary-censoring supervision.
    return [(s - start, e - start) for s, e in periods if e > start and s < start + duration]


def prepare_manifests(config):
    output = Path(config['workspace']) / 'manifests'
    summary = {}
    for source in config.get('experiment_sources', ('audio_source', 'visual_source')):
        train, dev = selected_rows(config, source, 'train'), selected_rows(config, source, 'dev')
        overlap = {r['family'] for r in train} & {r['family'] for r in dev}
        if overlap:
            raise ValueError('Original-family overlap between train and dev')
        for split, rows in [('train', train), ('dev', dev)]:
            atomic_json(output / (split + '_' + source + '.json'), rows)
        summary[source] = {'train_total': len(train), 'train_fit': len(source_training_rows(train)),
                           'genuine_fit': len(genuine_rows(train)), 'calibration_families': len({r['family'] for r in train if r['partition'] == 'calibration'}),
                           'dev_total': len(dev)}
    atomic_json(output / 'provenance.json', {'metadata_sha256': sha256(config['metadata']), 'summary': summary,
                                           'test_manifests_opened': False})
    return summary


def build_mixed_manifests(config, *, final_evaluation=False):
    """Official split, availability filtering only; no outcome-dependent selection.

    Final test CSV creation is deferred until explicit final evaluation. Existing
    source-only CSVs remain unchanged, allowing reference checkpoint reuse.
    """
    from collections import Counter
    records = json.loads(Path(config['metadata']).read_text())
    directory = Path(config['legacy_manifests'])
    directory.mkdir(parents=True, exist_ok=True)
    summary = {'availability_rule': 'raw video and existing AV-HuBERT bank must exist',
               'test_rows_selected': final_evaluation, 'splits': {}}
    keys_by_split = {}
    for split in (('test',) if final_evaluation else ('train', 'dev')):
        included, excluded, rows, families = Counter(), Counter(), [], set()
        for record in records:
            if record['split'].replace('val', 'dev') != split:
                continue
            key = relative(record['file'])
            raw = Path(config['data_root']) / key
            speech = Path(config['data_root']) / 'LAV-DF_emb_avhubert' / Path(key).with_suffix('') / 'features.npz'
            label = group(record)
            if not raw.is_file() or not speech.is_file():
                excluded[label] += 1
                continue
            included[label] += 1
            families.add(family(record))
            rows.append({'split': split, 'group': label, 'video_path': str(raw), 'features_path': str(speech),
                         'modify_audio': int(record['modify_audio']), 'modify_video': int(record['modify_video'])})
        if set(included) != ALLOWED['mixed_source']:
            raise ValueError('Mixed split must contain all four manipulation groups')
        output = directory / ('test_mixed.csv' if split == 'test' else split + '_mixed_source.csv')
        temporary = output.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=('split', 'group', 'video_path', 'features_path', 'modify_audio', 'modify_video'))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
        summary['splits'][split] = {'included': dict(included), 'excluded_missing_input': dict(excluded),
                                   'total': len(rows), 'manifest_sha256': sha256(output)}
        keys_by_split[split] = families
        if split != 'test' and config.get('reference_workspace'):
            mixed_real = {relative(r['video_path']) for r in rows if r['group'] == 'Real'}
            for source in ('audio_source', 'visual_source'):
                previous = selected_rows(config, source, split)
                if mixed_real != {r['file'] for r in previous if r['group'] == 'Real'}:
                    raise ValueError('Mixed genuine coverage differs from inherited reference protocol')
    if not final_evaluation and keys_by_split['train'] & keys_by_split['dev']:
        raise ValueError('Mixed train/dev original-family leakage')
    atomic_json(Path(config['workspace']) / ('mixed_test_inventory.json' if final_evaluation else 'mixed_data_inventory.json'), summary)
    return summary
