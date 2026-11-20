"""Download/extract the official small bundle AFTER the user's license registration.

The URL was supplied by the official form's successful submission page.
No third-party mirror, hidden test set or embedding bundle is downloaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip'
RAW = ROOT / 'data/raw/ebnerd-small'
MANIFEST = ROOT / 'data/source/ebnerd/manifest.json'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def extract(archive, destination):
    """Validate every path before extracting; never overwrite differing files."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        for member in members:
            name = PurePosixPath(member.filename)
            target = (destination / member.filename).resolve()
            if (name.is_absolute() or '..' in name.parts or '\\' in member.filename
                    or not target.is_relative_to(destination)
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Unsafe ZIP member')
        if sum(m.file_size for m in members) > 5_000_000_000:
            raise ValueError('Unexpectedly large small bundle')
        if z.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
        for member in members:
            target = destination / member.filename
            if '__MACOSX' in PurePosixPath(member.filename).parts:
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                with z.open(member) as source:
                    expected = hashlib.file_digest(source, 'sha256').hexdigest()
                if digest(target) != expected:
                    raise ValueError(f'Existing file differs: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered', action='store_true',
                        help='Official license registration has already been completed')
    args = parser.parse_args()
    if not args.registered:
        parser.error('Complete the official form at https://recsys.eb.dk/ first')
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / 'ebnerd_small.zip'
    if not archive.exists():
        part = archive.with_suffix('.zip.part')
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--retry', '3', '--connect-timeout', '20', '--max-time', '900',
                        URL, '-o', str(part)], check=True)
        with zipfile.ZipFile(part) as z:
            if z.testzip() is not None:
                raise ValueError('ZIP CRC verification failed')
        part.replace(archive)
    archive_hash = digest(archive)
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text())['sha256'] != archive_hash:
            raise ValueError('Archive differs from previously recorded download')
    extract(archive, RAW / 'extracted')
    files = []
    for path in sorted((RAW / 'extracted').rglob('*.parquet')):
        files.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                      'sha256': digest(path)})
    manifest = {'dataset': 'EB-NeRD small', 'source_url': URL,
                'source_page': 'https://recsys.eb.dk/',
                'license_url': 'https://recsys.eb.dk/assets/pdf/general_license_terms.pdf',
                'registration_completed': True, 'license_scope': 'non-commercial research',
                'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                'archive': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size,
                'sha256': archive_hash, 'checksum_note': 'Locally computed; not a publisher-supplied checksum',
                'zip_crc_verified': True, 'files': files}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'bytes': manifest['bytes'], 'sha256': archive_hash, 'parquet_files': len(files)}))


if __name__ == '__main__':
    main()
