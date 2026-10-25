"""Download verified public data; extract only selected remote ZIP members."""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
CITIES = {'Zhengzhou': 73, 'Kaifeng': 410, 'Xinxiang': 536}
YEARS = (2020, 2021)


def get(url, **kwargs):
    """Fail clearly on non-success responses; retry transient server failures."""
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(15, 120), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class RangeReader(io.RawIOBase):
    """Seekable HTTP reader that rejects servers ignoring byte ranges."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        response = get(url, headers={'Range': 'bytes=-65536'})
        if response.status_code != 206:
            raise ValueError('This server does not support selective ZIP download.')
        self.size = int(response.headers['Content-Range'].split('/')[-1])
        self.tail = response.content
        self.tail_start = self.size - len(self.tail)
        self.transferred = len(self.tail)

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def read(self, size=-1):
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        if size <= 0:
            return b''
        start, end = self.pos, self.pos + size - 1
        if start >= self.tail_start:
            body = self.tail[start - self.tail_start:end - self.tail_start + 1]
        else:
            r = get(self.url, headers={'Range': f'bytes={start}-{end}'})
            expected = f'bytes {start}-{end}/{self.size}'
            if r.status_code != 206 or r.headers.get('Content-Range') != expected or len(r.content) != size:
                raise ValueError('Invalid byte-range response.')
            body = r.content
            self.transferred += len(body)
        self.pos += len(body)
        return body


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(number):
    path = RAW / f'zenodo_{number}.json'
    if not path.exists():
        r = get(f'https://zenodo.org/api/records/{number}')
        path.write_text(json.dumps(r.json(), indent=2))
    return json.loads(path.read_text())


def download_city(item):
    city, rank = item
    first = ((rank - 1) // 10) * 10 + 1
    archive_name = f'{first:03d}-{first + 9:03d}.zip'
    metadata = record(17079409)
    entry = next(f for f in metadata['files'] if f['key'] == archive_name)
    url = entry['links']['self']
    existing = sorted((RAW / 'ntl').glob(f'*_{city}_*.tif'))
    if len(existing) == 4:
        return {'city': city, 'url': url, 'status': 'cached', 'files': [str(p.relative_to(ROOT)) for p in existing]}
    reader = RangeReader(url)
    paths = []
    with zipfile.ZipFile(reader) as archive:
        wanted = [i for i in archive.infolist()
                  if re.search(rf'_{city}_(2020|2021)(_Flag)?\.tif$', i.filename)]
        if len(wanted) != 4:
            raise ValueError(f'Expected four light/flag files for {city}; found {len(wanted)}.')
        for info in wanted:
            path = RAW / 'ntl' / Path(info.filename).name
            if not path.exists():
                body = archive.read(info)  # ZIP CRC is checked by the standard library.
                if len(body) != info.file_size:
                    raise ValueError('ZIP member size mismatch.')
                path.write_bytes(body)
            paths.append(path)
    return {'city': city, 'url': url, 'http_status': 206, 'archive_bytes': reader.size,
            'transferred_bytes': reader.transferred, 'members': [
                {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in paths]}


def download_ntl():
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / 'ntl').mkdir(exist_ok=True)
    metadata = record(17079409)
    for entry in metadata['files']:
        if entry['key'].endswith('.xlsx') or entry['key'].endswith('.csv'):
            path = RAW / entry['key']
            if not path.exists():
                path.write_bytes(get(entry['links']['self']).content)
    with ThreadPoolExecutor(max_workers=3) as pool:
        checks = list(pool.map(download_city, CITIES.items()))
    path = RAW / 'ntl_download_manifest.json'
    if not path.exists() or any(c.get('status') != 'cached' for c in checks):
        path.write_text(json.dumps({'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                                   'license': metadata['metadata'].get('license'), 'downloads': checks}, indent=2))
    return checks


def download_radar():
    metadata = record(7198950)
    entry = metadata['files'][0]
    path = RAW / entry['key']
    if not path.exists():
        partial = path.with_suffix('.part')
        response = get(entry['links']['self'], stream=True)
        with partial.open('wb') as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
        if partial.stat().st_size != entry['size']:
            raise ValueError('Radar archive size mismatch.')
        actual = hashlib.md5(partial.read_bytes()).hexdigest()
        if 'md5:' + actual != entry['checksum']:
            raise ValueError('Radar archive checksum mismatch.')
        partial.rename(path)
        (RAW / 'radar_download_manifest.json').write_text(json.dumps({
            'url': entry['links']['self'], 'http_status': response.status_code,
            'bytes': entry['size'], 'md5': actual, 'license': metadata['metadata'].get('license'),
            'checked_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    return path


def download_context():
    """Keep small local population windows and intersecting historical counties."""
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    out = RAW / 'population'
    out.mkdir(exist_ok=True)
    if all((out / f'{c}_2020.tif').exists() for c in CITIES) and (RAW / 'study_counties.geojson').exists():
        return
    url = 'https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/CHN/chn_ppp_2020_1km_Aggregated.tif'
    pop = RAW / 'chn_ppp_2020_1km_Aggregated.tif'
    if not pop.exists():
        pop.write_bytes(get(url).content)
    meta = RAW / 'boundaries_metadata.json'
    if not meta.exists():
        meta.write_text(json.dumps(get('https://www.geoboundaries.org/api/current/gbOpen/CHN/ADM2/').json(), indent=2))
    metadata = json.loads(meta.read_text())
    path = RAW / 'china_counties.geojson'
    if not path.exists() or path.read_bytes().startswith(b'version https://git-lfs'):
        boundary_url = metadata['simplifiedGeometryGeoJSON'].replace(
            'https://github.com/wmgeolab/geoBoundaries/raw/',
            'https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/')
        response = get(boundary_url)
        response.json()  # Reject HTML and Git LFS pointers before writing.
        path.write_bytes(response.content)
    features = json.loads(path.read_text())['features']
    chosen = {}
    for city in CITIES:
        ntl = next((RAW / 'ntl').glob(f'*_{city}_2021.tif'))
        with rasterio.open(ntl) as d:
            bounds = d.bounds
        with rasterio.open(pop) as d:
            # Include one neighbouring source pixel for interpolation at the boundary.
            w = from_bounds(bounds.left - 0.01, bounds.bottom - 0.01,
                            bounds.right + 0.01, bounds.top + 0.01, d.transform).round_offsets().round_lengths()
            profile = d.profile.copy()
            profile.update(width=int(w.width), height=int(w.height), transform=d.window_transform(w), compress='deflate')
            with rasterio.open(out / f'{city}_2020.tif', 'w', **profile) as target:
                target.write(d.read(window=w))
        for feature in features:
            geometry = feature['geometry']
            rings = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
            xy = np.array([point for polygon in rings for ring in polygon for point in ring])
            x0, y0 = xy.min(axis=0); x1, y1 = xy.max(axis=0)
            if x1 >= bounds.left and x0 <= bounds.right and y1 >= bounds.bottom and y0 <= bounds.top:
                chosen[feature['properties']['shapeID']] = feature
    (RAW / 'study_counties.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': list(chosen.values())}))
    (RAW / 'context_download_manifest.json').write_text(json.dumps({
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'population_url': url, 'population_bytes': pop.stat().st_size, 'population_sha256': sha256(pop),
        'boundary_url': metadata['simplifiedGeometryGeoJSON'], 'boundary_bytes': path.stat().st_size,
        'boundary_sha256': sha256(path), 'boundary_year': metadata['boundaryYearRepresented'],
        'county_features_retained': len(chosen)}, indent=2))


if __name__ == '__main__':
    print(download_ntl())
    download_context()
    print(download_radar())
