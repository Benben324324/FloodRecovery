"""Validate radiance/flag pairs and construct coverage-aware recovery indices."""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling

from .download import ROOT, RAW, CITIES

PROCESSED = ROOT / 'data' / 'processed'
TABLES = ROOT / 'outputs' / 'tables'
PERIODS = {'Pre-event': ('2021-06-01', '2021-07-16'),
           'July 17-31': ('2021-07-17', '2021-07-31'),
           'August 1-15': ('2021-08-01', '2021-08-15'),
           'August 16-31': ('2021-08-16', '2021-08-31'),
           'September': ('2021-09-01', '2021-09-30')}


def read_pair(city, year):
    path = next((RAW / 'ntl').glob(f'*_{city}_{year}.tif'))
    flag_path = path.with_name(path.stem + '_Flag.tif')
    with rasterio.open(path) as d, rasterio.open(flag_path) as f:
        if d.count != f.count or d.crs != f.crs:
            raise ValueError('Light/flag band count or CRS mismatch.')
        if str(d.crs) != 'EPSG:4326' or d.descriptions != f.descriptions:
            raise ValueError('Unexpected CRS or band descriptions.')
        if any(not re.fullmatch(r'NTL\d{4}_\d{2}_\d{2}', s or '') for s in d.descriptions):
            raise ValueError('Invalid date band name.')
        dates = pd.DatetimeIndex([pd.Timestamp(s[3:].replace('_', '-')) for s in d.descriptions])
        if dates.has_duplicates or not dates.is_monotonic_increasing or not (dates.year == year).all():
            raise ValueError('Invalid chronological band sequence.')
        indices = np.flatnonzero((dates.month >= 6) & (dates.month <= 9))
        light = d.read((indices + 1).tolist()).astype(float) * 0.1
        flag_source = f.read((indices + 1).tolist()).astype(np.uint8)
        flags = np.full(light.shape, 255, dtype=np.uint8)
        for t in range(len(indices)):
            reproject(source=flag_source[t], destination=flags[t],
                      src_transform=f.transform, src_crs=f.crs,
                      dst_transform=d.transform, dst_crs=d.crs,
                      src_nodata=0, dst_nodata=255, resampling=Resampling.nearest)
        flags[flags == 255] = 0
        known = np.isfinite(flags)
        if not np.isin(flags[known], [0, 10, 11, 20, 21, 30, 31]).all():
            raise ValueError('Unknown quality flag.')
        if np.any(light[np.isfinite(light)] < 0):
            raise ValueError('Negative radiance.')
        profile = d.profile.copy()
        bounds = d.bounds
        # Units digit 0 means no interpolation; positive tens digit means valid reference.
        valid = np.isfinite(light) & known & (flags >= 10)
        observed = valid & ((flags % 10) == 0)
        return {'city': city, 'year': year, 'dates': dates[indices], 'light': light,
                'flags': flags, 'observed': observed, 'valid': valid,
                'profile': profile, 'bounds': bounds, 'total_bands': d.count,
                'flag_shape': f.shape, 'flags_reprojected': d.transform != f.transform or d.shape != f.shape}


def population_weights(pair):
    path = RAW / 'population' / f'{pair["city"]}_2020.tif'
    with rasterio.open(path) as source:
        weights = np.zeros(pair['light'].shape[1:], dtype=np.float32)
        reproject(source=rasterio.band(source, 1), destination=weights,
                  src_transform=source.transform, src_crs=source.crs,
                  dst_transform=pair['profile']['transform'], dst_crs=pair['profile']['crs'],
                  src_nodata=source.nodata, dst_nodata=0, resampling=Resampling.bilinear)
    # Relative weights approximate resident density. Do not sum resampled counts.
    return np.maximum(weights, 0)


def load_inputs():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    pairs, audit = {}, []
    for city in CITIES:
        for year in (2020, 2021):
            p = read_pair(city, year)
            pairs[city, year] = p
            audit.append({'city': city, 'year': year, 'annual_bands': p['total_bands'],
                          'study_bands': len(p['dates']), 'height': p['light'].shape[1],
                          'width': p['light'].shape[2], 'flag_height': p['flag_shape'][0],
                          'flag_width': p['flag_shape'][1], 'flags_reprojected': p['flags_reprojected'],
                          'first_date': str(p['dates'].min().date()),
                          'last_date': str(p['dates'].max().date())})
        current = pairs[city, 2021]
        base = (current['dates'] >= '2021-06-01') & (current['dates'] <= '2021-07-16')
        x = np.where(current['observed'][base], current['light'][base], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pixel_base = np.nanmedian(x, axis=0)
        bright = (np.isfinite(x).sum(axis=0) >= 5) & (pixel_base >= 5)
        if bright.sum() < 20:
            raise ValueError(f'Insufficient bright baseline pixels: {city}.')
        weights = population_weights(current)
        for year in (2020, 2021):
            p = pairs[city, year]
            if p['light'].shape[1:] != bright.shape or p['profile']['transform'] != current['profile']['transform']:
                raise ValueError('Year-to-year grid mismatch.')
            p.update(mask=bright, weights=weights, pixel_baseline=pixel_base)
        audit[-1]['bright_pixels'] = int(bright.sum())
    audit = pd.DataFrame(audit)
    audit.to_csv(TABLES / 'data_audit.csv', index=False)
    return pairs, audit


def region_series(pair, mask=None, coverage=0.7, allow_interpolated=False):
    mask = pair['mask'] if mask is None else mask
    valid = pair['valid'] if allow_interpolated else pair['observed']
    rows = []
    for t, date in enumerate(pair['dates']):
        good = mask & valid[t]
        support = good.sum() / mask.sum()
        weights = pair['weights'][good]
        value = float(np.mean(pair['light'][t][good])) if good.any() else np.nan
        weighted = float(np.average(pair['light'][t][good], weights=weights)) if weights.sum() > 0 else np.nan
        rows.append({'city': pair['city'], 'year': pair['year'], 'date': date,
                     'coverage': support, 'weighted_coverage': pair['weights'][good].sum() / pair['weights'][mask].sum(),
                     'interpolated_fraction': (mask & pair['valid'][t] & ~pair['observed'][t]).sum() / mask.sum(),
                     'radiance': value if support >= coverage else np.nan,
                     'population_radiance': weighted if support >= coverage else np.nan})
    df = pd.DataFrame(rows).set_index('date')
    base = df.loc[f'{pair["year"]}-06-01':f'{pair["year"]}-07-16']
    for column, index in [('radiance', 'index'), ('population_radiance', 'population_index')]:
        denominator = base[column].median()
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Insufficient baseline support.')
        df[index] = df[column] / denominator
    return df


def trailing_median(series, days=7, minimum=3):
    daily = series.reindex(pd.date_range(series.index.min(), series.index.max()))
    return daily.rolling(f'{days}D', min_periods=minimum).median()


def return_date(series, threshold=0.95):
    """Require an observed decline, then three supported smoothed observations."""
    smooth = trailing_median(series)
    trough_window = smooth.loc['2021-07-17':'2021-07-31'].dropna()
    if trough_window.empty:
        return None, 'Insufficient July-window support'
    trough_date = trough_window.idxmin()
    if trough_window.min() >= threshold:
        return None, 'No supported threshold decline'
    observed = series.loc[trough_date:].dropna()
    candidates = smooth.reindex(observed.index)
    for i in range(len(candidates) - 2):
        block = candidates.iloc[i:i+3]
        gaps = block.index.to_series().diff().dropna().dt.days
        if (block >= threshold).all() and gaps.max() <= 3:
            return block.index[0], 'Supported return'
    return None, 'Return not established by September 30'


def analyze(pairs):
    series = {key: region_series(pair) for key, pair in pairs.items()}
    panel = pd.concat(series.values()).reset_index()
    panel.to_csv(PROCESSED / 'daily_light_panel.csv', index=False)
    summaries, recovery, sensitivity = [], [], []
    for city in CITIES:
        df = series[city, 2021]
        for label, (a, b) in PERIODS.items():
            part = df.loc[a:b]
            summaries.append({'city': city, 'period': label, 'available_dates': len(part),
                              'accepted_dates': int(part['index'].notna().sum()),
                              'median_index': part['index'].median(),
                              'population_median_index': part['population_index'].median(),
                              'median_coverage': part['coverage'].median()})
        for cov in (0.5, 0.7, 0.9):
            s = region_series(pairs[city, 2021], coverage=cov)['index']
            for threshold in (0.90, 0.95, 1.00):
                date, status = return_date(s, threshold)
                recovery.append({'city': city, 'coverage_threshold': cov, 'return_threshold': threshold,
                                 'return_date': str(date.date()) if date is not None else '', 'status': status})
            sensitivity.append({'city': city, 'coverage_threshold': cov,
                                'accepted_dates': int(s.notna().sum()),
                                'july_median': s.loc['2021-07-17':'2021-07-31'].median(),
                                'september_median': s.loc['2021-09-01':'2021-09-30'].median()})
    summary = pd.DataFrame(summaries)
    recovery = pd.DataFrame(recovery)
    sensitivity = pd.DataFrame(sensitivity)
    summary.to_csv(TABLES / 'period_summary.csv', index=False)
    recovery.to_csv(TABLES / 'return_sensitivity.csv', index=False)
    sensitivity.to_csv(TABLES / 'coverage_sensitivity.csv', index=False)
    return series, summary, recovery, sensitivity


def county_analysis(pairs):
    features = json.loads((RAW / 'study_counties.geojson').read_text())['features']
    units = []
    maps = {}
    for city in CITIES:
        p = pairs[city, 2021]
        raster = np.full(p['mask'].shape, np.nan)
        i = 0
        for feature in features:
            mask = geometry_mask([feature['geometry']], out_shape=p['mask'].shape,
                                 transform=p['profile']['transform'], invert=True) & p['mask']
            if mask.sum() < 20:
                continue
            i += 1
            df = region_series(p, mask)
            row = {'city': city, 'unit': f'{city[0]}{i:02}',
                   'shape_id': feature['properties']['shapeID'],
                   'source_name': feature['properties']['shapeName'],
                   'bright_pixels': int(mask.sum()), 'relative_population_weight': float(p['weights'][mask].sum())}
            for label, (a, b) in list(PERIODS.items())[1:]:
                row[label] = df.loc[a:b, 'index'].median()
                row[label + '_n'] = int(df.loc[a:b, 'index'].notna().sum())
            raster[mask] = row['September']
            units.append(row)
        maps[city] = raster
    table = pd.DataFrame(units)
    table.to_csv(TABLES / 'county_window_summary.csv', index=False)
    return table, maps
