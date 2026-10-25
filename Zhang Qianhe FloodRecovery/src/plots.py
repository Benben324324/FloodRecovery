"""Figures shared by the notebook and research report."""
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

from .download import ROOT, RAW, CITIES
from .lights import PERIODS, trailing_median

FIGURES = ROOT / 'outputs' / 'figures'
COLORS = {'Zhengzhou': '#176b87', 'Kaifeng': '#b06b32', 'Xinxiang': '#7a5287'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 110, 'savefig.dpi': 200})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches='tight', facecolor='white')
    return fig


def outlines(ax, pair, color='white'):
    features = json.loads((RAW / 'study_counties.geojson').read_text())['features']
    for feature in features:
        g = feature['geometry']
        polygons = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        for polygon in polygons:
            for ring in polygon:
                xy = np.array(ring)
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=.45, alpha=.7)
    b = pair['bounds']; ax.set_xlim(b.left, b.right); ax.set_ylim(b.bottom, b.top)


def extent(pair):
    b = pair['bounds']
    return [b.left, b.right, b.bottom, b.top]


def plot_context(pairs):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), layout='constrained')
    for j, city in enumerate(CITIES):
        p = pairs[city, 2021]
        a = axes[0, j]
        im = a.imshow(np.log1p(p['pixel_baseline']), extent=extent(p), vmin=0, vmax=np.log1p(80), cmap='magma')
        a.contour(p['mask'], levels=[.5], extent=extent(p), origin='upper', colors=['cyan'], linewidths=.35)
        outlines(a, p)
        a.set_title(f'{city}: pre-event light'); a.set_xlabel('Longitude'); a.set_ylabel('Latitude')
        a = axes[1, j]
        im2 = a.imshow(np.log1p(p['weights']), extent=extent(p), vmin=0, vmax=np.log1p(30000), cmap='YlGnBu')
        outlines(a, p, '#444444'); a.set_title('2020 population context')
        a.set_xlabel('Longitude'); a.set_ylabel('Latitude')
    fig.colorbar(im, ax=axes[0, :], shrink=.9, label='log(1 + radiance), nW cm$^{-2}$ sr$^{-1}$')
    fig.colorbar(im2, ax=axes[1, :], shrink=.9, label='log(1 + resampled source-cell count)')
    return save(fig, '01_context.png')


def plot_coverage(series, summary):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.4), height_ratios=[1, 1.2], layout='constrained')
    dates = pd.date_range('2021-06-01', '2021-09-30')
    values = np.vstack([series[c, 2021]['coverage'].reindex(dates).values for c in CITIES])
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#e5e7eb')
    im = axes[0].imshow(values, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                        extent=[mdates.date2num(dates[0]), mdates.date2num(dates[-1])+1, 2.5, -.5])
    axes[0].set_yticks(range(3), CITIES); axes[0].xaxis_date()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[0].axvline(pd.Timestamp('2021-07-20'), color='#d93b40', ls='--', lw=1)
    axes[0].set_title('Observed bright-pixel coverage; grey columns have no source band')
    fig.colorbar(im, ax=axes[0], label='Fraction', pad=.01)
    periods = list(PERIODS)
    x = np.arange(len(periods))
    for j, city in enumerate(CITIES):
        part = summary[summary.city == city].set_index('period').loc[periods]
        bars = axes[1].bar(x + (j-1)*.24, part.accepted_dates, width=.24, color=COLORS[city], label=city)
        for bar, n in zip(bars, part.accepted_dates):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.15, str(n), ha='center', fontsize=8)
    axes[1].set_xticks(x, ['Jun 1-Jul 16', 'Jul 17-31', 'Aug 1-15', 'Aug 16-31', 'September'])
    axes[1].set_ylabel('Accepted dates'); axes[1].set_ylim(0, 18); axes[1].legend(ncol=3, frameon=False)
    return save(fig, '02_coverage.png')


def plot_trajectories(series):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 6.8), sharex=True, layout='constrained')
    for ax, city in zip(axes, CITIES):
        df = series[city, 2021]
        ax.axvspan(pd.Timestamp('2021-07-17'), pd.Timestamp('2021-07-31'), color='#f5b9bb', alpha=.35)
        ax.axhline(1, color='#777777', ls=':', lw=1)
        ax.plot(df.index, df['index'], 'o', ms=3.5, alpha=.7, color=COLORS[city], label='Accepted dates')
        sm = trailing_median(df['index'])
        ax.plot(sm.index, sm, color=COLORS[city], lw=1.8, label='7-day trailing median')
        wp = trailing_median(df['population_index'])
        ax.plot(wp.index, wp, color='#252525', ls='--', lw=1.2, label='Population weighted')
        ax.set_ylabel('Light index'); ax.set_title(city, loc='left'); ax.set_ylim(.35, 1.5)
    axes[0].legend(ncol=3, loc='upper right', frameon=False, fontsize=8)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    return save(fig, '03_trajectories.png')


def plot_counties(pairs, table, maps):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), layout='constrained')
    norm = TwoSlopeNorm(vmin=.6, vcenter=1, vmax=1.15)
    for ax, city in zip(axes, CITIES):
        p = pairs[city, 2021]
        im = ax.imshow(maps[city], extent=extent(p), cmap='RdBu', norm=norm)
        outlines(ax, p, '#444444'); ax.set_title(city); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    fig.colorbar(im, ax=axes, label='September / pre-event median', shrink=.8)
    return save(fig, '04_spatial.png')


def plot_diagnostics(series, sensitivity):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), layout='constrained')
    x = np.arange(3)
    for j, city in enumerate(CITIES):
        part = sensitivity[sensitivity.city == city]
        axes[0].plot(part.coverage_threshold, part.july_median, 'o-', color=COLORS[city], label=city)
        for k, year in enumerate((2020, 2021)):
            df = series[city, year]
            value = df.loc[f'{year}-07-17':f'{year}-07-31', 'index'].median()
            axes[1].bar(j+(k-.5)*.32, value, width=.32, color=COLORS[city], alpha=.4 if year==2020 else 1,
                        hatch='//' if year==2020 else '', label=str(year) if j==0 else None)
    axes[0].set_xticks([.5, .7, .9]); axes[0].set_xlabel('Minimum coverage'); axes[0].set_ylabel('July 17-31 median index')
    axes[0].legend(frameon=False, fontsize=9); axes[0].set_title('Coverage sensitivity')
    axes[1].set_xticks(x, CITIES); axes[1].set_ylabel('July 17-31 median index'); axes[1].set_title('Same-calendar 2020 diagnostic')
    axes[1].legend(frameon=False); axes[1].axhline(1, color='#777777', ls=':')
    return save(fig, '05_diagnostics.png')


def plot_radar_gallery(splits, predictors):
    # Fixed test examples span three water fractions; selection does not use prediction scores.
    ordered = sorted(splits['test'], key=lambda s: s[2].mean())
    samples = [ordered[int(q*(len(ordered)-1))] for q in (.25, .5, .75)]
    fig, axes = plt.subplots(3, 4, figsize=(9.8, 6.8), layout='constrained')
    for i, (ident, image, label) in enumerate(samples):
        arrays = [image, label, predictors['Tuned intensity'](image), predictors['Local logistic'](image)]
        for j, (ax, a) in enumerate(zip(axes[i], arrays)):
            ax.imshow(a, cmap='gray' if j==0 else 'Blues', vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
            if i==0: ax.set_title(['SAR intensity', 'Water label', 'Tuned intensity', 'Local logistic'][j])
            if j==0: ax.set_ylabel(f'Tile {ident}\nwater {label.mean():.0%}')
    return save(fig, '06_radar_examples.png')


def plot_radar_scores(table):
    fig, ax = plt.subplots(figsize=(8.5, 2.7), layout='constrained')
    x = np.arange(len(table))
    ax.bar(x, table.iou, color=['#bababa', '#4a91ab', '#6c7199'])
    ax.errorbar(x, table.iou, yerr=[table.iou-table.iou_ci_low, table.iou_ci_high-table.iou], fmt='none', color='#333333', capsize=4)
    ax.set_xticks(x, table.model); ax.set_ylim(0, 1); ax.set_ylabel('Water intersection over union')
    ax.set_title('Test-patch IoU with 95% patch bootstrap intervals')
    return save(fig, '07_radar_scores.png')
