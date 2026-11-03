#!/usr/bin/env python3
"""Prepare SKIPP'D for image-based, 15-minute-ahead solar forecasting.

Dependencies: numpy, pandas, h5py. PyTorch is optional (for the DataLoader).

Example:
  python preprocess_skippd.py --data-dir data --inspect
  python preprocess_skippd.py --data-dir data --trusted-timestamps --max-days 60

Expects ONE benchmark HDF5 file, plus times_trainval.npy and times_test.npy.
Use --h5 if the directory contains multiple HDF5 files.
Supports individual observations (N,H,W,C) and the authors' prebuilt
forecast samples (N,K,H,W,C), with newest-first histories and pv_pred.
Prebuilt samples are assumed to use the authors' 15-minute forecast horizon.

Output is a SMALL index/statistics NPZ, not another copy of the images.
Keep the source HDF5 in place. PreparedSKIPPD loads and normalizes examples
on demand. The original HDF5 and timestamp files are never modified.

IMPORTANT: This creates NEW chronological day splits across all input groups;
it does not reproduce the authors' selected sunny/cloudy benchmark test split.
All cleaning and temporal checks are automatic, not a substitute for visual QA.

Source: https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_times(path, trusted):
    try:
        values = np.load(path, allow_pickle=False)
    except ValueError as error:
        if not trusted:
            raise ValueError(
                f"{path}: object timestamps require --trusted-timestamps. "
                "Enable this only for timestamp files from a trusted source."
            ) from error
        values = np.load(path, allow_pickle=True)
    values = np.asarray(values).reshape(-1)
    if np.issubdtype(values.dtype, np.number):
        raise ValueError("Numeric timestamps have ambiguous units; convert to datetime64 first.")
    if values.dtype.kind == 'S':
        values = np.char.decode(values, 'utf-8')
    times = pd.DatetimeIndex(pd.to_datetime(values, errors='coerce'))
    # Preserve supplied local wall-clock time; do not invent a timezone.
    if times.tz is not None:
        times = times.tz_localize(None)
    return times.to_numpy(dtype='datetime64[s]')


def describe(path):
    with h5py.File(path, 'r') as source:
        source.visititems(lambda name, obj: print(f"{name}: {obj.shape}, {obj.dtype}")
                          if isinstance(obj, h5py.Dataset) else None)


def prepare(args):
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}. Choose a new --output path.")
    records, schemas = [], {}
    with h5py.File(args.h5, 'r') as source:
        names = [name for name in source if isinstance(source[name], h5py.Group)
                 and 'images_log' in source[name] and 'pv_log' in source[name]]
        if not names:
            raise ValueError('Expected groups containing images_log and pv_log; run --inspect.')
        for name in names:
            group = source[name]
            images = group['images_log']
            n = len(images)
            if images.ndim not in (4, 5) or images.shape[-1] not in (1, 3):
                raise ValueError(f'{name}: expected NHWC or NKHWC images, got {images.shape}.')
            mode = 'frames' if images.ndim == 4 else 'prebuilt'
            k = args.history if mode == 'frames' else images.shape[1]
            if mode == 'prebuilt' and (args.horizon != 15 or 'pv_pred' not in group):
                raise ValueError('Prebuilt input requires pv_pred and the documented 15-minute horizon.')
            if mode == 'prebuilt' and k != args.history:
                raise ValueError(f'{name}: prebuilt history is {k}; use --history {k}.')
            schemas[name] = {'mode': mode, 'shape': list(images.shape), 'dtype': str(images.dtype)}
            times_path = args.data_dir / f'times_{name}.npy'
            times = load_times(times_path, args.trusted_timestamps)
            if len(times) != n:
                raise ValueError(f'{name}: image/timestamp counts differ.')
            pv_ds = group['pv_log']
            if pv_ds.shape[0] != n or int(np.prod(pv_ds.shape[1:])) != (1 if mode == 'frames' else k):
                raise ValueError(f'{name}: unexpected pv_log shape {pv_ds.shape}.')
            pv = np.asarray(pv_ds[:], dtype=np.float64).reshape(n, -1) * args.pv_to_kw
            if mode == 'prebuilt':
                if group['pv_pred'].shape[0] != n or group['pv_pred'].size != n:
                    raise ValueError(f'{name}: pv_pred must have one target per sample.')
                targets = np.asarray(group['pv_pred'][:], dtype=np.float64).reshape(n) * args.pv_to_kw
            else:
                targets = pv[:, 0]
            valid = (~np.isnat(times) & np.isfinite(pv).all(axis=1)
                     & (pv >= 0).all(axis=1) & np.isfinite(targets) & (targets >= 0))
            frame = pd.DataFrame({'time': times, 'group': name, 'row': np.arange(n),
                                  'current_kw': pv[:, 0], 'target_kw': targets})
            frame = frame.loc[valid].copy()
            records.append(frame)
            print(f'{name}: {n:,} records; {n-len(frame):,} invalid PV/timestamp records excluded.')

        if len({v['mode'] for v in schemas.values()}) != 1:
            raise ValueError('Do not mix individual-frame and prebuilt groups in one run.')
        if len({tuple(v['shape'][-3:]) for v in schemas.values()}) != 1:
            raise ValueError('Image dimensions must match across input groups.')
        mode = next(iter(schemas.values()))['mode']
        table = pd.concat(records, ignore_index=True).sort_values('time', kind='stable')
        ambiguous = table['time'].duplicated(keep=False)
        print(f'Excluding {ambiguous.sum():,} records with ambiguous duplicate timestamps.')
        table = table.loc[~ambiguous].copy()
        days = np.sort(table['time'].dt.normalize().unique())
        if args.max_days is not None:
            days = days[:args.max_days]
            table = table.loc[table['time'].dt.normalize().isin(days)].copy()
        if len(days) < 3:
            raise ValueError('Need at least three distinct days for train/validation/test.')
        n_train = max(1, min(len(days)-2, int(len(days)*args.train_fraction)))
        n_val = max(1, min(len(days)-n_train-1, int(len(days)*args.val_fraction)))
        day_split = {pd.Timestamp(day): ('train' if i < n_train else
                     'val' if i < n_train+n_val else 'test') for i, day in enumerate(days)}

        # Inspect selected images in bounded-memory blocks, retaining legitimate
        # identical sky images and valid zero-generation readings.
        image_ok = np.ones(len(table), dtype=bool)
        table = table.reset_index(drop=True)
        for name, subset in table.groupby('group', sort=False):
            source_rows = subset['row'].to_numpy()
            order = np.argsort(source_rows)
            positions = subset.index.to_numpy()[order]
            source_rows = source_rows[order]
            ds = source[name]['images_log']
            for offset in range(0, len(source_rows), args.batch_size):
                rows = source_rows[offset:offset+args.batch_size]
                block = np.asarray(ds[rows])
                axes = tuple(range(1, block.ndim))
                okay = (np.isfinite(block).all(axis=axes)
                        & (block >= 0).all(axis=axes)
                        & (block <= args.image_scale).all(axis=axes))
                image_ok[positions[offset:offset+len(rows)]] = okay
            print(f'{name}: checked {len(source_rows):,} selected image records.', flush=True)
        print(f'Excluding {(~image_ok).sum():,} records with invalid image values.')
        table = table.loc[image_ok].reset_index(drop=True)

    groups, rows_in, rows_target, origins, target_times, targets, splits, persistence = ([] for _ in range(8))
    # Build sequences within one source group/day. Every timestamp from the first
    # input through the future label must be present for individual-frame data.
    for (day, name), frame in table.groupby([table['time'].dt.normalize(), 'group'], sort=True):
        frame = frame.sort_values('time')
        ts = frame['time'].to_numpy(dtype='datetime64[s]')
        rows = frame['row'].to_numpy(dtype=np.int64)
        if mode == 'frames':
            breaks = np.r_[0, np.flatnonzero(np.diff(ts).astype('timedelta64[s]').astype(int) != 60)+1, len(ts)]
            candidates = [(start, end) for start, end in zip(breaks[:-1], breaks[1:])]
            for start, end in candidates:
                for anchor in range(start+args.history-1, end-args.horizon, args.stride):
                    target = anchor+args.horizon
                    groups.append(name)
                    rows_in.append(rows[anchor-args.history+1:anchor+1])
                    rows_target.append(rows[target])
                    origins.append(ts[anchor]); target_times.append(ts[target])
                    targets.append(frame['current_kw'].iloc[target])
                    persistence.append(frame['current_kw'].iloc[anchor])
                    splits.append(day_split[day])
        else:
            # Official samples already contain verified minute-spaced histories.
            # Their anchors alone cannot independently prove frame continuity.
            for anchor in range(0, len(ts), args.stride):
                first = ts[anchor] - np.timedelta64(args.history-1, 'm')
                future = ts[anchor] + np.timedelta64(args.horizon, 'm')
                if first.astype('datetime64[D]') != future.astype('datetime64[D]'):
                    continue
                groups.append(name); rows_in.append(np.full(args.history, rows[anchor]))
                rows_target.append(rows[anchor]); origins.append(ts[anchor]); target_times.append(future)
                targets.append(frame['target_kw'].iloc[anchor])
                persistence.append(frame['current_kw'].iloc[anchor]); splits.append(day_split[day])
    if not groups:
        raise ValueError('No valid sequences. Check timestamps, history, horizon, and image scale.')
    split = np.asarray(splits)
    counts = {s: int(np.sum(split == s)) for s in ('train', 'val', 'test')}
    if not all(counts.values()):
        raise ValueError(f'One split has no valid sequences: {counts}. Use more days.')
    # Scale using ONLY training labels. Histories and labels use the same units.
    y_kw = np.asarray(targets, dtype=np.float64)
    mean = float(y_kw[split == 'train'].mean())
    std = float(y_kw[split == 'train'].std())
    if std < 1e-8:
        std = 1.0
    metadata = {'source_h5': str(args.h5.resolve()), 'mode': mode,
                'history': args.history, 'horizon_minutes': args.horizon,
                'stride_minutes_for_frames': args.stride, 'prebuilt_row_stride': args.stride,
                'image_scale': args.image_scale, 'pv_to_kw': args.pv_to_kw,
                'pv_mean_kw': mean, 'pv_std_kw': std, 'counts': counts,
                'split_rule': 'new chronological day split; original group labels ignored for splitting',
                'schemas': schemas, 'target': 'minute-averaged power at t+horizon, NOT interval energy',
                'prebuilt_history_order': 'source newest-first; loader reverses to oldest-first'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)),
            group=np.asarray(groups), input_rows=np.asarray(rows_in, dtype=np.int64),
            target_row=np.asarray(rows_target, dtype=np.int64), split=split,
            input_time=np.asarray(origins, dtype='datetime64[s]').astype(np.int64),
            target_time=np.asarray(target_times, dtype='datetime64[s]').astype(np.int64),
            target_kw=y_kw.astype(np.float32),
            persistence_kw=np.asarray(persistence, dtype=np.float32))
    print(json.dumps(metadata, indent=2))
    print(f'Saved {args.output}; keep the source HDF5 file in place.')


def read_rows(dataset, rows):
    order = np.argsort(rows)
    return np.asarray(dataset[rows[order]])[np.argsort(order)]


class PreparedSKIPPD:
    """Lazy NumPy dataset; also works with torch.utils.data.DataLoader.

    images: (3*history, H, W), temporal frames stacked as CNN channels.
    pv_history: (history,), normalized; time_features: (2,), sine/cosine.
    target: normalized future power; target_kw/persistence_kw: physical kW.
    Use DataLoader(..., num_workers=0) initially, especially in VS Code/macOS.
    """
    def __init__(self, index_path, split='train', source_h5=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        with np.load(index_path, allow_pickle=False) as data:
            self.meta = json.loads(str(data['metadata']))
            self.data = {key: data[key] for key in data.files if key != 'metadata'}
        self.indices = np.flatnonzero(self.data['split'] == split)
        self.path = str(source_h5 or self.meta['source_h5'])
        self._file, self._pid = None, None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'], state['_pid'] = None, None
        return state

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getitem__(self, item):
        i = self.indices[item]
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.path, 'r')
            self._pid = os.getpid()
        group = self._file[str(self.data['group'][i])]
        rows = self.data['input_rows'][i]
        if self.meta['mode'] == 'frames':
            images = read_rows(group['images_log'], rows)
            pv = read_rows(group['pv_log'], rows).reshape(-1)
        else:
            images = np.asarray(group['images_log'][rows[0]])[::-1]
            pv = np.asarray(group['pv_log'][rows[0]]).reshape(-1)[::-1]
        images = images.astype(np.float32) / self.meta['image_scale']
        k, height, width, channels = images.shape
        images = images.transpose(0, 3, 1, 2).reshape(k*channels, height, width).copy()
        pv = pv.astype(np.float32) * self.meta['pv_to_kw']
        mean, std = self.meta['pv_mean_kw'], self.meta['pv_std_kw']
        seconds = int(self.data['input_time'][i]) % 86400
        phase = 2*np.pi*seconds/86400
        return {'images': images, 'pv_history': ((pv-mean)/std).astype(np.float32),
                'time_features': np.array([np.sin(phase), np.cos(phase)], dtype=np.float32),
                'target': np.float32((self.data['target_kw'][i]-mean)/std),
                'target_kw': self.data['target_kw'][i],
                'persistence_kw': self.data['persistence_kw'][i]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--h5', type=Path, help='Optional explicit HDF5 path')
    parser.add_argument('--output', type=Path, default=Path('prepared_skippd.npz'))
    parser.add_argument('--inspect', action='store_true', help='Print HDF5 layout only')
    parser.add_argument('--trusted-timestamps', action='store_true', help='Allow trusted pickled datetime .npy files')
    parser.add_argument('--history', type=int, default=16, help='16 frames cover t-15 through t')
    parser.add_argument('--horizon', type=int, default=15, help='Forecast minutes ahead')
    parser.add_argument('--stride', type=int, default=5, help='Frame-anchor minute stride, or prebuilt row stride')
    parser.add_argument('--max-days', type=int, help='Use earliest N days for a small first experiment')
    parser.add_argument('--train-fraction', type=float, default=0.7)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--batch-size', type=int, default=32, help='Image-validation block size')
    parser.add_argument('--image-scale', type=float, choices=[1.0, 255.0], default=255.0,
                        help='255 for official uint8 images; 1 only for already scaled inputs')
    parser.add_argument('--pv-to-kw', type=float, default=1.0, help='1 for kW, .001 for watts')
    args = parser.parse_args()
    if args.h5 is None:
        paths = sorted(set(args.data_dir.glob('*.hdf5')) | set(args.data_dir.glob('*.h5')))
        if len(paths) != 1:
            parser.error('Expected exactly one HDF5 file in --data-dir; otherwise specify --h5.')
        args.h5 = paths[0]
    if args.inspect:
        describe(args.h5)
        return
    if min(args.history, args.horizon, args.stride, args.batch_size) < 1:
        parser.error('history, horizon, stride and batch-size must be positive')
    if args.max_days is not None and args.max_days < 3:
        parser.error('--max-days must be at least 3')
    if not (0 < args.train_fraction < 1 and 0 < args.val_fraction < 1
            and args.train_fraction + args.val_fraction < 1 and args.pv_to_kw > 0):
        parser.error('Invalid split fractions or unit conversion')
    try:
        prepare(args)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
