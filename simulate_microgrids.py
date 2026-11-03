#!/usr/bin/env python3
"""Microgrid pricing/storage simulator and baseline comparison.

Run from the project folder:
  python3 simulate_microgrids.py --synthetic
  python3 simulate_microgrids.py --prepared prepared_skippd.npz --data-dir data

Dependencies: numpy, pandas, h5py (the last two only for SKIPP'D replay).
No CNN or optimal WCMDP solver is implemented here. Baselines observe current
supply, time and batteries; they never receive future demand or generation.
The myopic baseline knows the assumed demand parameters and optimizes an
approximate deterministic one-step objective on discrete price/allocation grids.
It does NOT optimize expected stochastic or multi-step return.

All demand, contract, cost, price and battery settings below are illustrative,
not fitted to real customers. Requested demand is fully observed even when
partly unmet. There is no backlog, network power flow, cross-grid battery
transfer, demand learning, or terminal battery salvage payment in this version.
Battery dispatch is automatic: serve customers first, then charge with surplus.
Each episode starts with half-full batteries and lasts one contiguous daytime
recording segment (not necessarily a whole day). Overnight demand is not modeled.
Use multiple seeds and parameter settings before drawing research conclusions.
"""

import argparse
import csv
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np


@dataclass
class Settings:
    microgrids: int = 3
    dt_minutes: float = 1.0
    reference_price: float = 0.25       # dollars per kWh
    min_price: float = 0.10
    max_price: float = 0.50
    fixed_price: float = 0.25
    contract_kw: float = 12.0
    contract_cost: float = 0.08         # dollars per contracted kWh, take-or-pay
    shortage_penalty: float = 0.80      # dollars per unmet requested kWh
    battery_wear: float = 0.01          # dollars per kWh withdrawn from battery
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    demand_noise: float = 0.15          # relative Gaussian shock before truncation
    allocation_bins: int = 40
    price_points: int = 9

    def __post_init__(self):
        if self.microgrids < 1 or self.dt_minutes <= 0:
            raise ValueError('Need positive grid count and timestep.')
        if not 0 < self.min_price <= self.fixed_price <= self.max_price:
            raise ValueError('Invalid price limits.')
        if self.reference_price <= 0 or self.allocation_bins < 1 or self.price_points < 2:
            raise ValueError('Invalid reference price or discretization.')
        if not (0 < self.charge_efficiency <= 1 and 0 < self.discharge_efficiency <= 1):
            raise ValueError('Efficiencies must be in (0,1].')
        if min(self.contract_kw, self.contract_cost, self.shortage_penalty,
               self.battery_wear, self.demand_noise) < 0:
            raise ValueError('Costs, capacities and noise cannot be negative.')

    @property
    def dt_hours(self):
        return self.dt_minutes / 60


class MicrogridEnv:
    """Local storage transitions coupled only by a shared allocation constraint.

    Actions: one price ($/kWh) and allocated energy (kWh) per microgrid.
    Observe current interval solar power, assumed constant within this minute.
    At dt=1 minute, measured minute-average PV is treated as observed at decision
    time: a simplified perfect-current-measurement assumption, not a forecast.
    step() then reveals that interval's independent local demand shocks.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        m = cfg.microgrids
        self.capacity = np.linspace(6, 10, m)          # kWh
        self.charge_kw = np.linspace(4, 6, m)
        self.discharge_kw = np.linspace(4, 6, m)
        self.base_kw = np.linspace(8, 14, m)
        self.elasticity = np.linspace(0.25, 0.65, m)  # response to relative price
        self.reset()

    def reset(self):
        self.battery = 0.5 * self.capacity.copy()
        return self.battery.copy()

    def observe(self, timestamp, solar_kw):
        if not np.isfinite(solar_kw) or solar_kw < 0:
            raise ValueError('Solar power must be finite and nonnegative.')
        return {'time': np.datetime64(timestamp, 's'), 'battery_kwh': self.battery.copy(),
                'solar_kw': float(solar_kw),
                'available_kwh': (solar_kw + self.cfg.contract_kw)*self.cfg.dt_hours}

    def demand(self, timestamp, prices, noise):
        # At reference price, daily baseline varies smoothly around base_kw.
        seconds = int(np.datetime64(timestamp, 's').astype(np.int64)) % 86400
        hour = seconds/3600
        profile = 0.85 + 0.15*np.cos(2*np.pi*(hour-18)/24)
        response = 1 - self.elasticity*(np.asarray(prices)/self.cfg.reference_price-1)
        return self.base_kw*profile*np.maximum(0, response+self.cfg.demand_noise*np.asarray(noise))*self.cfg.dt_hours

    def local_flows(self, battery, demand, allocation):
        c = self.cfg
        direct = np.minimum(demand, allocation)
        delivered_from_battery = np.minimum(np.maximum(demand-direct, 0),
                    np.minimum(battery*c.discharge_efficiency, self.discharge_kw*c.dt_hours))
        withdrawn = delivered_from_battery/c.discharge_efficiency
        charge_input = np.minimum(np.maximum(allocation-demand, 0),
                    np.minimum((self.capacity-battery)/c.charge_efficiency, self.charge_kw*c.dt_hours))
        stored = charge_input*c.charge_efficiency
        served = direct+delivered_from_battery
        return {'served': served, 'unmet': demand-served,
                'withdrawn': withdrawn, 'charge_input': charge_input,
                'spill': np.maximum(allocation-demand, 0)-charge_input,
                'loss': charge_input-stored+withdrawn-delivered_from_battery,
                'next_battery': battery+stored-withdrawn}

    def step(self, observation, prices, allocations, noise):
        c = self.cfg
        prices, allocations, noise = [np.asarray(v, dtype=float) for v in (prices, allocations, noise)]
        for value in (prices, allocations, noise):
            if value.shape != (c.microgrids,) or not np.isfinite(value).all():
                raise ValueError('Actions/noise must have one finite value per microgrid.')
        if np.any(prices < c.min_price-1e-10) or np.any(prices > c.max_price+1e-10):
            raise ValueError('Price outside permitted range.')
        if np.any(allocations < 0) or allocations.sum() > observation['available_kwh']+1e-9:
            raise ValueError('Allocations violate the shared supply constraint.')
        if not np.allclose(observation['battery_kwh'], self.battery, atol=1e-12):
            raise ValueError('Stale battery observation.')
        before = self.battery.copy()
        demand = self.demand(observation['time'], prices, noise)
        flow = self.local_flows(before, demand, allocations)
        self.battery = flow['next_battery']
        # Contracted energy is paid for whether allocated or not; solar has zero
        # marginal procurement cost here. Equal local cost shares sum correctly.
        contract_cost = c.contract_kw*c.dt_hours*c.contract_cost/c.microgrids
        revenue = prices*flow['served']
        reward = revenue-contract_cost-c.shortage_penalty*flow['unmet']-c.battery_wear*flow['withdrawn']
        unused = max(0., observation['available_kwh']-allocations.sum())
        lhs = observation['available_kwh']+before.sum()
        rhs = flow['served'].sum()+self.battery.sum()+flow['spill'].sum()+unused+flow['loss'].sum()
        if not np.isclose(lhs, rhs, atol=1e-8):
            raise AssertionError('Energy conservation failed.')
        if np.any(self.battery < -1e-9) or np.any(self.battery > self.capacity+1e-9):
            raise AssertionError('Battery capacity violated.')
        return {'demand': demand, 'revenue': revenue, 'reward': reward,
                'contract_cost_each': contract_cost, 'battery_before': before,
                'unused_source': unused, **flow}


def choose_action(policy, env, obs, rng):
    c = env.cfg
    if policy in ('fixed', 'explore'):
        prices = (np.full(c.microgrids, c.fixed_price) if policy == 'fixed' else
                  rng.uniform(c.min_price, c.max_price, c.microgrids))
        demand = env.demand(obs['time'], prices, np.zeros(c.microgrids))
        # Allocate toward anticipated consumption, with a small charging allowance.
        headroom = np.minimum((env.capacity-env.battery)/c.charge_efficiency,
                              env.charge_kw*c.dt_hours)
        desired = demand+0.25*headroom
        allocations = desired*min(1., obs['available_kwh']/max(desired.sum(), 1e-12))
        return prices, allocations

    if policy != 'myopic':
        raise ValueError('Unknown policy.')
    # Joint one-step allocation via small multiple-choice knapsack DP.
    bins = c.allocation_bins
    quantities = np.arange(bins+1)*obs['available_kwh']/bins
    price_grid = np.linspace(c.min_price, c.max_price, c.price_points)
    best = np.full((c.microgrids, bins+1), -np.inf)
    best_price = np.zeros_like(best)
    for price in price_grid:
        prices = np.full(c.microgrids, price)
        d = env.demand(obs['time'], prices, np.zeros(c.microgrids))
        # Grid-axis broadcasting for batched candidate actions.
        direct = np.minimum(d[:, None], quantities)
        delivery = np.minimum(np.maximum(d[:, None]-direct, 0),
                   np.minimum(env.battery[:, None]*c.discharge_efficiency,
                              env.discharge_kw[:, None]*c.dt_hours))
        served = direct+delivery
        score = price*served-c.shortage_penalty*(d[:, None]-served)-c.battery_wear*delivery/c.discharge_efficiency
        improve = score > best
        best[improve] = score[improve]
        best_price[improve] = price
    dp = np.zeros(bins+1)  # budget is an upper bound, so unused supply is allowed
    decisions = []
    for i in range(c.microgrids):
        nxt = np.empty_like(dp); take = np.zeros(bins+1, dtype=int)
        for budget in range(bins+1):
            options = best[i, :budget+1]+dp[budget::-1]
            take[budget] = int(np.argmax(options)); nxt[budget] = options[take[budget]]
        dp = nxt; decisions.append(take)
    budget = bins
    actions = np.zeros(c.microgrids, dtype=int)
    for i in range(c.microgrids-1, -1, -1):
        actions[i] = decisions[i][budget]; budget -= actions[i]
    return best_price[np.arange(c.microgrids), actions], quantities[actions]


def synthetic_episodes(days, seed):
    rng = np.random.default_rng(seed)
    episodes = []
    for day in range(days):
        n = 840
        times = np.datetime64('2020-06-01T06:00')+np.timedelta64(day, 'D')+np.arange(n)*np.timedelta64(1, 'm')
        clear = 28*np.maximum(0, np.sin(np.linspace(0, np.pi, n)))
        clouds = np.interp(np.arange(n), np.arange(0, n+60, 60), rng.uniform(0.35, 1, len(np.arange(0,n+60,60))))
        episodes.append((f'synthetic_day_{day+1}', times, clear*clouds))
    return episodes


def real_episodes(args):
    import h5py
    import pandas as pd
    with np.load(args.prepared, allow_pickle=False) as index:
        meta = json.loads(str(index['metadata']))
        origins = index['input_time'][index['split'] == args.split].astype('datetime64[s]')
        days = np.unique(origins.astype('datetime64[D]'))[:args.days]
    if not len(days):
        raise ValueError('No days in requested prepared-data split.')
    if meta['mode'] != 'frames':
        raise ValueError('Use the individual-observation SKIPP\'D file to replay consecutive minutes.')
    path = Path(args.h5) if args.h5 else Path(meta['source_h5'])
    records = []
    with h5py.File(path, 'r') as source:
        for group in meta['schemas']:
            # Only official/trusted files should be loaded with pickle enabled.
            raw = np.load(args.data_dir/f'times_{group}.npy', allow_pickle=args.trusted_timestamps)
            times = pd.DatetimeIndex(pd.to_datetime(raw.reshape(-1), errors='coerce'))
            if times.tz is not None:
                times = times.tz_localize(None)
            ts = times.to_numpy(dtype='datetime64[s]')
            pv = np.asarray(source[group]['pv_log'][:], dtype=float).reshape(-1)*meta['pv_to_kw']
            if len(ts) != len(pv):
                raise ValueError('Timestamp/PV count mismatch.')
            valid = ~np.isnat(ts) & np.isfinite(pv) & (pv >= 0) & np.isin(ts.astype('datetime64[D]'), days)
            records.append(pd.DataFrame({'time': ts[valid], 'solar_kw': pv[valid]}))
    frame = pd.concat(records, ignore_index=True).sort_values('time')
    frame = frame.loc[~frame['time'].duplicated(keep=False)]
    episodes = []
    for day, part in frame.groupby(frame['time'].dt.normalize()):
        ts = part['time'].to_numpy(dtype='datetime64[s]'); pv = part['solar_kw'].to_numpy()
        cuts = np.r_[0, np.flatnonzero(np.diff(ts).astype('timedelta64[s]').astype(int) != 60)+1, len(ts)]
        for segment, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
            if stop-start >= args.min_steps:
                episodes.append((f'{day.date()}_segment_{segment+1}', ts[start:stop], pv[start:stop]))
    if not episodes:
        raise ValueError('No sufficiently long consecutive-minute episodes; reduce --min-steps.')
    return episodes


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(args):
    c = Settings(microgrids=args.microgrids)
    episodes = synthetic_episodes(args.days, args.seed) if args.synthetic else real_episodes(args)
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed+1)
    summaries = []
    # Same latent demand shocks and recorded supply for every policy in an episode.
    # Actual demand differs across prices, as it should in a pricing experiment.
    with (args.output/'rollouts.csv').open('w', newline='') as stream:
        writer = None
        for episode_id, times, solar in episodes:
            noise = rng.normal(size=(len(times), c.microgrids))
            for policy in ['fixed', 'myopic', 'explore']:
                env = MicrogridEnv(c)
                policy_rng = np.random.default_rng(args.seed+2)
                totals = dict(profit=0., revenue=0., requested_kwh=0., served_kwh=0.,
                              unmet_kwh=0., unused_source_kwh=0., allocated_spill_kwh=0.,
                              losses_kwh=0., contract_cost=0., battery_wear_cost=0.)
                for step, (timestamp, power) in enumerate(zip(times, solar)):
                    obs = env.observe(timestamp, power)
                    prices, allocation = choose_action(policy, env, obs, policy_rng)
                    result = env.step(obs, prices, allocation, noise[step])
                    totals['profit'] += result['reward'].sum(); totals['revenue'] += result['revenue'].sum()
                    totals['requested_kwh'] += result['demand'].sum(); totals['served_kwh'] += result['served'].sum()
                    totals['unmet_kwh'] += result['unmet'].sum(); totals['unused_source_kwh'] += result['unused_source']
                    totals['allocated_spill_kwh'] += result['spill'].sum(); totals['losses_kwh'] += result['loss'].sum()
                    totals['contract_cost'] += result['contract_cost_each']*c.microgrids
                    totals['battery_wear_cost'] += c.battery_wear*result['withdrawn'].sum()
                    for i in range(c.microgrids):
                        row = dict(episode=episode_id, policy=policy, step=step, timestamp=str(timestamp), grid=i,
                            solar_kw=float(power), available_kwh=obs['available_kwh'], price=float(prices[i]),
                            allocated_kwh=float(allocation[i]), requested_kwh=float(result['demand'][i]),
                            served_kwh=float(result['served'][i]), unmet_kwh=float(result['unmet'][i]),
                            battery_before_kwh=float(result['battery_before'][i]),
                            battery_after_kwh=float(result['next_battery'][i]),
                            charged_input_kwh=float(result['charge_input'][i]),
                            battery_withdrawal_kwh=float(result['withdrawn'][i]),
                            loss_kwh=float(result['loss'][i]), spill_kwh=float(result['spill'][i]),
                            revenue=float(result['revenue'][i]), reward=float(result['reward'][i]))
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row)
                summaries.append(dict(episode=episode_id, policy=policy, steps=len(times),
                    initial_battery_kwh=float((0.5*env.capacity).sum()), final_battery_kwh=float(env.battery.sum()),
                    **{k: float(v) for k,v in totals.items()}))
                print(f"{episode_id} {policy:7s}: profit=${totals['profit']:.2f}, unmet={totals['unmet_kwh']:.2f} kWh", flush=True)
    write_csv(args.output/'episode_summary.csv', summaries)
    combined = []
    for policy in ['fixed', 'myopic', 'explore']:
        cases = [s for s in summaries if s['policy'] == policy]
        combined.append(dict(policy=policy, episodes=len(cases),
            total_profit=sum(s['profit'] for s in cases),
            total_unmet_kwh=sum(s['unmet_kwh'] for s in cases),
            total_served_kwh=sum(s['served_kwh'] for s in cases)))
    write_csv(args.output/'policy_summary.csv', combined)
    model = MicrogridEnv(c)
    config = dict(settings=asdict(c), base_demand_kw=model.base_kw.tolist(),
                  elasticities=model.elasticity.tolist(), battery_capacity_kwh=model.capacity.tolist(),
                  charge_limit_kw=model.charge_kw.tolist(), discharge_limit_kw=model.discharge_kw.tolist(),
                  seed=args.seed, source='synthetic' if args.synthetic else 'SKIPPD measured minute-average power',
                  prepared_split=None if args.synthetic else args.split,
                  demand_formula='base_kw * daily_profile * max(0,1-elasticity*(price/reference_price-1)+noise_sd*z) * dt_hours',
                  observation_assumption='perfect measurement of current minute-average supply; future supply hidden',
                  terminal_value='zero; compare final battery levels alongside profit',
                  parameter_status='illustrative simulation parameters, known to baselines, not empirically calibrated')
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'Saved rollouts.csv, episode_summary.csv, policy_summary.csv and config.json in {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', action='store_true')
    source.add_argument('--prepared', type=Path, help='prepared_skippd.npz from preprocessing')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--h5', type=Path, help='Override source HDF5 path stored in preprocessing index')
    parser.add_argument('--trusted-timestamps', action='store_true')
    parser.add_argument('--split', choices=['train','val','test'], default='test')
    parser.add_argument('--days', type=int, default=3)
    parser.add_argument('--min-steps', type=int, default=60)
    parser.add_argument('--microgrids', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('simulation_results'))
    args = parser.parse_args()
    if min(args.days, args.min_steps, args.microgrids) < 1 or args.seed < 0:
        parser.error('Counts must be positive and seed nonnegative.')
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
