import numpy as np

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def normalize_list(x):
    arr = np.array(x, dtype=float)
    mn = arr.min(); mx = arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def select_charging_station(agv_id,
                            agv_positions,
                            agv_socs,
                            agv_speeds,
                            agv_standby_power,
                            station_list,
                            station_next_free_times,
                            station_positions,
                            station_charge_rate,
                            current_time,
                            other_agv_positions,
                            weights=(0.4,0.3,0.3),
                            energy_per_cell=0.01):
    w1, w2, w3 = weights
    candidates = []
    for sid in station_list:
        P = station_positions[sid]
        agv_pos = agv_positions[agv_id]
        dist = manhattan(agv_pos, P)
        speed = max(agv_speeds.get(agv_id, 1.0), 1e-6)
        travel_time = dist / speed
        arrival_time = current_time + travel_time
        station_free = station_next_free_times.get(sid, current_time)
        Tearly = max(arrival_time, station_free)

        Erun = dist * energy_per_cell  # 简单距离能耗
        wait_time = max(0.0, station_free - arrival_time)
        standby_power = agv_standby_power.get(agv_id, 0.0)
        Ewait = standby_power * wait_time
        Eagv = Erun + Ewait

        desired_charge = max(0.0, 1.0 - agv_socs.get(agv_id, 0.5))
        charge_rate = station_charge_rate.get(sid, 0.5)
        charge_time = float('inf') if charge_rate <= 1e-9 else desired_charge / charge_rate
        finish_time = Tearly + charge_time
        candidates.append({
            'station': sid,
            'dist': dist,
            'travel_time': travel_time,
            'arrival_time': arrival_time,
            'station_free': station_free,
            'Tearly': Tearly,
            'Erun': Erun,
            'Ewait': Ewait,
            'Eagv': Eagv,
            'charge_time': charge_time,
            'finish_time': finish_time,
            'P': P
        })

    if not candidates:
        return None, {}

    for cand in candidates:
        P = cand['P']
        pis = [pos for pos in other_agv_positions.values()]
        n = max(1, len(pis))
        s = 0.0
        for pi in pis:
            s += manhattan(pi, P)
        cand['Fp'] = s / n

    Tvals = [c['Tearly'] for c in candidates]
    Eval = [c['Eagv'] for c in candidates]
    Fvals = [c['Fp'] for c in candidates]
    Tnorm = normalize_list(Tvals); Enorm = normalize_list(Eval); Fnorm = normalize_list(Fvals)

    for i, c in enumerate(candidates):
        t = float(Tnorm[i]); e = float(Enorm[i]); f = float(Fnorm[i])
        score = w1 * t + w2 * e + w3 * f
        c['Tnorm'] = t; c['Enorm'] = e; c['Fnorm'] = f; c['score'] = float(score)

    chosen = min(candidates, key=lambda x: x['score'])
    return chosen['station'], chosen
