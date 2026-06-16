import torch
import numpy as np
from .load_data import load_dataset
from .charging import select_charging_station
import random
from itertools import islice
import env.mysql

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class AGV:
    def __init__(self, aid, pos=(0,0), soc=1.0, speed=1.0, standby=0.0001, mass=120.0):
        self.id = int(aid)
        self.pos = tuple(pos)
        self.soc = float(soc)
        self.speed = float(speed)
        self.standby = float(standby)
        self.available_time = 0.0
        self.utilization = 0.0
        self.load_energy = 0.0
        self.standby_energy = 0.0
        self.mass = float(mass)

class JobObj:
    def __init__(self, jid, num_ops, arrival=0.0, due=9999.0, mass=20.0, pos=(0,0)):
        self.id = int(jid)
        self.End = []
        self.assign_for = []
        self.position = tuple(pos)
        self.arrival = arrival
        self.due = due
        self.num_ops = num_ops
        self.mass = mass

    def _add(self, start, end, machine, PT):
        self.End.append(end)
        self.assign_for.append(machine)

class MachineObj:
    def __init__(self, mid, pos=(0,0)):
        self.id = int(mid)
        self.pos = tuple(pos)
        self.T = []
        self.End = []
        self.proc_energy = 0.0
        self.idle_energy = 0.0

    def _add(self, start, end, job, PT):
        self.T.append(PT)
        self.End.append(end)

class ActionSelector:
    def __init__(self, env):
        self.env = env
        self.agvs = env.agvs
        self.jobs = env.Jobs
        self.machines = env.Machines

    def select_nearest_agv(self, job):
        jx, jy = job.position
        distances = [manhattan(a.pos, (jx, jy)) for a in self.agvs]
        return int(np.argmin(distances))

    def select_earliest_agv(self):
        available = [a.available_time for a in self.agvs]
        return int(np.argmin(available))

    def select_lowest_util_agv(self):
        utils = [a.utilization for a in self.agvs]
        return int(np.argmin(utils))

    def tardiness_score(self, job_id):
        cur_op = len(self.jobs[job_id].End)
        remain_time = 0.0
        for op in range(cur_op, self.env.J[job_id]):
            p = [t for t in self.env.Processing_time[job_id][op] if t != -1]
            if len(p) == 0:
                continue
            remain_time += sum(p) / len(p)
        avg_C = sum(self.env.CTK) / max(1, self.env.M_num)
        return avg_C + remain_time - self.env.D[job_id]

    def slack_time(self, job_id):
        cur_op = len(self.jobs[job_id].End)
        remain_time = 0.0
        for op in range(cur_op, self.env.J[job_id]):
            p = [t for t in self.env.Processing_time[job_id][op] if t != -1]
            if len(p) == 0:
                continue
            remain_time += sum(p) / len(p)
        now = sum(self.env.CTK) / max(1, self.env.M_num)
        return self.env.D[job_id] - (now + remain_time)

    def select_machines(self, job_id):
        sql = "SELECT meid from me2_1 WHERE emptytime=-1"
        env.mysql.ti_obj.ping(reconnect=True)
        env.mysql.cursor.execute(sql)
        result = env.mysql.cursor.fetchall()
        env.mysql.ti_obj.commit()
        sql = "SELECT meid FROM me2_1 WHERE res1=(SELECT MIN(res1) FROM me2_1 where ID!=9)"
        env.mysql.ti_obj.ping(reconnect=True)
        env.mysql.cursor.execute(sql)
        result = env.mysql.cursor.fetchall()
        env.mysql.ti_obj.commit()
        machine = env.mysql.Q.index(result[0][0])
        return machine

    def apply_action(self, action_id):
        unfinished = [j for j in range(self.env.J_num) if self.env.OP[j] < self.env.J[j]]
        if len(unfinished) == 0:
            return None
        if action_id in [0,1,2]:
            scores = [self.tardiness_score(j) for j in unfinished]
            job_id = unfinished[int(np.argmax(scores))]
        else:
            scores = [self.slack_time(j) for j in unfinished]
            job_id = unfinished[int(np.argmin(scores))]
        job = self.jobs[job_id]
        cur_op = len(job.End)
        machine_id = self.select_machines(job_id)
        PT = self.env.Processing_time[job_id][cur_op][machine_id]
        if action_id in [0,3]:
            agv_id = self.select_nearest_agv(job)
        elif action_id in [1,4]:
            agv_id = self.select_earliest_agv()
        else:
            agv_id = self.select_lowest_util_agv()
        return (agv_id, job_id, machine_id, cur_op, PT)

    def select_machine(self, job_id):
        cur_op = len(self.jobs[job_id].End)
        MR = []
        for m in range(self.env.M_num):
            if self.env.Processing_time[job_id][cur_op][m] == -1:
                MR.append(999999)
            else:
                MR.append(sum(self.machines[m].T) if len(self.machines[m].T) > 0 else 0.0)
        return int(np.argmin(MR))

class FJSPEnv:
    def __init__(self, instance, cfg):
        self.instance = instance
        self.cfg = cfg

        self.J_num = instance.get('num_jobs', len(instance.get('jobs', [])))
        self.M_num = instance.get('num_machines', 0)
        self.A_num = instance.get('num_agvs', 0)
        self.J = [len(job.get('ops', [])) for job in instance.get('jobs', [])]
        self.Processing_time = self._build_processing_time(instance)
        self.schedule = {job_id: [] for job_id in range(self.J_num)}
        self.machine_records = {m: [] for m in range(self.M_num)}
        self.Ai = [0.0]*self.J_num
        self.D = [job.get('due_date', 9999.0) for job in instance.get('jobs', [])]
        self.Machines = [MachineObj(i, pos=tuple(cfg.get('machine_positions', {}).get(str(i), (0,0)))) for i in range(self.M_num)]
        self.Jobs = [JobObj(i, self.J[i],
                            arrival=self.Ai[i],
                            due=self.D[i],
                            mass=float(self.cfg.get('job_mass', {}).get(str(i), self.cfg.get('default_job_mass',20.0))),
                            pos=tuple(self.cfg.get('job_positions', {}).get(str(i), (0,0))))
                     for i in range(self.J_num)]
        self.agvs = []
        agv_pos_cfg = self.cfg.get('agv_positions', {})
        agv_pos_cfg = dict(islice(agv_pos_cfg.items(), self.A_num))
        for k,v in agv_pos_cfg.items():
            aid = int(k)
            pos = tuple(v)
            soc = float(self.cfg.get('agv_initial_soc',1.0))
            speed = float(self.cfg.get('agv_speed',1.0))
            standby = float(self.cfg.get('agv_standby_power',0.0001))
            mass = float(self.cfg.get('agv_mass',120.0))
            self.agvs.append(AGV(aid, pos=pos, soc=soc, speed=speed, standby=standby, mass=mass))
        self.agv_map = {str(a.id): a for a in self.agvs}
        for a in self.agvs:
            self.agv_map[a.id] = a

        charger_cfg = self.cfg.get('charger_positions', {})
        self.charger_positions = {int(k): tuple(v) for k,v in charger_cfg.items()}
        self.charger_next_free = {int(k): 0.0 for k in charger_cfg}
        self.charger_charge_rate = {int(k): float(v) for k,v in self.cfg.get('charger_charge_rate', {}).items()} if self.cfg.get('charger_charge_rate') else {}
        self.CTK = [0.0]*self.M_num
        self.OP = [0]*self.J_num
        self.UK = [0.0]*self.M_num
        self.CRJ = [0.0]*self.J_num
        self.machine_proc_energy = {m:0.0 for m in range(self.M_num)}
        self.machine_idle_energy = {m:0.0 for m in range(self.M_num)}

        self.action_selector = ActionSelector(self)
        self.time = 0.0
        self.state_dim = int(self.cfg.get('embed_dim',64)*2)
        self.action_dim = int(self.cfg.get('action_dim',6))

    def _build_processing_time(self, instance):
        jobs = instance.get('jobs', [])
        P = []
        for j, job in enumerate(jobs):
            ops = job.get('ops', [])
            row_list = []
            for op in ops:
                row = [-1]*self.M_num
                for cand in op.get('candidates', []):
                    m = int(cand['machine'])
                    t = float(cand['proc_time'])
                    row[m] = t
                row_list.append(row)
            P.append(row_list)
        return P

    def reset(self):
        self.CTK = [0.0]*self.M_num
        self.OP = [0]*self.J_num
        self.UK = [0.0]*self.M_num
        self.CRJ = [0.0]*self.J_num
        for j in range(self.J_num):
            self.Jobs[j] = JobObj(j, self.J[j], arrival=self.Ai[j], due=self.D[j],
                                  mass=float(self.cfg.get('job_mass', {}).get(str(j), self.cfg.get('default_job_mass',20.0))),
                                  pos=tuple(self.cfg.get('job_positions', {}).get(str(j),(0,0))))
        for m in range(self.M_num):
            self.Machines[m] = MachineObj(m, pos=tuple(self.cfg.get('machine_positions', {}).get(str(m),(0,0))))
            self.machine_proc_energy[m] = 0.0
            self.machine_idle_energy[m] = 0.0
        for a in self.agvs:
            a.soc = float(self.cfg.get('agv_initial_soc', 1.0))
            a.available_time = 0.0
            a.utilization = 0.0
            a.load_energy = 0.0
            a.standby_energy = 0.0
        for sid in self.charger_next_free:
            self.charger_next_free[sid] = 0.0
        self.time = 0.0
        return self.get_raw_observation()

    def get_raw_observation(self):
        op_feats = []
        flat = []
        for j_idx, job in enumerate(self.instance['jobs']):
            for o_idx, op in enumerate(job.get('ops', [])):
                cands = op.get('candidates', [])
                c = len(cands)
                mean_t = float(sum([cand['proc_time'] for cand in cands]) / c) if c>0 else 0.0
                job_norm = j_idx / max(1, self.J_num)
                max_ops = max(self.J) if len(self.J)>0 else 1
                op_norm = o_idx / max(1, max_ops)
                op_feats.append([mean_t, float(c), job_norm, op_norm])
                flat.append((j_idx, o_idx))
        if len(op_feats)==0:
            op_tensor = torch.zeros((0, self.cfg.get('op_feat_dim',4)), dtype=torch.float32)
        else:
            op_tensor = torch.tensor(op_feats, dtype=torch.float32)
        agv_feat = torch.zeros((len(self.agvs), self.cfg.get('agv_feat_dim',4)), dtype=torch.float32)
        mach_feat = torch.zeros((len(self.Machines), self.cfg.get('mach_feat_dim',4)), dtype=torch.float32)
        op_agv_adj = torch.ones((op_tensor.size(0), len(self.agvs)), dtype=torch.float32)
        agv_mach_adj = torch.ones((len(self.agvs), len(self.Machines)), dtype=torch.float32)
        return (op_tensor, agv_feat, mach_feat, op_agv_adj, agv_mach_adj)

    def _flatten_ops(self):
        flattened = []
        for j_idx, job in enumerate(self.instance['jobs']):
            for o_idx, op in enumerate(job.get('ops', [])):
                flattened.append((j_idx, o_idx, op))
        return flattened

    def _candidate_triplets(self):
        triplets = []
        flat = self._flatten_ops()
        for idx, (j, o, op) in enumerate(flat):
            if self.OP[j] > o:
                continue
            for cand in op.get('candidates', []):
                m_idx = int(cand['machine'])
                for a in self.agvs:
                    if a.soc >= self.cfg.get('agv_min_soc_for_task', 0.2):
                        triplets.append((idx, a.id, m_idx))
        return triplets

    def compute_energy_metrics(self):
        E_agv = sum([a.load_energy + a.standby_energy for a in self.agvs])
        E_mach = sum([self.machine_proc_energy[m] + self.machine_idle_energy[m] for m in range(self.M_num)])
        E_sum = float(self.cfg.get("E_sum", 10000.0))
        E_ratio = (E_agv + E_mach) / (E_sum + 1e-9)
        return E_ratio, E_agv, E_mach

    def compute_tard_ratio(self, now_time):
        N_tard = 0
        N_left = 0
        for j in range(self.J_num):
            remain = self.J[j] - len(self.Jobs[j].End)
            if remain > 0:
                N_left += remain
                if now_time > self.Jobs[j].due:
                    N_tard += remain
        if N_left == 0:
            return 0.0, 0, 1
        return N_tard / N_left, N_tard, N_left

    def step(self, action_tuple):
        if action_tuple is None:
            return self.get_raw_observation(), 0.0, True, {}
        agv_id, job_id, machine_id, op_idx, PT = action_tuple
        E_t, Eagv_prev, Emach_prev = self.compute_energy_metrics()
        now = sum(self.CTK) / max(1, self.M_num)
        tard_t, N_tard_t, N_left_t = self.compute_tard_ratio(now)
        agv = None
        for a in self.agvs:
            if a.id == int(agv_id):
                agv = a; break
        if agv is None:
            agv = self.agvs[0]
        last_ot = max(self.Jobs[job_id].End) if len(self.Jobs[job_id].End)>0 else 0.0
        last_mt = max(self.Machines[machine_id].End) if len(self.Machines[machine_id].End)>0 else 0.0
        ai_time = self.Jobs[job_id].arrival if hasattr(self.Jobs[job_id], 'arrival') else 0.0
        Start_time = max(last_ot, last_mt, ai_time)
        mach_pos = self.Machines[machine_id].pos
        travel_dist = manhattan(agv.pos, mach_pos)
        agv_speed = float(self.cfg.get("agv_speed", agv.speed))
        travel_time = travel_dist / max(1e-6, agv_speed)
        if op_idx == 0:
            PT_adj = PT + travel_time
        else:
            prev_machine = self.Jobs[job_id].assign_for[-1] if len(self.Jobs[job_id].assign_for)>0 else None
            if prev_machine == machine_id:
                PT_adj = PT
            else:
                PT_adj = PT + travel_time
        end_time = Start_time + PT_adj
        self.schedule[job_id].append({
            'op_id': op_idx,
            'machine': machine_id,
            'agv': agv_id,
            'start': Start_time,
            'end': end_time
        })
        self.machine_records[machine_id].append({
            'job_id': job_id,
            'op_id': op_idx,
            'start': Start_time,
            'end': end_time,
            'processing_time': end_time - Start_time
        })
        self.Machines[machine_id]._add(Start_time, end_time, job_id, PT_adj)
        self.Jobs[job_id]._add(Start_time, end_time, machine_id, PT_adj)
        ed = float(self.cfg.get("agv_energy_ed", 0.02))
        eds = float(self.cfg.get("agv_energy_eds", 0.005))
        Magv = float(self.cfg.get("agv_mass", agv.mass if hasattr(agv,'mass') else 120.0))
        Mjob = float(self.cfg.get("job_mass", {}).get(str(job_id), self.cfg.get("default_job_mass",20.0)))
        E_agv_load = ed * agv_speed * travel_time * (Magv + Mjob)
        E_agv_static = eds * PT_adj
        agv.load_energy += E_agv_load
        agv.standby_energy += E_agv_static
        P_proc = float(self.cfg.get('machine_power_proc', {}).get(str(machine_id), self.cfg.get('default_machine_proc_power', 10.0)))
        P_idle = float(self.cfg.get('machine_power_idle', {}).get(str(machine_id), self.cfg.get('default_machine_idle_power', 1.0)))
        E_machine_proc = P_proc * PT_adj
        last_end = self.Machines[machine_id].End[-2] if len(self.Machines[machine_id].End) > 1 else 0.0
        idle_t = max(0.0, Start_time - last_end)
        E_machine_idle = P_idle * idle_t
        self.machine_proc_energy[machine_id] += E_machine_proc
        self.machine_idle_energy[machine_id] += E_machine_idle
        agv.pos = mach_pos
        agv.utilization += PT_adj
        agv.available_time = end_time
        soc_consumption = (E_agv_load + E_agv_static) / (self.cfg.get('E_sum', 10000.0) + 1e-9)
        agv.soc = max(0.0, agv.soc - soc_consumption)
        self.CTK[machine_id] = max(self.Machines[machine_id].End) if len(self.Machines[machine_id].End)>0 else self.CTK[machine_id]
        self.OP[job_id] += 1
        self.UK[machine_id] = sum(self.Machines[machine_id].T) / (self.CTK[machine_id] + 1e-9)
        self.CRJ[job_id] = self.OP[job_id] / max(1, self.J[job_id])
        E_t1, Eagv_now, Emach_now = self.compute_energy_metrics()
        tard_t1, N_tard_t1, N_left_t1 = self.compute_tard_ratio(end_time)
        W1 = float(self.cfg.get('W1', 0.5))
        W2 = float(self.cfg.get('W2', 0.5))
        reward = W1 * (E_t1 - E_t) * 100.0 + W2 * (tard_t - tard_t1) * 100.0
        charge_info = None
        if agv.soc < float(self.cfg.get('agv_min_soc_for_task', 0.2)):
            charge_info = self.request_charge(agv.id)
        raw_obs = self.get_raw_observation()
        done = all([len(self.Jobs[j].End) >= self.J[j] for j in range(self.J_num)])
        info = {
            "machine_id": machine_id, "job_id": job_id, "operation_id": op_idx, "agv_id": agv.id,
            "E_agv_load": E_agv_load, "E_agv_standby": E_agv_static,
            "E_mach_proc": E_machine_proc, "E_mach_idle": E_machine_idle,
            "E_t": E_t, "E_t1": E_t1,
            "t_t": tard_t, "t_t1": tard_t1,
            "N_tard_t": N_tard_t, "N_left_t": N_left_t,
            "N_tard_t1": N_tard_t1, "N_left_t1": N_left_t1,
            "charge_info": charge_info
        }
        return raw_obs, float(reward), bool(done), info

    def request_charge(self, agv_id):
        current_time = self.time
        agv_positions = {a.id: a.pos for a in self.agvs}
        agv_socs = {a.id: a.soc for a in self.agvs}
        agv_speeds = {a.id: a.speed for a in self.agvs}
        agv_standby_power = {a.id: a.standby for a in self.agvs}
        station_list = list(self.charger_positions.keys())
        station_next_free = self.charger_next_free
        station_positions = self.charger_positions
        station_charge_rate = self.charger_charge_rate
        other_positions = {a.id: a.pos for a in self.agvs}
        weights = (self.cfg.get('charge_w1',0.4), self.cfg.get('charge_w2',0.3), self.cfg.get('charge_w3',0.3))
        energy_per_cell = self.cfg.get('energy_per_cell', 0.01)
        chosen_sid, details = select_charging_station(
            agv_id=agv_id,
            agv_positions=agv_positions,
            agv_socs=agv_socs,
            agv_speeds=agv_speeds,
            agv_standby_power=agv_standby_power,
            station_list=station_list,
            station_next_free_times=station_next_free,
            station_positions=station_positions,
            station_charge_rate=station_charge_rate,
            current_time=current_time,
            other_agv_positions=other_positions,
            weights=weights,
            energy_per_cell=energy_per_cell
        )
        if chosen_sid is None:
            return {'error':'no_station'}
        Tearly = details['Tearly']
        desired_charge = max(0.0, 1.0 - self.agv_map[str(agv_id)].soc)
        charge_rate = station_charge_rate.get(chosen_sid, self.cfg.get('default_charge_rate', 0.5))
        charge_duration = float('inf') if charge_rate <= 1e-9 else desired_charge / charge_rate
        finish_time = Tearly + charge_duration
        self.charger_next_free[chosen_sid] = finish_time
        agv = self.agv_map.get(str(agv_id), self.agv_map.get(agv_id))
        agv.pos = self.charger_positions[chosen_sid]
        agv.soc = 1.0
        agv.available_time = finish_time
        info = {
            'chosen_station': chosen_sid,
            'Tearly': Tearly,
            'Eagv': details.get('Eagv'),
            'Fp': details.get('Fp'),
            'score': details.get('score'),
            'start_charge_time': Tearly,
            'charge_duration': charge_duration,
            'finish_time': finish_time
        }
        return info
