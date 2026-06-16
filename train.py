import argparse
import json
import os
import random
import time
import torch
import torch.nn.functional as F
import pymysql

from env.load_data import load_dataset
from env.fjsp_env import FJSPEnv
from graph.hgnn import MLPsim, GATTriple
from model.D3QN_model import D3QN

def build_state_from_raw_obs(hgnn_modules, raw_obs, device):
    mlp_op = hgnn_modules['mlp_op']
    gat_triple = hgnn_modules['gat_triple']
    op_feat, agv_feat, mach_feat, op_agv_adj, agv_mach_adj = raw_obs
    op_feat = op_feat.to(device); agv_feat = agv_feat.to(device); mach_feat = mach_feat.to(device)
    op_agv_adj = op_agv_adj.to(device); agv_mach_adj = agv_mach_adj.to(device)
    with torch.no_grad():
        op_emb = mlp_op(op_feat) if op_feat.size(0)>0 else mlp_op(op_feat)
        agv_emb, mach_emb = gat_triple(op_emb, agv_feat, mach_feat, op_agv_adj, agv_mach_adj)
    s_op = op_emb.mean(dim=0) if op_emb.size(0)>0 else torch.zeros(mlp_op.project[-1].out_features, device=device)
    s_agv = agv_emb.mean(dim=0) if agv_emb.size(0)>0 else torch.zeros(gat_triple._out, device=device)
    state = torch.cat([s_op, s_agv], dim=0)
    return state.cpu().numpy()

def raw_obs_to_numpy(raw_obs):
    return [x.cpu().numpy() for x in raw_obs]

def numpy_to_raw_obs(arr_list, device):
    return tuple(torch.from_numpy(x).float().to(device) for x in arr_list)

def main(config_path):
    cfg = json.load(open(config_path,'r',encoding='utf-8'))
    train_path = cfg.get('train_data', None)
    if train_path is None:
        raise ValueError("train_data must be set in config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(train_path)
    env = FJSPEnv(data, cfg)
    embed_dim = int(cfg.get('embed_dim', 64))
    mlp_op = MLPsim(cfg.get('op_feat_dim',4), embed_dim).to(device)
    gat_triple = GATTriple(in_feats_op=embed_dim,
                           in_feats_agv=cfg.get('agv_feat_dim',4),
                           in_feats_mach=cfg.get('mach_feat_dim',4),
                           out_feats=embed_dim, num_head=1).to(device)
    hgnn_modules = {'mlp_op': mlp_op, 'gat_triple': gat_triple}
    state_dim = embed_dim * 2
    action_dim = int(cfg.get('action_dim', 6))
    agent = D3QN(state_dim, action_dim, cfg)
    joint_params = list(agent.online.parameters()) + list(mlp_op.parameters()) + list(gat_triple.parameters())
    joint_optimizer = torch.optim.Adam(joint_params, lr=cfg.get('lr', 1e-4))
    episodes = int(cfg.get('episodes', 10))
    max_steps = int(cfg.get('max_steps', 500))
    N_iter = int(cfg.get('N_iter', episodes * max_steps))
    n_step_global = 0
    os.makedirs('checkpoints', exist_ok=True)
    print("训练开始，设备：", device)
    for ep in range(1, episodes+1):
        raw_obs = env.reset()
        state_vec = build_state_from_raw_obs(hgnn_modules, raw_obs, device)
        total_reward = 0.0
        for step in range(max_steps):
            action_id = agent.select_action(state_vec, n_step_global, N_iter)
            triplet = env.action_selector.apply_action(action_id)
            if triplet is None:
                break
            next_raw_obs, reward, done, info = env.step(triplet)
            next_vec = build_state_from_raw_obs(hgnn_modules, next_raw_obs, device)
            agent.store(state_vec, action_id, float(reward), next_vec, float(done))
            job_id = info['job_id']
            machine_id = info['machine_id']
            op_id = info['operation_id']
            Q = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
            ti_obj = pymysql.connect(
                host='172.23.46.214',
                port=3306,
                user='root',
                passwd='test123',
                database='mysql',
                charset='utf8'
            )
            if op_id == 0:
                froms = Q[machine_id]
                tos = froms
            else:
                machine_before = env.schedule[job_id][-1]['machine']
                froms = Q[machine_before]
                tos = Q[machine_id]
            cursor = ti_obj.cursor()
            sql = "SELECT time,number FROM me1_2 WHERE froms='%s' AND tos='%s'" % (froms, tos)
            ti_obj.ping(reconnect=True)
            cursor.execute(sql)
            number = cursor.fetchall()
            ti_obj.commit()
            timenew = number[0][0]
            numbernew = number[0][1] + 1
            sql = "update me1_2 SET time=%lf, number=%d WHERE froms='%s' AND tos='%s'" % (
                timenew, numbernew, froms, tos)
            ti_obj.ping(reconnect=True)
            cursor.execute(sql)
            ti_obj.commit()
            lasttime = env.machine_records[machine_id][-1]['end']
            t = 0
            for rec in env.machine_records[machine_id]:
                t += rec['processing_time']
            t = lasttime - t
            if t != 0:
                sql = "update me2_2 SET emptytime=%lf WHERE meid='%s'" % (t, tos)
                ti_obj.ping(reconnect=True)
                cursor.execute(sql)
                ti_obj.commit()
            else:
                sql = "update me2_2 SET emptytime=0 WHERE meid='%s'" % (tos)
                ti_obj.ping(reconnect=True)
                cursor.execute(sql)
                ti_obj.commit()
            ti_obj.close()
            if len(agent.buffer) >= agent.batch_size:
                minib = random.sample(agent.buffer, min(agent.batch_size, 32))
                joint_optimizer.zero_grad()
                total_joint_loss = 0.0
                for (s0,a0,r0,s1,d0) in minib:
                    s0t = torch.FloatTensor(s0).to(device)
                    s1t = torch.FloatTensor(s1).to(device)
                    q_pred = agent.online(s0t.unsqueeze(0))
                    q_val = q_pred[0, a0]
                    with torch.no_grad():
                        q_next_online = agent.online(s1t.unsqueeze(0)).argmax(dim=1)
                        q_next_target = agent.target(s1t.unsqueeze(0))[0, q_next_online]
                        target = torch.tensor(r0, dtype=torch.float32, device=device)
                        if d0 == 0.0:
                            target = target + agent.gamma * q_next_target
                    q_val = q_val.view(1)
                    target = target.view(1)
                    loss_j = F.mse_loss(q_val, target.detach())
                    total_joint_loss = total_joint_loss + loss_j
                total_joint_loss = total_joint_loss / float(max(1,len(minib)))
                total_joint_loss.backward()
                torch.nn.utils.clip_grad_norm_(joint_params, 1.0)
                joint_optimizer.step()
            state_vec = next_vec
            total_reward += reward
            n_step_global += 1
        if ep % max(1, cfg.get('save_every', 50)) == 0:
            torch.save({
                'online': agent.online.state_dict(),
                'target': agent.target.state_dict(),
                'mlp_op': mlp_op.state_dict(),
                'gat_triple': gat_triple.state_dict()
            }, f'model/ckpt_ep{ep}.pt')
    torch.save({
        'online': agent.online.state_dict(),
        'target': agent.target.state_dict(),
        'mlp_op': mlp_op.state_dict(),
        'gat_triple': gat_triple.state_dict()
    }, 'model/final.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='utils/config.json')
    args = parser.parse_args()
    start = time.time()
    main(args.config)
    end = time.time()
    print(f"Total elapsed: {end-start:.2f}s")
