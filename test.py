import copy
import json
import os
import time
import random
import argparse

import numpy as np
import torch
import pandas as pd

from env.load_data import load_dataset
from env.fjsp_env import FJSPEnv
from graph.hgnn import MLPsim, GATTriple
from model.D3QN_model import D3QN

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def build_state_from_raw_obs(hgnn_modules, raw_obs, device):
    mlp_op = hgnn_modules['mlp_op']
    gat_triple = hgnn_modules['gat_triple']
    op_feat, agv_feat, mach_feat, op_agv_adj, agv_mach_adj = raw_obs
    op_feat = op_feat.to(device)
    agv_feat = agv_feat.to(device)
    mach_feat = mach_feat.to(device)
    op_agv_adj = op_agv_adj.to(device)
    agv_mach_adj = agv_mach_adj.to(device)
    with torch.no_grad():
        op_emb = mlp_op(op_feat)
        agv_emb, _ = gat_triple(
            op_emb, agv_feat, mach_feat,
            op_agv_adj, agv_mach_adj
        )
    s_op = op_emb.mean(dim=0)
    s_agv = agv_emb.mean(dim=0)
    state = torch.cat([s_op, s_agv], dim=0)
    return state.cpu().numpy()

def schedule(env, agent, hgnn_modules, device, max_steps):
    raw_obs = env.reset()
    state = build_state_from_raw_obs(hgnn_modules, raw_obs, device)
    done = False
    start_time = time.time()
    step = 0
    while not done and step < max_steps:
        action = agent.select_action(
            state,
            n_step=10**9,
            N_iter=10**9
        )
        triplet = env.action_selector.apply_action(action)
        if triplet is None:
            break
        next_raw_obs, reward, done, info = env.step(triplet)
        state = build_state_from_raw_obs(hgnn_modules, next_raw_obs, device)
        step += 1

    spend_time = time.time() - start_time

    if hasattr(env, "validate_gantt"):
        ok = env.validate_gantt()
        if not ok:
            print("Scheduling Error!")

    return env.makespan, spend_time

def main():
    setup_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch device:", device)

    torch.set_default_tensor_type(
        'torch.cuda.FloatTensor' if device.type == 'cuda'
        else 'torch.FloatTensor'
    )
    with open("utils/config.json", 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    test_paras = cfg["test_paras"]
    num_ins = test_paras["num_ins"]
    max_steps = cfg.get("max_steps", 500)
    data_path = "./data_dev/{0}/".format(test_paras["data_path"])
    test_files = sorted(os.listdir(data_path))[:num_ins]
    embed_dim = cfg.get("embed_dim", 64)
    mlp_op = MLPsim(cfg["op_feat_dim"], embed_dim).to(device)
    gat_triple = GATTriple(
        embed_dim,
        cfg["agv_feat_dim"],
        cfg["mach_feat_dim"],
        embed_dim
    ).to(device)

    hgnn_modules = {
        "mlp_op": mlp_op,
        "gat_triple": gat_triple
    }
    agent = D3QN(embed_dim * 2, cfg["action_dim"], cfg)
    rules = test_paras["rules"]
    model_files = []

    if "DRL" in rules:
        for f in os.listdir("./model"):
            if f.endswith(".pt"):
                model_files.append(f)

    if len(model_files) > 0:
        rules = model_files
    str_time = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"./save/test_{str_time}"
    os.makedirs(save_path, exist_ok=True)
    writer_mk = pd.ExcelWriter(f"{save_path}/makespan.xlsx")
    writer_t = pd.ExcelWriter(f"{save_path}/time.xlsx")
    pd.DataFrame(test_files, columns=["file"]).to_excel(
        writer_mk, index=False
    )
    pd.DataFrame(test_files, columns=["file"]).to_excel(
        writer_t, index=False
    )
    start = time.time()
    for rule_id, rule in enumerate(rules):
        print("\nTesting rule:", rule)
        ckpt = torch.load(f"./model/{rule}", map_location=device)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        mlp_op.load_state_dict(ckpt["mlp_op"])
        gat_triple.load_state_dict(ckpt["gat_triple"])
        agent.online.eval()
        mlp_op.eval()
        gat_triple.eval()

        makespans = []
        times = []
        for i, fname in enumerate(test_files):
            data = load_dataset(os.path.join(data_path, fname))
            env = FJSPEnv(data, cfg)
            mk, t = schedule(
                env, agent, hgnn_modules, device, max_steps
            )
            makespans.append(mk)
            times.append(t)
            print(f"  Env {i}: makespan={mk:.2f}, time={t:.2f}s")
        pd.DataFrame(makespans, columns=[rule]).to_excel(
            writer_mk, startcol=rule_id + 1, index=False
        )
        pd.DataFrame(times, columns=[rule]).to_excel(
            writer_t, startcol=rule_id + 1, index=False
        )
    writer_mk.close()
    writer_t.close()
    print("Total test time:", time.time() - start)

if __name__ == "__main__":
    main()
