from types import SimpleNamespace
import os, json, h5py
import multiprocessing as mp
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from config import ConfigCL, ConfigAE
from trainer import TrainerCL, TrainerAE
from cadlib.macro import *

np.random.seed(2024)
torch.manual_seed(2024)
data_dir = 'data'

def pad(cad_vec):
    pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
    return np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

def read_vec(path):
    path = os.path.join(data_dir, 'cad_vec', path) + '.h5'
    with h5py.File(path, 'r') as fp:
        cad_vec = fp['vec'][:].astype(int)
    cad_vec = pad(cad_vec)
    return cad_vec

def read_data(phase):
    json_path = os.path.join(data_dir, 'train_val_test_split.json')
    with open(json_path, 'r') as f:
        split = json.load(f)[phase]
    print('Reading data...')
    cad_vecs = mp.Pool(8).map(read_vec, tqdm(split))
    return cad_vecs

def encode_data(data, args):
    if not args.baseline:
        cfg = ConfigCL('test')
        agent = TrainerCL(cfg)
    else:
        cfg = ConfigAE('test')
        agent = TrainerAE(cfg)
    agent.load_ckpt(cfg.ckpt)
    agent.net.eval()
    with torch.no_grad():
        if len(data['command']) > 10000:
            z = []
            for i in tqdm(range(0, len(data['command']), 10000)):
                z.append(agent.encode({'command': data['command'][i:i+10000], 'args': data['args'][i:i+10000]}, is_batch=True))
            z = torch.cat(z, dim=0)
        else:
            z = agent.encode(data, is_batch=True)
            # z = agent.net.forward(batch['command'], batch['args'])['representation']
    z.squeeze_(1)
    return z.cpu().numpy()

def main(args):
    max_iter = args.max_iter

    cad_vecs_test = read_data('test')
    cad_vecs_test = torch.from_numpy(np.stack(cad_vecs_test, axis=0)).cuda()
    inp_data_test = {'command': cad_vecs_test[...,0], 'args': cad_vecs_test[...,1:]}

    # encode to latent vector
    print('>>> Encoding...')
    # encoded_zs_train = encode_data(inp_data_train)
    encoded_zs_test = encode_data(inp_data_test)
    del cad_vecs_test, inp_data_test

    # cluster them
    print('>>> Clustering...')
    k = int(args.k_rate * encoded_zs_test.shape[0])
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=2024).fit(encoded_zs_test)
    labels_test = kmeans.labels_

    # TSNE visualization
    if args.viz:
        print('>>> Visualizing...')
        tsne = TSNE(random_state=2024).fit_transform(encoded_zs_test, labels_test)
        plt.figure(figsize=(8,8))
        plt.scatter(tsne[:,0], tsne[:,1], c=labels_test)
        plt.legend()
        plt.show()
    
    # evaluate silhouette score
    print('>>> Evaluating...')
    sc_test = silhouette_score(encoded_zs_test, labels_test)
    sse_test = kmeans.inertia_
    print('Silhouette score:', sc_test)
    print('SSE:', sse_test)
    
    if not args.debug:
        with open('kmeans_sc_sse_result.txt', 'a') as f:
            my_str = f'[{"Ours" if not args.baseline else "Base"}] k={k}, max_iter={max_iter}, sc_test={sc_test}, sse_test={sse_test}\n'
            f.write(my_str)
    

if __name__ == '__main__':
    args = SimpleNamespace()
    args.max_iter = 300
    args.viz = False
    args.debug = False
    args.baseline = False
    rate_list = np.linspace(0.001, 0.25, 50)
    for i, rate in enumerate(rate_list):
        args.k_rate = rate
        main(args)