import os 
import pickle 
import time
import json 

import torch 
import torch.distributed as dist
import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from models.embedder import NodeEmbedder
from models.gan import GCNDiscriminator, NodeGenerator
from gan_test import test_emb_input

N_JOBS = 4 # How many worker processes will train the model
P_THREADS = 4 # About the point of diminishing returns from experiments

criterion = BCEWithLogitsLoss()
mse = MSELoss()

# Embedder params
EMB_HIDDEN = 64
EMB_OUT = 32
EMBED_SIZE = 64
T2V = 8
ATTN_KW = {
    'layers': 4,
    'heads': 8
}

# GAN Params
HIDDEN_GEN = 128
HIDDEN_DISC= 64 
STATIC_DIM = 8

# Training params
EPOCHS = 25
EMB_LR = 0.00025
GEN_LR = 0.00025
DISC_LR= 0.00025

TRUE_VAL = 0.1 # One-sided label smoothing (real values smoothed)
FALSE_VAL = 1. # False should approach 1 as in an anomaly score
MAX_SAMPLES = 50

# Decide which architecture to use here
NodeEmb = NodeEmbedder
NodeGen = NodeGenerator
NodeDisc = GCNDiscriminator

def sample(nodes, max_samples=0):
    return {
        'regs': nodes.sample_feat('regs', max_samples=max_samples),
        'files': nodes.sample_feat('files', max_samples=max_samples)
    }

def get_emb(nodes, emb, ms=MAX_SAMPLES):
    data = sample(nodes, max_samples=ms)
    
    with torch.no_grad():
        emb.eval()
        zs = emb.forward(data)

    return zs 

def get_cutoff(graph, nodes, emb, disc):
    emb.eval()
    disc.eval()
    
    with torch.no_grad():
        zs = get_emb(nodes, emb)
        preds = disc.forward(zs, graph)

    return torch.quantile(preds, 0.999)

def emb_step(data, nodes, graph, emb, gen, disc, e_opt, g_opt, d_opt):
    # Positive samples & train embedder
    e_opt.zero_grad()
    
    # Improve stability
    emb.train()
    disc.eval()

    # Positive samples
    embs = emb.forward(data)
    t_preds = disc.forward(embs, graph)
    
    labels = torch.full(t_preds.size(), TRUE_VAL)
    loss = criterion(t_preds, labels)

    loss.backward()
    e_opt.step()

    return loss

def gen_step(z, data, nodes, graph, emb, gen, disc, e_opt, g_opt, d_opt):
    g_opt.zero_grad()
    
    emb.eval()
    disc.eval()
    gen.train()

    fake = gen(graph)
    g_loss = mse(fake, z)

    g_loss.backward()
    g_opt.step()
    
    return g_loss

def disc_step(z, data, nodes, graph, emb, gen, disc, e_opt, g_opt, d_opt):
    # Positive samples & train embedder
    d_opt.zero_grad()
    
    # Improve stability
    emb.eval()
    disc.train()
    gen.eval()

    # Positive samples
    t_preds = disc.forward(z, graph)

    # Negative samples
    fake = gen.forward(graph).detach()
    f_preds = disc.forward(fake, graph)

    # Random samples
    rand = torch.rand(graph.x.size(0), EMBED_SIZE)
    r_preds = disc.forward(rand, graph)

    # Calculate loss (combine true and false so only need
    # one call to backward)
    preds = torch.cat([t_preds, f_preds, r_preds], dim=1)
    labels = torch.cat([
        torch.full(t_preds.size(), TRUE_VAL),
        torch.full(f_preds.size(), FALSE_VAL),
        torch.full(r_preds.size(), FALSE_VAL)
    ], dim=1)

    d_loss = criterion(preds, labels)
    d_loss.backward()
    d_opt.step()

    return d_loss


def data_split(graphs, workers):
    '''
    Assign an equal number of train graphs to each worker process
    '''
    min_jobs = [len(graphs) // workers] * workers
    remainder = len(graphs) % workers 
    for i in range(remainder):
        min_jobs[i] += 1

    jobs = [0]
    for j in min_jobs:
        jobs.append(jobs[-1] + j)

    return jobs

def proc_job(rank, world_size, all_graphs, jobs, epochs=EPOCHS):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Sets number of threads used by this worker
    torch.set_num_threads(P_THREADS)

    # Load in training graphs
    my_graphs=[]; my_nodes=[]
    for gid in range(jobs[rank], jobs[rank+1]):
        with open(DATA_HOME + 'graph%d.pkl' % all_graphs[gid], 'rb') as f:
            my_graphs.append(pickle.load(f))
        with open(DATA_HOME + 'nodes%d.pkl' % all_graphs[gid], 'rb') as f:
            my_nodes.append(pickle.load(f))

    # Leader loads in test graph so we can log how
    # the model is doing
    if rank == 0:
        with open(TEST_HOME + 'graph201.pkl', 'rb') as f:
            test_graph = pickle.load(f)
        with open(TEST_HOME + 'nodes201.pkl', 'rb') as f:
            test_nodes = pickle.load(f)
        with open('testlog.txt', 'w+') as f:
            f.write('AUC\tAP\tE-loss\tD-loss\tG-loss\tBest\n')

    emb = NodeEmb(
        my_nodes[0].file_dim, 
        my_nodes[0].reg_dim,
        EMB_HIDDEN, EMB_OUT,
        EMBED_SIZE,
        t2v_dim=T2V,
        attn_kw=ATTN_KW
    )
    gen = NodeGen(my_graphs[0].x.size(1), STATIC_DIM, HIDDEN_GEN, EMBED_SIZE)
    disc = NodeDisc(EMBED_SIZE, HIDDEN_DISC)

    # Initialize shared models
    gen = DDP(gen)
    emb = DDP(emb)
    disc = DDP(disc)

    # Initialize optimizers
    e_opt = Adam(emb.parameters(), lr=EMB_LR, betas=(0.5, 0.999))
    d_opt = Adam(disc.parameters(), lr=DISC_LR, betas=(0.5, 0.999))
    g_opt = Adam(gen.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
    
    num_samples = len(my_graphs)

    # Validation criterion
    best = float('inf')
    new_best = False

    for e in range(epochs):
        for i in range(num_samples):
            st = time.time() 
            
            emb.train()
            gen.train() 
            disc.train()

            data = sample(my_nodes[i], max_samples=MAX_SAMPLES)

            # Makes for clunky method signatures, but I hate typing
            # all of this every time
            args = (
                data, my_nodes[i], my_graphs[i],
                emb, gen, disc,
                e_opt, g_opt, d_opt
            )

            z = get_emb(my_nodes[i], emb)
            
            d_loss = disc_step(z, *args)
            g_loss = gen_step(z, *args)
            e_loss = emb_step(*args)
   
            # Synchronize loss vals across workers
            dist.all_reduce(e_loss)
            dist.all_reduce(d_loss)
            dist.all_reduce(g_loss)

            if rank == 0:
                print(
                    "[%d-%d] Emb: %0.4f Disc: %0.4f, Gen: %0.4f (%0.2fs)" 
                    % (e, i, e_loss.item(), d_loss.item(), g_loss.item(), time.time()-st),
                    end=''
                )   

                # Performance seems somewhat tied to embedder loss
                # but especially when we're in a local minimum
                if e_loss.item() < best:
                    new_best = True
                    print("*")
                    best = e_loss.item()
                    torch.save(disc.module, 'saved_models/disc_0.pkl')
                    torch.save(gen.module, 'saved_models/gen_0.pkl')
                    torch.save(emb.module, 'saved_models/emb_0.pkl')  
                else:
                    print()

            # It's notoriously hard to validate GANs. Trying to see if theres
            # a good stopping point by looking at all this
            if rank == 0:
                torch.save(disc.module, 'saved_models/disc_1.pkl')
                torch.save(gen.module, 'saved_models/gen_1.pkl')
                torch.save(emb.module, 'saved_models/emb_1.pkl')

                test_z = get_emb(test_nodes, emb)
                auc,ap,pr = test_emb_input(test_z, test_nodes, test_graph, disc)

                print('AUC: ', auc, 'AP: ', ap)
                print("Top-k (Pr/Re):\n%s" % json.dumps(pr, indent=2))
                print()
                
                with open('testlog.txt', 'a+') as f:
                    f.write('%f\t%f\t%f\t%f\t%f\t%d\n' % (auc,ap,e_loss,d_loss,g_loss,new_best))
                
                new_best = False

            # Other workers wait for leader to finish validating before 
            # they move on (otherwise may be deadlock, etc.)
            dist.barrier()

    if rank == 0:
        # Last 
        test_z = get_emb(test_nodes, emb)
        auc,ap,pr = test_emb_input(test_z, test_nodes, test_graph, disc)

        # Best
        b_disc = torch.load('saved_models/disc_0.pkl')
        b_emb = torch.load('saved_models/emb_0.pkl')  

        test_z = get_emb(test_nodes, b_emb)
        b_auc,b_ap,b_pr = test_emb_input(test_z, test_nodes, test_graph, b_disc)
        
        with open('final.txt', 'a+') as f:
            f.write(
                '%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'
                % (b_auc, b_ap, b_pr[200][0], b_pr[200][1],
                    auc, ap, pr[200][0], pr[200][1],)
            )

    dist.barrier()
    
    # Cleanup
    if rank == 0:
        dist.destroy_process_group()


TRAIN_GRAPHS = [i for i in range(1,25)]
DATA_HOME = '../../../../code/NestedGraphs/inputs/benign/'
TEST_HOME = '../../../../code/NestedGraphs/inputs/mal/'
def main():
    world_size = min(N_JOBS, len(TRAIN_GRAPHS))
    jobs = data_split(TRAIN_GRAPHS, world_size)

    mp.spawn(proc_job,
        args=(world_size,TRAIN_GRAPHS,jobs,21),
        nprocs=world_size,
        join=True)

if __name__ == '__main__':
    with open('final.txt', 'w+') as f:
        f.write('Best AUC\tBest AP\tBest Pr\tBest Re\tLast AUC\tLast AP\tLast Pr\tLast Re\n')

    [main() for _ in range(10)]
