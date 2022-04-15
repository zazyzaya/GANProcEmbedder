import argparse
import pickle 
import torch 
import datetime as dt
from zoneinfo import ZoneInfo
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.metrics import precision_score, recall_score

from graph_utils import propogate_labels

HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'
MS = 50 
torch.set_num_threads(8)

fmt_ts = lambda x : dt.datetime.fromtimestamp(x).astimezone(
    ZoneInfo('Etc/GMT+4')
).isoformat()[11:-6]

def sample(nodes, max_samples=MS):
    return {
        'regs': nodes.sample_feat('regs', max_samples=max_samples),
        'files': nodes.sample_feat('files', max_samples=max_samples)
    }

def test_no_labels(nodes, graph, model_path=HOME+'saved_models/'):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    
    emb = torch.load(model_path+'emb.pkl')
    desc = torch.load(model_path+'desc.pkl')

    with torch.no_grad():
        data = sample(nodes)
        zs = emb(data)
        preds = desc(zs, graph.x, graph.edge_index)

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)

    with open(HOME+"predictions/preds%d.csv" % graph.gid, 'w+') as f:
        f.write('PID,anom_score\n')

        for i in range(vals.size(0)):
            outstr = '%s,%f,%s\n' % (
                inv_map[idx[i].item()],
                vals[i],
                fmt_ts(nodes[i].ts)
            )

            f.write(outstr)
            print(outstr, end='')

def test_emb(nodes, graph, model_str, dim, model_path=HOME+'saved_models/', verbose=True, max_samples=None):
    inv_map = {v:k for k,v in nodes.node_map.items()}
    labels = propogate_labels(graph,nodes)
    
    max_samples = max_samples if max_samples else MS
    emb = torch.load(model_path+'embedder/emb%s_%d.pkl' % (model_str, dim))
    desc = torch.load(model_path+'embedder/disc%s_%d.pkl' % (model_str, dim))

    with torch.no_grad():
        emb.eval()
        desc.eval()

        data = sample(nodes, max_samples=max_samples)
        zs = emb(data, graph)
        preds = torch.sigmoid(desc(zs, graph))

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx]

    auc_score = auc(labels.clamp(0,1), vals)
    ap_score = ap(labels.clamp(0,1), vals)

    if verbose:
        with open(HOME+"predictions/embedder/preds%d%s.csv" % (graph.gid, model_str), 'w+') as f:
            aucap = "AUC: %f\tAP: %f\n" % (auc_score, ap_score)
            f.write(aucap + '\n')

            for i in range(vals.size(0)):
                outstr = '%s\t%f\t%0.1f\n' % (
                    inv_map[idx[i].item()],
                    vals[i],
                    labels[i]
                )

                f.write(outstr)
                print(outstr, end='')
            
            print()
            print(aucap,end='')

    return auc_score, ap_score

def test_emb_input(zs, nodes, graph, disc):
    labels = propogate_labels(graph,nodes)
    
    with torch.no_grad():
        disc.eval()
        preds = torch.sigmoid(disc(zs, graph))

    vals, idx = torch.sort(preds.squeeze(-1), descending=True)
    labels = labels[idx].clamp(0,1)

    auc_score = auc(labels, vals)
    ap_score = ap(labels, vals)

    top_k = [50, 100, 150, 200, 250]
    pr = {}
    for k in top_k: 
        preds = torch.zeros(vals.size())
        preds[:k] =1.
        
        p = precision_score(labels, preds)
        r = recall_score(labels, preds)

        pr[k] = (p,r)

    return auc_score, ap_score, pr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hostname', '-n',
        default=201, type=int
    )
    parser.add_argument(
        '--model', '-m',
        default=''
    )
    parser.add_argument(
        '--dim', '-d',
        type=int, default=64
    )
    args = parser.parse_args()

    print("Testing host %04d with %s model" % (args.hostname, args.model))
    with open(HOME+'inputs/mal/graph%d.pkl' % args.hostname, 'rb') as f:
        graph = pickle.load(f)
    with open(HOME+'inputs/mal/nodes%d.pkl' % args.hostname, 'rb') as f:
        nodes = pickle.load(f)

    test_emb(nodes, graph, args.model, args.dim)

