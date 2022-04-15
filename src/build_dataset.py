import pickle as pkl
from preprocessing.build_hostgraph import build_graphs

# Benign
bids = list(range(1,26))
benign = build_graphs(bids)
for i in range(len(bids)):
    g,n = benign[i]
    id = bids[i]

    with open('inputs/benign/graph%d.pkl' % id, 'wb+') as f:
        pkl.dump(g, f)
    with open('inputs/benign/nodes%d.pkl' % id, 'wb+') as f:
        pkl.dump(n, f)

# Mal
mids = [201,402,660,104,205,321,255,355,503,462,559,419,609,771,955,874]
malicious = build_graphs(mids)
for i in range(len(mids)):
    g,n = malicious[i]
    id = mids[i]

    with open('inputs/mal/graph%d.pkl' % id, 'wb+') as f:
        pkl.dump(g, f)
    with open('inputs/mal/nodes%d.pkl' % id, 'wb+') as f:
        pkl.dump(n, f)