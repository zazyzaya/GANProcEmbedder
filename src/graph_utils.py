import torch 
HOME = '/mnt/raid0_24TB/isaiah/code/NestedGraphs/'

def propogate_labels(g, nodes, label_f=HOME+'inputs/anoms.csv'):
    with open(label_f, 'r') as f:
        labels = f.read()

    anoms = dict() 
    for line in labels.split('\n')[1:]:
        host,pid = line.split(',')
        host = int(host)

        if host not in anoms:
            anoms[host] = []
        
        anoms[host].append(pid)
    
    labels = torch.zeros(g.num_nodes)
    if g.gid not in anoms:
        return labels 

    # Assume that anomalous processes produce other anomalous processes
    # Luckilly, the graph structure is a tree so we (shouldnt) have to 
    # check for loops and can just do BFS to assign labels 
    domain = set([nodes.node_map[n] for n in anoms[g.gid]])
    for d in domain:
        labels[d] = 1

    while domain:
        mal = domain.pop()
        children = g.edge_index[1][g.edge_index[0] == mal]

        for c in children:
            # Troubling that there are loops in here..
            if labels[c] == 0:
                domain.add(c.item())
                labels[c] = labels[mal]+1

    #labels[labels != 0] = 1/labels[labels!=0]
    return labels
