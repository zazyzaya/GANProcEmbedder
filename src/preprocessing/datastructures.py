from argparse import ArgumentError
import torch 
from torch.nn.utils.rnn import pack_sequence
from torch_geometric.data import Data 
        
class DynamicFeatures():
    '''
    Object holding dynamic features. Should never be accessed
    by the user. If it is, I have done something terribly wrong
    in writing the other classes--please let me know.
    '''
    def __init__(self):
        self.feats = [] 
        self.times = []

    def __len__(self):
        self.defrag()
        
        if len(self.feats):
            return self.feats[0].size(0)
        return 0

    # Avoid stacking tensors every time
    def get_data(self, s_time=None, e_time=None):
        if len(self.feats) == 0:
            return torch.tensor([]), torch.tensor([])

        self.defrag()
        if s_time is None and e_time is None:
            return self.times[0], self.feats[0]
        
        # If desired, only access feats observed up until `time`
        if e_time is None:
            indices = (s_time <= self.times[0]).squeeze()
        elif s_time is None:
            indices = (self.times[0] <= e_time).squeeze()
        else:
            indices = (s_time <= self.times[0] and self.times[0] <= e_time).squeeze()
        
        return self.times[0][indices], self.feats[0][indices]

    # Just mainitain a list until accessed
    def add_data(self, time: float, feat: torch.Tensor) -> None:
        self.feats.append(feat)
        self.times.append(torch.tensor([time]))

    # Make sure all tensors are stacked
    def defrag(self):
        if len(self.feats) > 1:
            # If this is not the first defrag
            if self.feats[0].dim() == 2:
                self.feats = [torch.cat([self.feats[0], torch.stack(self.feats[1:])], dim=0)]
                self.times = [torch.cat([self.times[0], torch.stack(self.times[1:])], dim=0)]
            else: 
                self.feats = [torch.stack(self.feats)]
                self.times = [torch.stack(self.times)]
        
        if len(self.feats) == 1 and self.feats[0].dim() == 1:
            self.feats = [self.feats[0].unsqueeze(0)]
            self.times = [self.times[0].unsqueeze(0)]

class Node():
    '''
    Object holding node features
    '''
    def __init__(self, nid, first_occurence):
        self.nid = nid
        self.ts = first_occurence
    
        self.files = DynamicFeatures()
        self.regs = DynamicFeatures()

    def __len__(self):
        return len(self.files), len(self.regs), len(self.mods)

    # Getters
    def get_files(self, s=None,e=None):
        return self.files.get_data(s,e)
    def get_regs(self,  s=None,e=None):
        return self.regs.get_data(s,e)
    
    def getter(self, fn,  s=None,e=None):
        '''
        Gets arbitrary value specified by fn 
        '''
        if fn == 'files': 
            return self.files.get_data(s,e)
        elif fn == 'regs': 
            return self.regs.get_data(s,e)
        else:
            raise ArgumentError("fn must be in ['files', 'regs', 'mods']")

    # Setters
    def add_file(self, ts, file):
        self.files.add_data(ts, file)
    def add_reg(self, ts, reg):
        self.regs.add_data(ts, reg)

    # Clean up in preparation for analysis
    def finalize(self):
        self.files.defrag()
        self.regs.defrag()


class NodeList():
    '''
    Object holding dynamic nodes (no edges)
    '''
    def __init__(self):
        # Give procs unique IDs starting at 0
        self.node_map = dict() 
        self.num_nodes = 0
        self.nodes = []

        self.file_dim = None 
        self.reg_dim = None 


    def __getitem__(self, idx):
        if type(idx) == int:
            return self.nodes[idx]
        else:
            return self.nodes[self.node_map[idx]]

    def __len__(self):
        return len(self.nodes)

    def __get_empty(self, fn):
        '''
        Account for procs which have no associated activities
        of a certain kind (otherwise the RNN output is indexed wrong)
        '''
        if fn == 'files': 
            return torch.zeros(1,self.file_dim)
        elif fn == 'regs':
            return torch.zeros(1,self.reg_dim)

    def add_node(self, ts, pid):
        # Some logs are out of order. Make sure it's the most
        # recent one. 
        if pid in self.node_map:
            if self.nodes[self.node_map[pid]].ts > ts:
                self.nodes[self.node_map[pid]].ts = ts 

            return False

        if '-1' in pid:
            return False

        self.node_map[pid] = self.num_nodes 
        self.nodes.append(Node(self.num_nodes, ts))
        self.num_nodes += 1
        return True

    # Add features to nodes
    def add_file(self, ts, pid, file):
        if '-1' in pid:
            return 

        if self.file_dim is None:
            self.file_dim = file.size(-1)
        if pid in self.node_map:
            self.nodes[self.node_map[pid]].add_file(ts, file)

    def add_reg(self, ts, pid, reg):
        if '-1' in pid:
            return 

        if self.reg_dim is None:
            self.reg_dim = reg.size(-1)
        if pid in self.node_map:
            self.nodes[self.node_map[pid]].add_reg(ts, reg)

    def finalize(self):
        [n.finalize() for n in self.nodes]

    def sample(self, batch=[], s=None, e=None, norm_time=True, max_samples=0):
        if not len(batch):
            batch = list(range(self.num_nodes))
        
        return {
            'files': self.sample_feat('files', batch=batch, s=s, e=e, norm_time=norm_time, max_samples=max_samples),
            'regs': self.sample_feat('regs', batch=batch, s=s, e=e, norm_time=norm_time, max_samples=max_samples)
        }

    def sample_feat(self, fn, batch=[], s=None, e=None, norm_time=True, max_samples=0):
        times, feats = [],[]

        if not len(batch):
            batch = list(range(self.num_nodes))

        for b in batch:
            t,x = self.nodes[b].getter(fn,s,e)

            if norm_time and x.size(0):
                node_time = self.nodes[b].ts / 1000
            else:
                node_time = 0.
            
            # Avoid errors from empty tensors
            if x.size(0):
                if max_samples > 0 and x.size(0) > max_samples:
                    perm = torch.randperm(x.size(0))[:max_samples]
                    times.append(t[perm]-node_time)
                    feats.append(x[perm])
                else:
                    times.append(t-node_time)
                    feats.append(x)

            else:
                times.append(torch.zeros(1,1))
                feats.append(self.__get_empty(fn))


        x = pack_sequence(feats, enforce_sorted=False)
        t = pack_sequence(times, enforce_sorted=False)

        return t,x
        

class HostGraph(Data):
    def __init__(self, gid, **kwargs):
        super().__init__(**kwargs)
        '''
        Object holding only edges, and intransient node features
        '''
        # Unique identifier
        self.gid = gid 

        # Will be filled and converted to tensors eventually
        self.src, self.dst = [],[] # To be converted to edge_index
        self.edge_attr = []
        self.x = []
        self.node_times = []

        # Turns graph write-only after everything is built
        self.ready = False

    def add_node(self, ts, pid, feat, nodelist):
        assert not self.ready, 'add_node undefined after self.finalize() has been called'
        
        if '-1' in pid:
            return 

        if nodelist.add_node(ts, pid):
            self.x.append(feat)
            self.node_times.append(ts)

    def add_edge(self, ts, pid, ppid, feat, pfeat, nodelist):
        assert not self.ready, 'add_edge undefined after self.finalize() has been called'

        self.add_node(ts, ppid, pfeat, nodelist)
        self.add_node(ts, pid, feat, nodelist)

        if '-1' in pid or '-1' in ppid:
            return

        self.src.append(nodelist.node_map[ppid])
        self.dst.append(nodelist.node_map[pid])
        self.edge_attr.append(ts)

    
    def finalize(self):
        '''
        Convert all lists being constructed during processing
        into tensors to be used in later processing
        '''
        # Preserve idempotence
        if self.ready:
            return 

        self.ready = True 
        self.edge_index = torch.tensor([self.src, self.dst])
        self.x = torch.stack(self.x)
        self.num_nodes = self.x.size(0)
        self.edge_attr = torch.tensor(self.edge_attr)
        self.node_times = torch.tensor(self.node_times)