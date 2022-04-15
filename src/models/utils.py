import torch 
from torch import nn 
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

def repack(data, x):
    '''
    Takes manipulated data `data` and puts it back into 
    a packed sequence with metadata matching `x`
    '''
    return PackedSequence(
        data, 
        batch_sizes=x.batch_sizes, 
        sorted_indices=x.sorted_indices, 
        unsorted_indices=x.unsorted_indices
    )

def packed_fn(x, fn):
    '''
    Helper fn to manipulate packed sequences (torch, pls fix this)
    '''
    return PackedSequence(
        fn(x.data), 
        batch_sizes=x.batch_sizes, 
        sorted_indices=x.sorted_indices, 
        unsorted_indices=x.unsorted_indices
    )

def packed_cat(seq, dim=1):
    '''
    Concatonates packed sequences
    '''
    x = seq[0]
    catted = torch.cat([packed.data for packed in seq], dim=dim)
    return PackedSequence(
        catted, 
        batch_sizes=x.batch_sizes, 
        sorted_indices=x.sorted_indices, 
        unsorted_indices=x.unsorted_indices
    )

def get_last_vectors_padded(seq, batch_sizes):
    '''
    Selects the vector at each time step in seq at 
    sequence position batch_size. Useful for grabbing the
    last vector in a padded sequence

    Inputs: 
        [T, B, *] tensor (seq len, batch size, dim)
        [B] tensor of indexes to get

    Output: 
        [B, *] tensor of vectors s.t. seq[i] returns batch_sizes[i]
    '''
    # I made this a lot harder than it had to be initially.. but it's
    # still annoying to type this every time. This is more concise
    return seq[torch.arange(seq.size(0)), batch_sizes-1]


def get_last_vectors_unpadded(seq):
    '''
    Returns the last value for each element in a PackedSequence 
    object (not padded!)
    '''
    idxs = []
    i = -1
    num_found = 0
    end = seq.data.size(0)
    
    while len(idxs) < seq.sorted_indices.size(0):
        start = end-seq.batch_sizes[i]
        idx_start = start + num_found
        found = seq.sorted_indices[num_found:seq.batch_sizes[i]]
        
        if found.size():
            idxs += list(range(idx_start, end))
            num_found += len(found)
        
        end = start
        i -= 1
    
    mapped_indices = torch.zeros(seq.sorted_indices.size(0), dtype=torch.int64)
    mapped_indices[seq.sorted_indices] = torch.tensor(idxs)
    return seq.data[mapped_indices]


def packed_aggr(seq, aggr='mean'):
    seq,batch = pad_packed_sequence(seq)
    
    if aggr == 'mean' or aggr == 'avg':
        return seq.sum(dim=0)/batch.unsqueeze(1)
    else:
        raise NotImplementedError('Sorry, havent written the %s function yet' % aggr)

def masked_softmax(A, dim):
    # matrix A is the one you want to do mask softmax at dim=1
    A_max = torch.max(A,dim=dim,keepdim=True)[0]
    A_exp = torch.exp(A-A_max)
    A_exp = A_exp * (A != 0).type(torch.FloatTensor) # this step masks
    return A_exp / (torch.sum(A_exp,dim=dim,keepdim=True)+10**-6)

def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

def kld_gauss(mean_in, std_in):
    num_nodes = mean_in.size(0)
    std_log = torch.log(std_in + 1e-6)
    kld_element =  torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                        torch.pow(torch.exp(std_log), 2), 1))
    return (-0.5 / num_nodes) * kld_element