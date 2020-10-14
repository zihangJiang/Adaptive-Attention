# coding=utf-8
import numpy as np
import torch


class ReweighterBatchSampler(object):
    '''
    ReweighterBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  1 samples
    for 'classes_per_it' random classes and batch_size of extra query.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations, sample_per_class = 1,regular = False):
        '''
        Initialize the ReweighterBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for query
        - iterations: number of iterations (episodes) per epoch
        - sample_per_class: support sample per class (default 1-shot)
        - regular: sample same number of query for each class (default False)
        '''
        super(ReweighterBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.num_sample_query = num_samples
        self.iterations = iterations
        self.sample_per_class = sample_per_class
        self.regular = regular
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        with 1 sample per class and a nsq of query
        total len = classes_num + query_batch_size
        '''
        nsq = self.num_sample_query
        cpi = self.classes_per_it
        spc = self.sample_per_class
        if self.regular:
            assert nsq%cpi==0
            nsqpc = nsq//cpi
        for it in range(self.iterations):
            batch_size = nsq + cpi*spc
            batch = torch.LongTensor(batch_size)

            # select cpi class index
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            if self.regular:
                query = torch.LongTensor(nsq)
            else:
                # collect all examples belong to the class chosen above
                #whole = torch.LongTensor([i for i in self.indexes[c_idxs].view(-1) if not np.isnan(i)])
                whole = self.indexes[c_idxs].view(-1)
                whole = whole[(torch.isnan(whole)-1).nonzero().view(-1)]
                # select one sample per class
                rest = torch.zeros_like(whole)
                length = 0
            for i, c in enumerate(self.classes[c_idxs]):
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                temp_idxs = torch.randperm(self.numel_per_class[label_idx])
                sample_idxs = temp_idxs[:spc]
                
                if self.regular:
                    query_idx = temp_idxs[spc:spc + nsqpc]
                    query[i*nsqpc: i*nsqpc+nsqpc] = self.indexes[label_idx][query_idx]
                else:
                    non_sample_idx = temp_idxs[spc:]
                    rest[length:length+self.numel_per_class[label_idx]-spc] = self.indexes[label_idx][non_sample_idx]
                    length = length+self.numel_per_class[label_idx]-spc
                batch[i * spc : i * spc + spc] = self.indexes[label_idx][sample_idxs]

            # select batch of samples for query from the rest of the data within the class
            if self.regular:
                batch[cpi*spc:] = query[torch.randperm(len(query))]
            else:
                rest = rest[:length]
                batch[cpi*spc:] = rest[torch.randperm(len(rest))][:nsq]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations

class SpecialReweighterBatchSampler(ReweighterBatchSampler):
    '''
    SpecialReweighterBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'sample_per_class' samples
    for 'classes_per_it' random classes and batch_size of extra query.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations, sample_per_class = 1, augment = 3):
        '''
        Initialize the SpecialReweighterBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for query
        - iterations: number of iterations (episodes) per epoch
        '''
        super(SpecialReweighterBatchSampler, self).__init__(labels, classes_per_it, num_samples, iterations, sample_per_class = sample_per_class)
        self.augment = augment
        assert sample_per_class%augment==0
        self.sample_per_class = int(sample_per_class/augment)

    def __iter__(self):
        '''
        yield a batch of indexes
        with 1 sample per class and a nsq of query
        total len = classes_num + query_batch_size
        '''
        nsq = self.num_sample_query
        cpi = self.classes_per_it
        spc = self.sample_per_class
        aug = self.augment
        if self.regular:
            assert nsq%cpi==0
            nsqpc = nsq//cpi
        for it in range(self.iterations):
            batch_size = nsq + cpi*spc*aug
            batch = torch.LongTensor(batch_size)
            reference_size = cpi*spc
            reference = torch.LongTensor(reference_size)
            # select cpi class index
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            if self.regular:
                query = torch.LongTensor(nsq)
            else:
                # collect all examples belong to the class chosen above
                whole = self.indexes[c_idxs].view(-1)
                whole = whole[(torch.isnan(whole)-1).nonzero().view(-1)]
                # select one sample per class
                rest = torch.zeros_like(whole)
                length = 0
            for i, c in enumerate(self.classes[c_idxs]):
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                temp_idxs = torch.randperm(self.numel_per_class[label_idx])
                sample_idxs = temp_idxs[:spc]
                
                reference[i * spc : i * spc + spc] = self.indexes[label_idx][sample_idxs]
                if self.regular:
                    query_idx = temp_idxs[spc:spc + nsqpc]
                    query[i*nsqpc: i*nsqpc+nsqpc] = self.indexes[label_idx][query_idx]
                else:
                    non_sample_idx = temp_idxs[spc:]
                    rest[length:length+self.numel_per_class[label_idx]-spc] = self.indexes[label_idx][non_sample_idx]
                    length = length+self.numel_per_class[label_idx]-spc


            batch[:cpi*spc*aug] = reference.repeat(self.augment,1).transpose(0,1).contiguous().view(-1)
            if self.regular:
                batch[cpi*spc:] = query[torch.randperm(len(query))]
            else:
                batch[cpi*spc*aug:] = rest[torch.randperm(len(rest))][:nsq]
                rest = rest[:length]
            yield batch