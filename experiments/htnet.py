import os
from typing import Optional, Sequence, List

import xarray as xr
import numpy as np
import pandas as pd
from tsl.datasets import PandasDataset
from tsl.ops.similarities import gaussian_kernel
from tsl.datasets.prototypes.mixin import MissingValuesMixin

class HtNet(PandasDataset, MissingValuesMixin):
    similarity_options = {'pearson'}
    temporal_aggregation_options = {'mean'}
    spatial_aggregation_options = {'mean'}
    def __init__(self, root=None, freq=None, norm = None, elec_excl = None,
                 train_excl = None, test_excl = None, name = None):
        self.norm = norm
        self.elec_excl = elec_excl
        self.train_excl = train_excl
        self.test_excl = test_excl
        self.name = name
        # Set root path
        self.root = root
        # load dataset
        df, pearson, mask, eval_mask = self.load()
        super().__init__(dataframe=df,
                         mask=mask,
                         attributes=dict(pearson=pearson),
                         freq=freq,
                         similarity_score="pearson",
                         temporal_aggregation="mean",
                         spatial_aggregation= "mean",
                         name="HtNet")
        # import pdb
        # pdb.set_trace()
        self.set_eval_mask(eval_mask)
    
    def load(self):
        path = os.path.join(self.root_dir, self.name)
         # having the 0 up front gets rid of the varibale dimension that isn't used
        pt = xr.open_dataset(path).to_array()[0, :, :, :]
        last_day = np.unique(pt.events)[-1]
        bool_mask_not_last_day = pt.events != last_day
        bool_mask_last_day = pt.events == last_day
        train_data = self.build_init_data(pt, bool_mask_not_last_day, self.train_excl)
        test_data = self.build_init_data(pt, bool_mask_last_day, self.test_excl)
        data = np.concatenate((train_data, test_data), axis=0)
        final_data = data.reshape(-1, data.shape[-1])
        df = pd.DataFrame(final_data)
        mask = ~np.isnan(df.values)
        total_train = train_data.shape[0] * train_data.shape[1]
        index_arr = np.arange(total_train, df.shape[0])
        eval_mask = np.zeros((df.shape[0],df.shape[1]),dtype=bool)
        eval_mask[index_arr] = [True for i in range(df.shape[1])]
        pearson = (df.corr()).to_numpy()
        # import pdb
        # pdb.set_trace()
        return df.astype('float32'), pearson, mask, eval_mask
    
    def build_init_data(self, pt, day_mask, inst_excl):
        # doing the :-1 gets rid of the labels which we're not using for now, but they can be incorporated
        orig_data = np.array(pt[day_mask][:, :-1, :])
        no_label_data = np.swapaxes(orig_data, 1, 2)  # switches the electrode and time step axis to be what model expects

        # TODO we standardize across the electrode dimension first before removing the bad instances and electrodes
        # TODO and therefore get this error occasionally "RuntimeWarning: invalid value encountered in divide"
        # TODO swap order if important
        std_data = np.subtract(np.swapaxes(no_label_data[:, self.norm:, :], 0, 1),
                                        np.mean(no_label_data[:, 0:self.norm, :], axis=1)) / np.std(no_label_data[:, 0:self.norm, :], axis=1)
        std_data = np.swapaxes(std_data, 0, 1)
        # get rid of bad electrodes
        good_elec_std_data = np.delete(std_data, self.elec_excl, axis=-1)
        # get rid of bad instances
        good_data = np.delete(good_elec_std_data, inst_excl, axis=0)
        return good_data

    def compute_similarity(self, method: str, **kwargs):
        if method == 'pearson':
            finite_pearson = self.pearson.reshape(-1)
            finite_pearson = finite_pearson[~np.isinf(finite_pearson)]
            sigma = finite_pearson.std()
            return gaussian_kernel(self.pearson, sigma)

# HtNet(root='experiments', norm = 250, elec_excl=[72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85], train_excl=[10, 83, 84, 86, 168, 209, 259, 260, 261, 262, 267, 268, 269, 270, 274, 289], test_excl=[38, 39, 82], name="EC02_ecog_data.nc")

# "normalization_amount": 250,
#         "elec_exclude": [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
#         "train_instance_exclude": [10, 83, 84, 86, 168, 209, 259, 260, 261, 262, 267, 268, 269, 270, 274, 289],
#         "test_instance_exclude": [38, 39, 82]