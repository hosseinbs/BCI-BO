#
# This code is adapted from jasper Snoek's Bayesian optimization code
#

##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
# 
# This code is written for research and educational purposes only to 
# supplement the paper entitled
# "Practical Bayesian Optimization of Machine Learning Algorithms"
# by Snoek, Larochelle and Adams
# Advances in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy        as np
import numpy.random as npr
import scipy.stats  as sps
import sklearn.ensemble
import sklearn.ensemble.forest
import util

from sklearn.externals.joblib import Parallel, delayed

def init(expt_dir, arg_string, subject):
    args = util.unpack_args(arg_string)
    return RandomForestEIChooser(**args)

class RandomForestRegressorWithVariance(sklearn.ensemble.RandomForestRegressor):

    def predict(self,X):
        # Check data
        X = np.atleast_2d(X)

        all_y_hat = [ tree.predict(X) for tree in self.estimators_ ]

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators
        y_var = np.var(all_y_hat,axis=0,ddof=1)

        return y_hat, y_var

class RandomForestEIChooser:

    def __init__(self,n_trees=50,
                 max_depth=None,
                 min_samples_split =2,
                 max_features="auto",
                 n_jobs=1,
                 random_state=None):
        self.n_trees = float(n_trees)
        self.max_depth = max_depth
        self.min_split = min_samples_split 
        self.max_features = max_features
        self.n_jobs = float(n_jobs)
        self.random_state = random_state
        self.rf = RandomForestRegressorWithVariance(n_estimators=n_trees, 
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    max_features=max_features,
                                                    n_jobs=n_jobs,
                                                    random_state=random_state)
        
    def next(self, grid, values, durations,
             candidates, pending, complete):
                # Grab out the relevant sets.

        # Don't bother using fancy RF stuff at first.
        if complete.shape[0] < 2:
            return int(candidates[0])

        # Grab out the relevant sets.
        comp = grid[complete,:]
        cand = grid[candidates,:]
        pend = grid[pending,:]
        vals = values[complete]

        self.rf.fit(comp,vals)

        if pend.shape[0] != 0:
            # Generate fantasies for pending
            func_m, func_v = self.rf.predict(pend)
            vals_pend = func_m + np.sqrt(func_v) + npr.randn(func_m.shape[0])
            
            # Re-fit using fantasies
            self.rf.fit(np.vstack[comp,pend],np.hstack[vals,vals_pend])            
            
        # Predict the marginal means and variances at candidates.
        func_m, func_v = self.rf.predict(cand)

        # Current best.
        best = np.min(vals)

        # Expected improvement
        func_s = np.sqrt(func_v) + 0.0001
        u      = (best - func_m) / func_s
        ncdf   = sps.norm.cdf(u)
        npdf   = sps.norm.pdf(u)
        ei     = func_s*( u*ncdf + npdf)
        
        best_cand = np.argmax(ei)
        ei.sort()

        return int(candidates[best_cand])
