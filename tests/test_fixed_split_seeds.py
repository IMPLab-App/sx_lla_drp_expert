import contextlib
import hashlib
import io
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from semilearn.datasets.cv_datasets.raw_signal import DATASET_FILENAME,get_raw_signal,_stratified_split


class FixedSplitSeedTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.labels=np.repeat(np.arange(9),20)
        values=np.arange(180*1024,dtype=np.float32).reshape(180,1024)/1000
        self.names=np.asarray([f'C{c}' for c in range(9)])
        np.savez(self.root/DATASET_FILENAME,data=values,labels=self.labels,class_names=self.names)
        self.base=dict(seed=2,num_classes=9,full_train_counts=[10]*9,labeled_counts=[3]*9,
            noise_ratio=0,include_lb_to_ulb=False,noise_type='asym')
        self.legacy=self.load(self.base)
        split=_stratified_split(self.labels,9,2)
        self.frozen=self.root/'split.npz'
        np.savez_compressed(self.frozen,train_pool=split['train'],class_names=self.names,
            **{key:ds.sample_indices for key,ds in zip(('labeled','unlabeled','validation','test'),self.legacy[:4])})

    def tearDown(self):self.temporary.cleanup()

    def load(self,args):
        with contextlib.redirect_stdout(io.StringIO()):
            return get_raw_signal(SimpleNamespace(**args),data_dir=self.root,return_test=True)

    def config(self,seed):
        return dict(self.base,seed=seed,train_seed=seed,split_seed=2,fixed_split_file=str(self.frozen),
            fixed_split_sha256=hashlib.sha256(self.frozen.read_bytes()).hexdigest())

    def test_five_train_seeds_preserve_data_and_normalization(self):
        for seed in (2,3,4,5,6):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
            result=self.load(self.config(seed))
            for old,new in zip(self.legacy[:4],result[:4]):
                np.testing.assert_array_equal(old.sample_indices,new.sample_indices)
                np.testing.assert_array_equal(old.data,new.data)
                self.assertEqual((old.weak_transform.mean,old.weak_transform.std),(new.weak_transform.mean,new.weak_transform.std))
            self.assertEqual(random.getstate(),state[0])
            np.testing.assert_array_equal(np.random.get_state()[1],state[1][1])
            self.assertTrue(torch.equal(torch.get_rng_state(),state[2]))

    def test_explicit_split_seed_without_file_preserves_legacy(self):
        for seed in (3,6):
            result=self.load(dict(self.base,seed=seed,split_seed=2))
            for old,new in zip(self.legacy[:4],result[:4]):np.testing.assert_array_equal(old.sample_indices,new.sample_indices)

    def test_legacy_seed_still_controls_split_when_not_opted_in(self):
        other=self.load(dict(self.base,seed=3))
        self.assertFalse(np.array_equal(other[3].sample_indices,self.legacy[3].sample_indices))

    def test_tamper_or_wrong_split_seed_is_rejected(self):
        cfg=self.config(4);cfg['fixed_split_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'SHA256'):self.load(cfg)
        cfg=self.config(4);cfg['split_seed']=3
        with self.assertRaisesRegex(ValueError,'split_seed'):self.load(cfg)

    def test_overlap_is_rejected(self):
        with np.load(self.frozen,allow_pickle=False) as data:saved={k:data[k].copy() for k in data.files}
        saved['unlabeled'][0]=saved['labeled'][0]
        np.savez_compressed(self.frozen,**saved)
        with self.assertRaisesRegex(ValueError,'overlap'):self.load(self.config(4))

    def test_train_seed_alias_resolves_runtime_seed(self):
        from train import get_config
        previous=sys.argv
        try:
            sys.argv=['train.py','--seed','2','--train_seed','6','--split_seed','2']
            args=get_config()
        finally:sys.argv=previous
        self.assertEqual((args.seed,args.train_seed,args.split_seed),(6,6,2))

    def test_sampler_seed_changes_order_without_changing_membership(self):
        from semilearn.datasets.samplers import DistributedSampler
        orders=[]
        for seed in (2,3,4,5,6):
            sampler=DistributedSampler(list(range(37)),num_replicas=1,rank=0,num_samples=74,seed=seed)
            sampler.set_epoch(4);first=list(sampler)
            self.assertEqual(first,list(sampler))
            self.assertEqual(sorted(first),sorted(list(range(37))*2))
            orders.append(first)
        self.assertEqual(len({tuple(x) for x in orders}),5)
        legacy=DistributedSampler(list(range(37)),num_replicas=1,rank=0,num_samples=74)
        legacy.set_epoch(4)
        generator=torch.Generator().manual_seed(4)
        expected=torch.cat([torch.randperm(37,generator=generator),torch.randperm(37,generator=generator)]).tolist()
        self.assertEqual(list(legacy),expected)


if __name__=='__main__':unittest.main()
