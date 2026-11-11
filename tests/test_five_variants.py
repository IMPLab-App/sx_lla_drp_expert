"""Estimator ablations must not silently evaluate their disabled mechanisms."""
import unittest
import inspect
from unittest.mock import patch
import torch

from semilearn.imb_algorithms.metaexpert.five_variants import DistributionController,FiveVariantMetaExpert,SPECS,l1_distance
from semilearn.imb_algorithms.metaexpert.cola_ddde import normalize_distribution,symmetric_kl,tau_from_kl


class VariantMathTest(unittest.TestCase):
    def test_explicit_signature_preserves_framework_batch_dispatch(self):
        self.assertEqual(list(inspect.signature(FiveVariantMetaExpert.train_step).parameters),
            ['self','x_lb','y_lb','y_lb_noised','x_ulb_w','x_ulb_s','y_ulb'])

    def setUp(self):
        torch.set_num_threads(2)
        self.prior=torch.tensor([35,28,23,19,15,12,10,8,7],dtype=torch.float)
        self.labels=torch.tensor([0,0,8])
        self.logits=torch.full((3,9),-20.).scatter_(1,self.labels[:,None],20.)
        self.features=torch.tensor([[1.,2.,0.],[2.,0.,1.],[0.,1.,3.]])

    def test_v2_never_calls_symmetric_kl(self):
        c=DistributionController(self.prior,'V2',est_epoch=0);c.begin_epoch(0)
        with patch('semilearn.imb_algorithms.metaexpert.five_variants.symmetric_kl',side_effect=AssertionError('KL enabled')):
            d=c.observe(self.logits,self.features);c.begin_epoch(1)
        self.assertTrue(torch.isfinite(d).all())
        self.assertTrue(0<=float(c.tau_dyn)<=4)
        self.assertIn('L1_uni',c.snapshot());self.assertNotIn('KL_uni',c.snapshot())

    def test_v3_histogram_ema_and_no_effective_rank(self):
        c=DistributionController(self.prior,'V3',est_epoch=0,ema_u=.9);c.begin_epoch(0)
        with patch('semilearn.imb_algorithms.metaexpert.five_variants.effective_rank',side_effect=AssertionError('DDDE enabled')):
            d=c.observe(self.logits,self.features)
        raw=torch.zeros(9);raw[0]=2/3;raw[8]=1/3
        torch.testing.assert_close(c.pi_ddde,raw)
        torch.testing.assert_close(c.pi_u,normalize_distribution(torch.ones(9)/9*.9+raw*.1))
        expected=tau_from_kl(*d,4);c.begin_epoch(1)
        torch.testing.assert_close(c.tau_dyn,expected)
        self.assertIn('pi_histogram',c.snapshot());self.assertNotIn('pi_DDDE',c.snapshot())

    def test_no_confident_predictions_keep_previous_tau(self):
        for v in ('V2','V3'):
            c=DistributionController(self.prior,v,est_epoch=0);c.begin_epoch(0)
            self.assertIsNone(c.observe(torch.zeros(3,9),self.features))
            c.begin_epoch(1);self.assertEqual(float(c.tau_dyn),4.)

    def test_warmup_does_not_collect(self):
        for v in ('V2','V3'):
            c=DistributionController(self.prior,v,est_epoch=18);c.begin_epoch(17)
            self.assertIsNone(c.observe(self.logits,self.features));self.assertEqual(int(c.total_valid_batches),0)

    def test_checkpoint_reproduces_pending_distance_window(self):
        for v in ('V2','V3'):
            a=DistributionController(self.prior,v,est_epoch=0);a.begin_epoch(0);a.observe(self.logits,self.features)
            b=DistributionController(self.prior,v,est_epoch=0);b.load_state_dict(a.state_dict())
            a.begin_epoch(1);b.begin_epoch(1)
            for k,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[k],rtol=0,atol=0)

    def test_fixed_versions_have_no_estimator_or_distance(self):
        for v,t in [('V1',4),('V4',2),('S0',0),('S1',1),('S3',3)]:
            self.assertEqual(SPECS[v],dict(estimator='none',distance='none',dynamic=False,fixed_tau=float(t)))

    def test_l1_is_full_l1_norm(self):
        p=torch.tensor([.75,.25]);q=torch.tensor([.25,.75])
        self.assertAlmostEqual(float(l1_distance(p,q)),1.)
        self.assertEqual(float(l1_distance(p,p)),0.)


if __name__=='__main__':unittest.main()
