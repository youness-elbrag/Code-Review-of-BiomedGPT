import unittest
import torch
from torch import nn
from blocks_Model.FrozenBatchNorm2d import FrozenBatchNorm2d  

class TestFrozenBatchNorm2d(unittest.TestCase):

    def test_forward_pass(self):
        num_features = 64
        eps = 1e-5
        batch_size = 8
        height, width = 32, 32
        x = torch.randn(batch_size, num_features, height, width)

        bn = FrozenBatchNorm2d(num_features, eps)
        bn_output = bn(x)

        self.assertEqual(bn_output.shape, x.shape)

    def test_state_dict_loading(self):
        num_features = 64
        eps = 1e-5
        bn = FrozenBatchNorm2d(num_features, eps)

        """ Create a dummy state_dict with 
        running_mean and running_var
        
        """
        dummy_state_dict = {
            'running_mean': torch.randn(num_features),
            'running_var': torch.rand(num_features),
            'weight': torch.ones(num_features),
            'bias': torch.zeros(num_features)
        }

        # Call _load_from_state_dict to adjust state keys based on version information
        bn._load_from_state_dict(dummy_state_dict, 'module.', {}, True, [], [], [])

        self.assertTrue('module.running_mean' in dummy_state_dict)
        self.assertTrue('module.running_var' in dummy_state_dict)

    def test_repr(self):
        num_features = 64
        eps = 1e-5
        bn = FrozenBatchNorm2d(num_features, eps)
        repr_str = repr(bn)

        self.assertIn("FrozenBatchNorm2d", repr_str)
        self.assertIn(str(num_features), repr_str)
        self.assertIn(str(eps), repr_str)

if __name__ == '__main__':
    unittest.main()
