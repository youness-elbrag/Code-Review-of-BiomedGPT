import unittest
import torch
import torch.nn as nn
from specific_file.dropath import drop_path, DropPath

"""
Test Drop Path Function and Module:

    Test Cases:
    - Initialize the tensor with 3D dimensions [B, C, H, D] => Sample input tensor
    - Assert both input and output tensors have the same shape
    - For the function:
        - Test drop_prob = 0.0 (no drop), verify no elements are dropped
        - Test drop_prob = 0.5 (50% drop), verify some elements are dropped
    - For the module:
        - Test drop_prob = 0.0 (no drop), verify no elements are dropped
        - Test drop_prob = 0.5 (50% drop), verify some elements are dropped

Note: Replace 'blocks_Model' with the actual name of the module where the 'drop_path' function and 'DropPath' class are defined.
"""


class TestDropPathFunction(unittest.TestCase):

    def test_drop_path_no_drop(self):

        x = torch.randn(16, 32, 32)  
        output = drop_path(x, drop_prob=0.0, training=True)
        self.assertTrue(torch.all(torch.eq(x, output)))  

    def test_drop_path_drop(self):
        x = torch.randn(16, 32, 32)  
        output = drop_path(x, drop_prob=0.5, training=True)
        self.assertFalse(torch.all(torch.eq(x, output)))  

class TestDropPathModule(unittest.TestCase):

    def test_drop_path_module_no_drop(self):
        x = torch.randn(16, 32, 32)  
        drop_path_layer = DropPath(drop_prob=0.0)
        output = drop_path_layer(x)
        self.assertTrue(torch.all(torch.eq(x, output)))  

    def test_drop_path_module_drop(self):
        x = torch.randn(16, 32, 32) 
        drop_path_layer = DropPath(drop_prob=0.5)
        output = drop_path_layer(x)
        self.assertFalse(torch.all(torch.eq(x, output))) 

if __name__ == '__main__':
    unittest.main()
