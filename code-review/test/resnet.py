import unittest
import torch
import torch.nn as nn
from blocks_Model.Resnet import ResNet

class TestResNet(unittest.TestCase):

    def test_resnet_forward(self):
        """
        Test the forward pass of the ResNet model.

        Steps:
        1. Initialize a ResNet model with a given layer configuration.
        2. Generate a random input tensor.
        3. Pass the input tensor through the model's forward pass.
        4. Verify that the output tensor has the expected shape.

        This test ensures that the ResNet model processes input data correctly and produces
        output of the expected shape.
        """
        layers = [2, 2, 2, 2]  
        resnet_model = ResNet(layers)
        input_tensor = torch.randn(4, 3, 224, 224)
        output = resnet_model(input_tensor)
        expected_shape = (4, 256, 7, 7)  
        self.assertEqual(output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
