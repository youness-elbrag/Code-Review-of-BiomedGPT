import unittest
import torch
from blocks_Model.PromptEncoder import PromptEncoder  # Import your PromptEncoder class from the module

class TestPromptEncoder(unittest.TestCase):

    def test_prefix_encoding(self):
        type = "prefix"
        length = 100
        projection = True
        embed_dim = 128
        proj_dim = 64
        layers = 3
        vocab_size = 1000

        encoder = PromptEncoder(type, length, projection, embed_dim, proj_dim, layers, vocab_size)

        prefix_input = torch.tensor([1, 2, 3, 4, 5])  # Replace with your prefix tensor
        encoded_output = encoder(prefix_input)

        # Assert the shape of the encoded output
        expected_shape = (prefix_input.size(0), layers * 2 * embed_dim)
        self.assertEqual(encoded_output.shape, expected_shape)

    def test_embedding_only_encoding(self):
        type = "other"  # Replace with the appropriate type other than "prefix"
        length = 100
        projection = False
        embed_dim = 128
        proj_dim = 64
        layers = 3
        vocab_size = 1000

        encoder = PromptEncoder(type, length, projection, embed_dim, proj_dim, layers, vocab_size)

        prefix_input = torch.tensor([6, 7, 8, 9, 10])  # Replace with your prefix tensor
        encoded_output = encoder(prefix_input)

        expected_shape = (prefix_input.size(0), layers * 2 * embed_dim)
        self.assertEqual(encoded_output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
