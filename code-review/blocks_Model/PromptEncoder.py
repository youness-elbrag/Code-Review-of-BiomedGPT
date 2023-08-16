import torch

class PromptEncoder(torch.nn.Module):
    """
    Prompt encoder to generate prompts, including prompt, prefix, instance, and instruction.

    This module provides functionality for encoding prompts based on different types:
    - For type "prefix", it generates prefix encodings using embeddings and projections.
    - For other types, it generates encodings using embeddings only.

    Args:
        type (str): Type of prompt encoding ("prefix" or other types).
        length (int): Length of the prompt vocabulary.
        projection (bool): Whether to use projection for prefix encoding.
        embed_dim (int): Dimension of the embedding space.
        proj_dim (int): Dimension of the projected space (if using projection).
        layers (int): Number of layers for projection.
        vocab_size (int): Vocabulary size for embedding.

    Attributes:
        prefix_projection (bool): Whether prefix projection is enabled.
        embedding (torch.nn.Embedding): Embedding layer for prompt tokens.
        trans (torch.nn.Sequential): Sequential module for prefix projection.

    """
    def __init__(self, type, length, projection, embed_dim, proj_dim, layers, vocab_size):
        super().__init__()
        self.prefix_projection = projection

        if type == "prefix":
            layers = layers
            prompt_vocab_size = length

        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(prompt_vocab_size, embed_dim)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, proj_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_dim, layers * 2 * embed_dim)
            )
        else:
            if type == "prefix":
                self.embedding = torch.nn.Embedding(
                    prompt_vocab_size, layers * 2 * embed_dim)

    def forward(self, prefix: torch.Tensor):
        """
        Forward pass of the prompt encoder.

        Args:
            prefix (torch.Tensor): Input tensor representing prompt tokens.

        Returns:
            torch.Tensor: Encoded prompt tensor.

        """
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
