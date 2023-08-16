class PromptEncoder(torch.nn.Module):
    r"""
    Prompt encoder to generate prompts, including prompt, prefix, instance and instruction
    """

    def __init__(
            self,
            type,
            length,
            projection,
            embed_dim,
            proj_dim,
            layers,
            vocab_size):
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
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values