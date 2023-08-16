class FrozenBatchNorm2d(nn.Module):
    """
    Batch normalization layer with fixed batch statistics and affine parameters.

    This module implements batch normalization where the batch statistics and the affine
    parameters are fixed. The class registers non-trainable buffers for "weight", "bias",
    "running_mean", and "running_var", initialized to perform an identity transformation.

    The forward pass is implemented using `F.batch_norm(..., training=False)`.

    Args:
        num_features (int): Number of input features.
        eps (float, optional): A small value added to the denominator for numerical stability.

    Attributes:
        num_features (int): Number of input features.
        eps (float): Small value added to the denominator.
        weight (torch.Tensor): Non-trainable buffer for affine weight.
        bias (torch.Tensor): Non-trainable buffer for affine bias.
        running_mean (torch.Tensor): Non-trainable buffer for running mean.
        running_var (torch.Tensor): Non-trainable buffer for running variance.

    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        """
        Forward pass through the FrozenBatchNorm2d layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Load model state from a dictionary.

        This method handles loading of state dictionary and adjusts the state keys
        based on version information.

        Args:
            state_dict (dict): Model state dictionary.
            prefix (str): Prefix to apply to state keys.
            local_metadata (dict): Local metadata.
            strict (bool): Whether to strictly enforce that the keys match.
            missing_keys (list): List of missing keys.
            unexpected_keys (list): List of unexpected keys.
            error_msgs (list): List of error messages.

        """
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        """
        Return a string representation of the FrozenBatchNorm2d module.

        Returns:
            str: String representation.

        """
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)