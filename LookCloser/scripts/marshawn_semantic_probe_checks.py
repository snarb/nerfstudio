"""Diagnostic-only comparisons for native semantic technical probes."""
def gradient_max_difference(left, right):
    if left is None or right is None:
        raise ValueError('Expected two gradient tensors, not missing gradients')
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('Gradient shapes or dtypes differ')
    return float((left-right).abs().max()) if left.numel() else 0.
