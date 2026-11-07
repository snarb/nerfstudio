"""Temporary selector composition for a read-only field counterfactual.

The original coarse field selector remains in force. The point lookup is an
additional admission condition, never a replacement for native traversal.
"""
import types


def compose_normalization(field, lookup):
    original = field._normalize_positions

    def normalized(self, positions):
        flat, selector, shape, world = original(positions)
        admitted = lookup(positions)
        if admitted.shape != selector.shape or admitted.dtype != selector.dtype:
            raise ValueError('Point admission must match native Boolean selector')
        return flat, selector & admitted, shape, world

    field._normalize_positions = types.MethodType(normalized, field)
    return original


def restore_normalization(field, original):
    field._normalize_positions = original
