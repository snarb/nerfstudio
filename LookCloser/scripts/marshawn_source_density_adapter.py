"""Install only source-coordinate density admission; native traversal stays separate."""
import types
from marshawn_source_point_admission import SourcePointAdmission


def install_source_density_admission(field, source_mask, source_bounds, epsilon=2e-7):
    if getattr(field, '_semantic_admission_installed', False):
        raise ValueError('Density gate already installed')
    lookup = SourcePointAdmission(source_mask, source_bounds, epsilon=epsilon)
    original = field._normalize_positions

    def normalized(self, positions):
        flat, selector, shape, world = original(positions)
        return flat, selector & lookup(positions), shape, world

    field._semantic_original_normalize = original
    field._normalize_positions = types.MethodType(normalized, field)
    field._semantic_admission_installed = True
    return lookup
