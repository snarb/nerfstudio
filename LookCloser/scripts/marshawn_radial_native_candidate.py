"""Cold native radiometry diagnostic; no field architecture or frequency changes."""


def __getattr__(name):
    if name != 'NativeRadialColdPipeline':
        raise AttributeError(name)
    from marshawn_packed_distortion_candidate import NativePackedDistortionPipeline
    from marshawn_mvs_distribution_candidate import MVSDistributionDepth
    from marshawn_true_los import TrueTauFreeSpaceObjective

    class NativeRadialColdPipeline(NativePackedDistortionPipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if any(hasattr(self.config, key) for key in
                   ['floor_band_prior_manifest', 'floor_band_prior_weight']):
                raise ValueError('No assumed finite floor-plane prior in this diagnostic')
            if not bool((self._final_free.starts >= .052 - 1e-8).all()):
                raise ValueError('LOS artifact must start at max(original near, .052)')
            object.__setattr__(self, '_final_free',
                               TrueTauFreeSpaceObjective.from_existing(self._final_free))
            objective = MVSDistributionDepth(self.config.MVS_depth_manifest,
                self.config.MVS_depth_manifest_sha256, self,
                weight=self.config.MVS_depth_weight)
            object.__setattr__(self, '_MVS_depth', objective)

        def get_train_loss_dict(self, step):
            outputs, losses, metrics = super().get_train_loss_dict(step)
            # The main capture has exited before the separate 128 MVS rays.
            value, diagnostic = self._MVS_depth(self.model, int(step))
            losses['MVS_distribution_depth'] = value
            metrics.update(diagnostic)
            return outputs, losses, metrics

    NativeRadialColdPipeline.__name__ = name
    NativeRadialColdPipeline.__qualname__ = name
    NativeRadialColdPipeline.__module__ = __name__
    globals()[name] = NativeRadialColdPipeline
    return NativeRadialColdPipeline
