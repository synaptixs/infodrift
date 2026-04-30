"""Population Stability Index (PSI) monitor — Layer 1 evidence health."""

# TODO: implement PSIMonitor with .fit(train_df, features, entity_key)
# and .compute(prod_df, features, entity_key) -> list of PSIResult.
# See white paper §3.1: Laplace smoothing eps=1e-6, quantile binning, B=10-20.


class PSIMonitor:
    def __init__(self, n_bins: int = 10):
        raise NotImplementedError
