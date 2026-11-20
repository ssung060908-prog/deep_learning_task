"""Standalone inference for the frozen Task 1 v2 model; no training data needed."""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

TARGET = "is_at_risk"

ID = "sample_id"

CODES = (-9, -8, -7)

RATIO_FEATURES = (
    ("number_of_satisfactory_trades", "nr_total_trades"),
    ("nr_trades_insolvent_for_over_60_days", "nr_total_trades"),
    ("nr_trades_insolvent_for_over_90_days", "nr_total_trades"),
    ("nr_trades_initiated_in_last_year", "nr_total_trades"),
    ("nr_revolving_trades_with_balance", "nr_total_trades"),
    ("nr_installment_trades_with_balance", "nr_total_trades"),
    ("nr_banks_with_high_ratio", "nr_total_trades"),
    ("average_duration_of_resolution", "months_since_first_trade"),
    ("months_since_last_trade", "months_since_first_trade"),
)

def numeric_frame(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    return frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

class Preprocessor:
    """Fit only on each training partition; serialize pure data in the checkpoint."""
    def __init__(self, columns, scaling="standard", interactions=False, domain_features=False):
        self.columns = list(columns)
        self.scaling = scaling
        self.interactions = interactions
        self.domain_features = domain_features

    def clean(self, frame):
        raw = numeric_frame(frame, self.columns)
        flags = np.concatenate([(raw == c).astype(float) for c in CODES] +
                               [(~np.isfinite(raw)).astype(float)], axis=1)
        missing = np.isin(raw, CODES) | ~np.isfinite(raw)
        values = np.where(missing, np.nan, raw)
        if self.domain_features:
            extra = []
            for numerator, denominator in RATIO_FEATURES:
                if numerator in self.columns and denominator in self.columns:
                    a = values[:, self.columns.index(numerator)]
                    b = values[:, self.columns.index(denominator)]
                    ratio = np.full(len(raw), np.nan)
                    np.divide(a, b, out=ratio, where=np.isfinite(a) & np.isfinite(b) & (b > 0))
                    extra.append(ratio)
            if extra:
                derived = np.column_stack(extra)
                derived_missing = ~np.isfinite(derived)
                flags = np.concatenate([flags, derived_missing.astype(float)], axis=1)
                missing = np.concatenate([missing, derived_missing], axis=1)
                values = np.concatenate([values, derived], axis=1)
        return values, flags, missing

    def fit(self, frame):
        values, _, _ = self.clean(frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nanmedian(values, axis=0)
        self.median = np.nan_to_num(self.median, nan=0.0)
        values = np.where(np.isnan(values), self.median, values)
        if self.scaling == "quantile":
            transformer = QuantileTransformer(n_quantiles=min(256, len(values)),
                                              output_distribution="uniform", random_state=0)
            transformer.fit(values)
            self.quantiles = transformer.quantiles_
            self.references = transformer.references_
        else:
            transformer = StandardScaler() if self.scaling == "standard" else RobustScaler()
            transformer.fit(values)
            self.center = transformer.mean_ if self.scaling == "standard" else transformer.center_
            self.scale = transformer.scale_
        return self

    def transform(self, frame):
        values, flags, missing = self.clean(frame)
        values = np.where(np.isnan(values), self.median, values)
        if self.scaling == "quantile":
            # Same symmetric interpolation as sklearn, including repeated quantiles.
            result = np.empty_like(values)
            for j in range(values.shape[1]):
                q, v = self.quantiles[:, j], values[:, j]
                result[:, j] = 0.5 * (np.interp(v, q, self.references) -
                                     np.interp(-v, -q[::-1], -self.references[::-1]))
                result[v == q[-1], j] = 1
                result[v == q[0], j] = 0
            values = (result - 0.5) * np.sqrt(12.0)
        else:
            values = (values - self.center) / self.scale
        values = np.clip(values, -8, 8)
        parts = [values, flags, missing.mean(axis=1, keepdims=True)]
        if self.interactions:
            a, b = np.triu_indices(values.shape[1], k=0)
            parts.append(values[:, a] * values[:, b])
        output = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("Non-finite preprocessed values")
        return output

    def state(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in vars(self).items()}

    @classmethod
    def restore(cls, state):
        obj = cls(state["columns"], state["scaling"], state["interactions"], state.get("domain_features", False))
        for k, v in state.items():
            setattr(obj, k, np.asarray(v) if k in
                    {"median", "quantiles", "references", "center", "scale"} else v)
        return obj

class DNN(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        layers = []
        width = input_dim
        for hidden in config["hidden"]:
            layers.append(nn.Linear(width, hidden))
            if config["norm"] == "layer":
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU() if config["activation"] == "relu" else nn.GELU())
            layers.append(nn.Dropout(config["dropout"]))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

@torch.inference_mode()
def probabilities(model, values, device, batch_size=2048):
    model.eval()
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    return np.concatenate([model(torch.as_tensor(values[i:i + batch_size], device=device))
                           .sigmoid().cpu().numpy() for i in range(0, len(values), batch_size)])

def predict_proba(frame, weights, device="cpu"):
    """TA interface: raw DataFrame -> P(is_at_risk=1), original row order."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    prep = Preprocessor.restore(checkpoint["preprocessor"])
    model = DNN(checkpoint["input_dim"], checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return probabilities(model, prep.transform(frame), device)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='CSV containing the 23 named credit features')
    parser.add_argument('--weights', default=str(Path(__file__).resolve().with_name('task1.pt')))
    parser.add_argument('--output', default='predictions.csv')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    frame = pd.read_csv(args.data)
    result = pd.DataFrame({ID: frame[ID] if ID in frame else np.arange(len(frame)),
                           'probability': predict_proba(frame, args.weights)})
    result.to_csv(args.output, index=False)
    print(f'Saved {len(result)} predictions to {args.output}')

if __name__ == '__main__':
    main()
