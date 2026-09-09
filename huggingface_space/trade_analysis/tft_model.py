# tft_model.py
from pathlib import Path, PurePath
import os 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
import joblib 



# Where the checkpoints live. Three things were wrong here and each hid the next:
#
#   * MODEL_DIR was assigned TWICE -- the os.getenv() form was immediately overwritten by
#     a hardcoded path, so TFT_MODEL_DIR silently did nothing.
#   * both paths were university HPC mounts (/scratch/..., /courses/...) that exist on
#     none of the machines this actually runs on, least of all a Docker container.
#   * resolve_tft_path was also DEFINED TWICE; the first was a stub with no return
#     statement, shadowed by the second, so the bug was invisible.
#
# None of it ever surfaced because the TFT was gated on `len(daily_df) >= 96` and the old
# data layer returned a single row, so this code had never once executed in production.
#
# Now: package-relative by default (works in the container, the repo, and a clone),
# TFT_MODEL_DIR honoured for real, HPC mounts kept last as genuine fallbacks.
_PKG_MODELS = Path(__file__).resolve().parent.parent / "trained_models"
_FALLBACK_DIRS = [
    Path("/scratch/kharche.c/kharche.c/trained_models"),
    Path("/courses/CS6120.202550/students/kharche.c/trade_analysis/trained_models"),
]


def _model_dirs() -> list[Path]:
    dirs = []
    env = os.getenv("TFT_MODEL_DIR")
    if env:
        dirs.append(Path(env))
    dirs.append(_PKG_MODELS)
    dirs.extend(_FALLBACK_DIRS)
    return dirs


MODEL_DIR = _PKG_MODELS
DEFAULT_MODEL = MODEL_DIR / "tft_model.pth"


def resolve_tft_path(symbol: str | None = None, explicit: str | None = None) -> Path:
    """First existing checkpoint for `symbol`, searched across every candidate directory.

    `_validated` is tried FIRST and was previously not tried at all -- most of the
    checkpoints on disk are named tft_<SYM>_validated.pth, so the old candidate list
    could not have found them even with a correct directory.
    """
    if explicit:
        return Path(explicit)
    tried = []
    for d in _model_dirs():
        names = []
        if symbol:
            s = symbol.upper()
            names += [f"tft_{s}_validated.pth", f"tft_{s}_e200_.pth", f"tft_{s}.pth"]
        names += ["tft_model.pth"]
        for n in names:
            p = d / n
            tried.append(str(p))
            if p.exists():
                return p
    raise FileNotFoundError("No TFT weights found. Tried: " + ", ".join(tried))
    
    

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for financial gap prediction.
    Shapes are enforced.
    """
    def __init__(
        self,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        num_heads: int = 8,
        output_size: int = 1,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        dropout: float = 0.1,
        context_length: int = 96,
        prediction_length: int = 1,
        num_temporal_features: int = 10,
        num_static_features: int = 5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_heads = num_heads
        self.output_size = output_size
        self.quantiles = quantiles
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_temporal_features = num_temporal_features
        self.num_static_features = num_static_features

        self.static_embedding = nn.Linear(num_static_features, hidden_size)
        self.temporal_embedding = nn.Linear(num_temporal_features, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.gate_lstm = nn.Linear(hidden_size, hidden_size)
        self.gate_attention = nn.Linear(hidden_size, hidden_size)

        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size)
            ) for _ in quantiles
        ])

        self.gap_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # classes UP DOWN FLAT
        )

        self.dropout = nn.Dropout(dropout)
        
        

    def forward(self, temporal_data: torch.Tensor, static_data: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        temporal_data = temporal_data.float()
        assert temporal_data.dim() == 3, f"Expected 3D temporal tensor, got {temporal_data.shape}"

        batch_size, seq_len, num_features = temporal_data.shape

        if num_features != self.num_temporal_features:
            if num_features < self.num_temporal_features:
                padding = torch.zeros(
                    batch_size, seq_len, self.num_temporal_features - num_features,
                    device=temporal_data.device, dtype=temporal_data.dtype
                )
                temporal_data = torch.cat([temporal_data, padding], dim=-1)
            else:
                temporal_data = temporal_data[:, :, :self.num_temporal_features]

        temporal_embedded = self.temporal_embedding(temporal_data)

        if static_data is not None:
            static_data = static_data.float()
            if static_data.shape[-1] != self.num_static_features:
                if static_data.shape[-1] < self.num_static_features:
                    padding = torch.zeros(
                        batch_size, self.num_static_features - static_data.shape[-1],
                        device=static_data.device, dtype=static_data.dtype
                    )
                    static_data = torch.cat([static_data, padding], dim=-1)
                else:
                    static_data = static_data[:, :self.num_static_features]

            static_embedded = self.static_embedding(static_data)
            static_expanded = static_embedded.unsqueeze(1).expand(-1, seq_len, -1)
            temporal_embedded = temporal_embedded + static_expanded

        lstm_out, _ = self.lstm(temporal_embedded)

        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        gate_lstm = torch.sigmoid(self.gate_lstm(lstm_out))
        gate_attn = torch.sigmoid(self.gate_attention(attn_out))

        combined = gate_lstm * lstm_out + gate_attn * attn_out
        combined = self.dropout(combined)

        final_hidden = combined[:, -1, :]

        quantile_outputs = {}
        for i, q in enumerate(self.quantiles):
            quantile_outputs[f'quantile_{q}'] = self.quantile_heads[i](final_hidden)

        gap_logits = self.gap_classifier(final_hidden)
        gap_probs = torch.softmax(gap_logits, dim=-1)

        return {
            'quantile_predictions': quantile_outputs,
            'gap_classification': gap_probs,
            'attention_weights': attn_weights,
            'hidden_state': final_hidden
        }

class GapPredictionTFT:
    """
    Wrapper for data prep, train, save, load, predict.
    """
    def __init__(self, context_length: int = 96, prediction_length: int = 1):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Feature names aligned with your DataFrame
        self.temporal_features = [
            'close',
            'volume',
            'RSI_14',
            'MACDh_12_26_9',
            'ADX_9',
            'ATR_14',
            'EMA_9',
            'returns',
            'volatility',
            'high_low_ratio'
        ]
        self.static_features = [
            'market_cap_category',
            'beta_estimate',
            'sector_code',
            'vix_regime',
            'liquidity_score'
        ]

        self.model = TemporalFusionTransformer(
            hidden_size=128,
            lstm_layers=2,
            num_heads=8,
            output_size=prediction_length,
            quantiles=[0.1, 0.5, 0.9],
            context_length=context_length,
            prediction_length=prediction_length,
            num_temporal_features=len(self.temporal_features),
            num_static_features=len(self.static_features)
        ).to(self.device)

        self.is_trained = False
        self.scaler_temporal = None
        self.scaler_static = None
    
    def load_pretrained(self, symbol: str | None = None, path: str | None = None, map_location=None):
        """
        Loads weights from .pth and scalers from .joblib.
        This is the corrected, safe version.
        """
        p_weights = resolve_tft_path(symbol, path)
        p_scalers = p_weights.with_suffix('.joblib')
        ml = map_location or self.device

        # This will now work because the .pth file ONLY contains tensors.
        state_dict = torch.load(p_weights, map_location=ml, weights_only=True)
        self.model.load_state_dict(state_dict)
        print(f"🔎 Loaded TFT weights from: {p_weights}")

        # Load scalers from the separate .joblib file
        if p_scalers.exists():
            metadata = joblib.load(p_scalers)
            self.scaler_temporal = metadata.get('scaler_temporal')
            self.scaler_static = metadata.get('scaler_static')
            print(f"🔎 Loaded scalers from: {p_scalers}")
        else:
            print(f"⚠️ Scaler file not found at {p_scalers}. Scalers not loaded.")

        self.is_trained = True
        return self
        
    
    def _col(self, df: pd.DataFrame, name: str) -> Optional[pd.Series]:
        # Prefer lower then upper
        if name in df.columns:
            return df[name]
        cap = name[0].upper() + name[1:]
        if cap in df.columns:
            return df[cap]
        return None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(df) < self.context_length + 10:
            dummy_temporal = torch.zeros((1, self.context_length, len(self.temporal_features)))
            dummy_static = torch.zeros((1, len(self.static_features)))
            dummy_targets = torch.zeros((1, 1))
            return dummy_temporal, dummy_static, dummy_targets

        close_col = self._col(df, 'close')
        high_col = self._col(df, 'high')
        low_col = self._col(df, 'low')
        volume_col = self._col(df, 'volume')

        temporal_data = []
        for col in self.temporal_features:
            if col == 'returns':
                if close_col is not None:
                    series = close_col.pct_change().fillna(0.0)
                else:
                    series = pd.Series(0.0, index=df.index)
            elif col == 'volatility':
                if close_col is not None:
                    series = close_col.rolling(20).std().fillna(close_col.std())
                else:
                    series = pd.Series(0.02, index=df.index)
            elif col == 'high_low_ratio':
                if high_col is not None and low_col is not None and close_col is not None:
                    denom = close_col.replace(0, np.nan)
                    series = ((high_col - low_col) / denom).fillna(0.0)
                else:
                    series = pd.Series(0.02, index=df.index)
            elif col == 'close':
                if close_col is not None:
                    fillv = close_col.mean() if not close_col.isna().all() else 0.0
                    series = close_col.fillna(fillv)
                else:
                    series = pd.Series(100.0, index=df.index)
            elif col == 'volume':
                if volume_col is not None:
                    fillv = volume_col.median() if not volume_col.isna().all() else 0.0
                    series = volume_col.fillna(fillv)
                else:
                    series = pd.Series(0.0, index=df.index)
            else:
                if col in df.columns:
                    base = df[col]
                else:
                    base = self._col(df, col)  # try capitalized
                if base is not None:
                    fillv = base.mean() if not base.isna().all() else 0.0
                    series = base.fillna(fillv)
                else:
                    series = pd.Series(0.0, index=df.index)

            temporal_data.append(series.values)

        temporal_array = np.column_stack(temporal_data)

        static_data = []
        for col in self.static_features:
            if col == 'market_cap_category':
                static_data.append(2.0)
            elif col == 'beta_estimate':
                static_data.append(1.0)
            elif col == 'sector_code':
                static_data.append(1.0)
            elif col == 'vix_regime':
                static_data.append(2.0)
            elif col == 'liquidity_score':
                static_data.append(50.0)
            else:
                static_data.append(0.0)
        static_array = np.array(static_data)

        from sklearn.preprocessing import StandardScaler

        if self.scaler_temporal is None:
            self.scaler_temporal = StandardScaler()
            temporal_normalized = self.scaler_temporal.fit_transform(temporal_array)
        else:
            temporal_normalized = self.scaler_temporal.transform(temporal_array)

        if self.scaler_static is None:
            self.scaler_static = StandardScaler()
            static_normalized = self.scaler_static.fit_transform(static_array.reshape(1, -1)).flatten()
        else:
            static_normalized = self.scaler_static.transform(static_array.reshape(1, -1)).flatten()

        sequences = []
        static_sequences = []
        targets = []

        for i in range(len(temporal_normalized) - self.context_length - self.prediction_length + 1):
            seq = temporal_normalized[i:i + self.context_length]
            sequences.append(seq)
            static_sequences.append(static_normalized)

            if close_col is not None:
                future_price = close_col.iloc[i + self.context_length + self.prediction_length - 1]
                current_price = close_col.iloc[i + self.context_length - 1]
            else:
                future_price = 100.0
                current_price = 100.0
            gap_magnitude = (future_price - current_price) / max(current_price, 1e-9)
            targets.append(gap_magnitude)

        if not sequences:
            dummy_temporal = torch.zeros((1, self.context_length, len(self.temporal_features)))
            dummy_static = torch.zeros((1, len(self.static_features)))
            dummy_targets = torch.zeros((1, 1))
            return dummy_temporal, dummy_static, dummy_targets

        temporal_tensor = torch.FloatTensor(sequences)
        static_tensor = torch.FloatTensor(static_sequences)
        target_tensor = torch.FloatTensor(targets).unsqueeze(-1)

        print(f"FIXED: Tensor shapes - Temporal: {temporal_tensor.shape}, Static: {static_tensor.shape}, Targets: {target_tensor.shape}")
        return temporal_tensor, static_tensor, target_tensor

    def train(self, df: pd.DataFrame, epochs: int = 50, learning_rate: float = 1e-3):
        # Split data: 80% train, 20% validation
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx]
        val_df = df[split_idx:]
        
        train_temporal, train_static, train_targets = self.prepare_data(train_df)
        val_temporal, val_static, val_targets = self.prepare_data(val_df)
        
        # Move to device
        train_temporal, train_static, train_targets = train_temporal.to(self.device), train_static.to(self.device), train_targets.to(self.device)
        val_temporal, val_static, val_targets = val_temporal.to(self.device), val_static.to(self.device), val_targets.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(train_temporal, train_static)
            train_loss = nn.MSELoss()(outputs['quantile_predictions']['quantile_0.5'], train_targets)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(val_temporal, val_static)
                val_loss = nn.MSELoss()(val_outputs['quantile_predictions']['quantile_0.5'], val_targets)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model_temp.pth')
            else:
                patience_counter += 1
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model_temp.pth'))
        os.remove('best_model_temp.pth')
    
                
            
            
    def save_model(self, save_path: str):
        """
        Saves model weights to .pth and scalers to a separate .joblib file.
        This is the corrected version.
        """
        base_path = Path(save_path)
        model_weights_path = base_path.with_suffix('.pth')
        scalers_path = base_path.with_suffix('.joblib')

        os.makedirs(base_path.parent, exist_ok=True)
        
        # 1. Save only the model's state_dict (tensors) to the .pth file
        torch.save(self.model.state_dict(), model_weights_path)
        print(f"Saved model weights to {model_weights_path}")

        # 2. Save the scalers and other metadata to the .joblib file
        metadata = {
            'scaler_temporal': self.scaler_temporal,
            'scaler_static': self.scaler_static,
            # You can add other non-tensor metadata here if needed
        }
        joblib.dump(metadata, scalers_path)
        print(f"Saved scalers to {scalers_path}")

    @classmethod
    def load_model(cls, load_path: str, device: Optional[torch.device] = None) -> "GapPredictionTFT":
        """
        Restore model, scalers, and meta from disk.
        """
        dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(load_path, map_location=dev, weights_only=False)

        ctx = ckpt.get('context_length', 96)
        pred_len = ckpt.get('prediction_length', 1)

        obj = cls(context_length=ctx, prediction_length=pred_len)
        if 'model_config' in ckpt:
            cfg = ckpt['model_config']
            # rebuild to exact sizes if needed
            obj.model = TemporalFusionTransformer(
                hidden_size=cfg['hidden_size'],
                lstm_layers=cfg['lstm_layers'],
                num_heads=cfg['num_heads'],
                output_size=cfg['output_size'],
                quantiles=cfg['quantiles'],
                context_length=ctx,
                prediction_length=pred_len,
                num_temporal_features=cfg['num_temporal_features'],
                num_static_features=cfg['num_static_features']
            ).to(dev)

        obj.temporal_features = ckpt.get('temporal_features', obj.temporal_features)
        obj.static_features = ckpt.get('static_features', obj.static_features)

        obj.model.load_state_dict(ckpt['model_state_dict'])
        obj.model.eval()
        obj.device = dev

        obj.scaler_temporal = ckpt.get('scaler_temporal', None)
        obj.scaler_static = ckpt.get('scaler_static', None)

        obj.is_trained = True
        print(f"Loaded from {load_path}")
        return obj

    def predict_gap_probability(self, df: pd.DataFrame) -> Dict:
        try:
            if not self.is_trained:
                print("Model not trained. Training now.")
                self.train(df, epochs=10)

            self.model.eval()
            temporal_data, static_data, _ = self.prepare_data(df)
            if len(temporal_data) == 0:
                return self._default_prediction()

            temporal_input = temporal_data[-1:].to(self.device)
            static_input = static_data[-1:].to(self.device)

            with torch.no_grad():
                outputs = self.model(temporal_input, static_input)

            quantile_preds = outputs['quantile_predictions']
            gap_class_probs = outputs['gap_classification'].cpu().numpy()[0]

            lower = float(quantile_preds['quantile_0.1'].cpu().numpy()[0, 0])
            median = float(quantile_preds['quantile_0.5'].cpu().numpy()[0, 0])
            upper = float(quantile_preds['quantile_0.9'].cpu().numpy()[0, 0])

            up_prob, down_prob, flat_prob = gap_class_probs.tolist()
            direction_probs = {"UP": up_prob, "DOWN": down_prob, "FLAT": flat_prob}
            expected_direction = max(direction_probs, key=direction_probs.get)

            gap_probability = (up_prob + down_prob) * 100.0

            uncertainty = upper - lower
            if uncertainty < 0.01 and max(gap_class_probs) > 0.7:
                confidence = "HIGH"
            elif uncertainty < 0.02 and max(gap_class_probs) > 0.5:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            return {
                "gap_probability": round(gap_probability, 1),
                "expected_direction": expected_direction,
                "confidence_intervals": {
                    "lower": round(lower, 4),
                    "median": round(median, 4),
                    "upper": round(upper, 4)
                },
                "confidence": confidence,
                "direction_probabilities": {
                    "UP": round(up_prob, 3),
                    "DOWN": round(down_prob, 3),
                    "FLAT": round(flat_prob, 3)
                }
            }
        except Exception as e:
            print(f"TFT prediction failed: {e}")
            return self._default_prediction()

    def _default_prediction(self) -> Dict:
        return {
            "gap_probability": 50.0,
            "expected_direction": "FLAT",
            "confidence_intervals": {"lower": -0.01, "median": 0.0, "upper": 0.01},
            "confidence": "LOW",
            "direction_probabilities": {"UP": 0.33, "DOWN": 0.33, "FLAT": 0.34}
        }
