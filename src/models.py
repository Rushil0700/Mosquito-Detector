"""Models for 64 x 63 log-mel windows (target device: Android phone, TensorFlow Lite).

Backbone (config `model.arch`):
  * cnn (default): small plain CNN, `model.channels` conv blocks (M3: 79k parameters, m4_final: 316k).
  * resnet: a Keras ResNet (`model.resnet`, e.g. ResNet50V2), optionally ImageNet-pretrained, behind adapter layers that
    turn the 64 x 63 x 1 window into the image the ResNet expects (see build_cnn).

Preprocessing inside the model (so the exported model takes raw log-mel dB exactly as computed by src/features.py):
  * optional spectral whitening: subtract the window's broad spectral envelope (time-mean of each band, smoothed across
    `whiten_bands` mel bands). Removes fixed microphone / room colouring and overall level, keeps narrow harmonic peaks.
  * per-band normalisation with fixed mean / std (from real training windows, after whitening if enabled).
"""
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

SPECIES = ["ae_aegypti", "ae_albopictus", "an_stephensi", "cx_quinquefasciatus", "other_mosquito"]
DEFAULT_SPECIES = list(SPECIES)


def set_species(classes=None):
    """Choose the species classes of the species head (config `species_classes`). Mutates SPECIES in place, so every
    module that imported it sees the same list. Mosquito windows of any other species keep their presence label but get
    no species target (masked), i.e. they still train the detector."""
    SPECIES[:] = list(classes or DEFAULT_SPECIES)
    return SPECIES


@tf.keras.utils.register_keras_serializable(package="mosquito")
class SpectralWhitening(layers.Layer):
    """x[f, t] - smooth_f(mean_t x[f, t]); average pooling excludes padding at the band edges. TFLite-friendly ops."""

    def __init__(self, bands=15, **kw):
        super().__init__(**kw)
        self.bands = bands
        self.pool = layers.AveragePooling1D(pool_size=bands, strides=1, padding="same")

    def call(self, x):  # x: [batch, mels, frames]
        env = self.pool(tf.reduce_mean(x, axis=2, keepdims=True))  # [batch, mels, 1]
        return x - env

    def get_config(self):
        return {**super().get_config(), "bands": self.bands}


def norm_stats(cfg, norm_path, sample=None):
    """Per-band mean / std: from norm.json (raw log-mel) or, with whitening, computed on a sample of training windows."""
    wb = cfg["model"].get("whiten_bands")
    if not wb:
        n = json.load(open(norm_path))
        return np.array(n["mean"], np.float32), np.array(n["std"], np.float32)
    assert sample is not None, "whitening needs a sample of training windows to compute normalisation statistics"
    w = SpectralWhitening(wb)(tf.constant(sample, tf.float32)).numpy()
    return w.mean(axis=(0, 2)).astype(np.float32), (w.std(axis=(0, 2)) + 1e-3).astype(np.float32)


def resnet_backbone(mcfg, x):
    """Adapter layers + Keras ResNet. x: band-normalised window [batch, 64 mels, 63 frames, 1].

    * to_resnet_size: bilinear resize to `input_size` (default 128 x 128). A ResNet's stem (stride-2 conv + max-pool)
      divides the input by 4 at once and the whole network by 32; on the raw 64 x 63 window that would merge 4 mel bands
      (~60-250 Hz) before the first residual block and end on a 2 x 2 map. Upsampling keeps the harmonic peaks apart
      for longer (no new information, more compute).
    * to_rgb: learnable 1 x 1 conv, 1 -> 3 channels (initialised to copy the window into all three), because the
      ImageNet weights expect a 3-channel image.
    * the ResNet without its classifier (include_top=False); its 2-D feature map goes to the shared pooling + heads."""
    h, w = mcfg.get("input_size", [128, 128])
    x = layers.Resizing(h, w, interpolation="bilinear", name="to_resnet_size")(x)
    x = layers.Conv2D(3, 1, kernel_initializer="ones", name="to_rgb")(x)
    weights = mcfg.get("pretrained") or None  # "imagenet" or None (random init)
    net = getattr(tf.keras.applications, mcfg.get("resnet", "ResNet50V2"))
    return net(include_top=False, weights=weights, input_shape=(h, w, 3), name="resnet")(x)


def build_cnn(cfg, norm_path, input_shape=(64, 63), tasks=("presence",), norm_sample=None):
    mean, std = norm_stats(cfg, norm_path, norm_sample)
    inp = layers.Input(shape=input_shape, name="logmel_db")
    x = inp
    if cfg["model"].get("whiten_bands"):
        x = SpectralWhitening(cfg["model"]["whiten_bands"], name="whiten")(x)
    x = layers.Reshape((*input_shape, 1))(x)
    x = layers.Rescaling(1.0 / std[:, None, None], offset=-mean[:, None, None] / std[:, None, None], name="band_norm")(x)
    if cfg["model"].get("arch", "cnn") == "resnet":
        x = resnet_backbone(cfg["model"], x)
    else:
        chans = cfg["model"]["channels"]
        for i, c in enumerate(chans):
            x = layers.Conv2D(c, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
            x = layers.BatchNormalization(name=f"bn{i}")(x)
            x = layers.ReLU(name=f"relu{i}")(x)
            if i < len(chans) - 1:
                x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(cfg["model"]["dropout"], name="dropout")(x)
    outs = {}
    if "presence" in tasks:
        outs["presence"] = layers.Dense(1, activation="sigmoid", name="presence")(x)
    if "species" in tasks:
        outs["species"] = layers.Dense(len(SPECIES), activation="softmax", name="species")(x)
    if "sex" in tasks:  # hook: trained only where sex labels exist (masked loss)
        outs["sex"] = layers.Dense(1, activation="sigmoid", name="sex")(x)
    return tf.keras.Model(inp, outs, name=cfg.get("name", "mosquito_cnn"))
