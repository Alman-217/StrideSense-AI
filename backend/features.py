
import numpy as np


SAMPLE_RATE = 64


def extract_features(signal, sample_rate=SAMPLE_RATE):
    """
    Extract features from one two-second acceleration window.

    Input:
        One-dimensional acceleration magnitude in mg.

    Output:
        Dictionary of statistical and frequency-domain features.
    """

    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1 or len(x) < 64:
        raise ValueError("Invalid or insufficient signal window.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains invalid values.")

    x = x - np.mean(x)

    spectrum = np.abs(np.fft.rfft(x)) ** 2
    frequencies = np.fft.rfftfreq(
        len(x),
        d=1.0 / sample_rate
    )

    def band_power(low, high):
        mask = (
            (frequencies >= low) &
            (frequencies < high)
        )
        return float(np.sum(spectrum[mask]))

    low_power = band_power(0.5, 3.0)
    mid_power = band_power(3.0, 8.0)
    high_power = band_power(8.0, 15.0)

    total_power = low_power + mid_power + high_power + 1e-9

    probabilities = spectrum / (np.sum(spectrum) + 1e-9)
    probabilities = probabilities[probabilities > 0]

    spectral_entropy = float(
        -np.sum(probabilities * np.log2(probabilities))
    )

    features = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
        "range": float(np.ptp(x)),
        "mean_absolute_change": float(
            np.mean(np.abs(np.diff(x)))
        ),
        "percentile_10": float(np.percentile(x, 10)),
        "percentile_90": float(np.percentile(x, 90)),
        "low_frequency_ratio": low_power / total_power,
        "mid_frequency_ratio": mid_power / total_power,
        "high_frequency_ratio": high_power / total_power,
        "mid_to_low_ratio": mid_power / (low_power + 1e-9),
        "spectral_entropy": spectral_entropy
    }

    return features