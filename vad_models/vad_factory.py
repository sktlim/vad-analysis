import yaml

from vad_models.pyannoteVAD import PyannoteStreamingVAD
from vad_models.sileroVAD import SileroVAD
from vad_models.webrtc_vad import WebRTCVAD


def load_vad(output_dir, filename):
    """Loads the VAD model specified in the config file"""
    with open("configs/vad_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vad_type = config.get("vad_model", "pyannote")  # Default to Pyannote
    chunk_size = config.get("chunk_size", 5)
    overlap = config.get("overlap", 1)
    sample_rate = config.get("sample_rate", 16000)

    if vad_type == "pyannote":
        return PyannoteStreamingVAD(
            chunk_size, overlap, sample_rate, output_dir, filename
        )
    elif vad_type == "silero":
        return SileroVAD(chunk_size, overlap, sample_rate)
    elif vad_type == "webrtc":
        return WebRTCVAD(chunk_size, overlap, sample_rate)
    else:
        raise ValueError(f"Unsupported VAD model: {vad_type}")
