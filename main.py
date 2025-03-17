from glob import glob
import os

import yaml

from utils import audio_stream_from_wav
from vad_models.vad_factory import load_vad


def load_config(config_path="configs/vad_config.yaml"):
    """Loads the VAD configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    input_audio_dir = config["input_audio_dir"]
    lab_files_dir = config["lab_files_dir"]
    output_files_dir = config["output_files_dir"]

    vad_class = load_vad()
    print(f"Using {type(vad_class)}")

    # Get all FLAC files in the input_audio_dir
    audio_files = sorted(glob(os.path.join(input_audio_dir, "*.wav")))

    for audio_file in audio_files:
        filename = os.path.basename(audio_file).replace(".wav", "")
        lab_file = os.path.join(lab_files_dir, f"{filename}.lab")

        print(f"Processing {audio_file} with label {lab_file}")

        # Simulating Streaming
        for audio_chunk in audio_stream_from_wav(audio_file):
            vad_class.process_audio_chunk(audio_chunk)

        vad_class.log_vad(output_files_dir)


if __name__ == "__main__":
    main()
