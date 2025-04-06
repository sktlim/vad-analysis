import os
from glob import glob

import yaml
from memory_profiler import memory_usage
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

from utils import audio_stream_from_wav, setup_logger
from vad_models.vad_factory import load_vad
from tqdm import tqdm


def load_config(config_path="configs/vad_config.yaml"):
    """Loads the VAD configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)


def get_gpu_memory():
    """Get current VRAM usage in MB."""
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024**2)  # Convert bytes to MB


def profile_vad_processing(vad_class, audio_chunk, chunk_idx, output_dir, filename):
    """Profiles both RAM and VRAM usage for processing a single audio chunk."""

    memory_logger = setup_logger(output_dir, filename, "memory")

    before_vram = get_gpu_memory()

    def process():
        vad_class.process_audio_chunk(audio_chunk)

    mem_usage = memory_usage(process, interval=0.1, timeout=10)

    after_vram = get_gpu_memory()

    log_entry = (
        f"Chunk {chunk_idx}: Max RAM usage = {max(mem_usage):.2f} MB, "
        f"VRAM used = {after_vram - before_vram:.2f} MB"
    )

    memory_logger.info(log_entry)


def main():
    config = load_config()
    input_audio_dir = config["input_audio_dir"]
    lab_files_dir = config["lab_files_dir"]
    output_files_dir = config["output_files_dir"]

    audio_files = sorted(glob(os.path.join(input_audio_dir, "*.flac")))

    for audio_file in tqdm(audio_files):
        filename = os.path.basename(audio_file).replace(".flac", "")
        lab_file = os.path.join(lab_files_dir, f"{filename}.lab")
        vad_class = load_vad(output_files_dir, filename)
        vad_class.reset()

        os.makedirs(os.path.join(output_files_dir, filename), exist_ok=True)

        vad_class.output_dir = output_files_dir
        vad_class.filename = filename

        # Simulating Streaming
        for chunk_idx, audio_chunk in enumerate(
            audio_stream_from_wav(audio_file), start=1
        ):
            vad_class.process_audio_chunk(audio_chunk)
            # profile_vad_processing(
            #     vad_class, audio_chunk, chunk_idx, output_files_dir, filename
            # )

        vad_class.label_file_path = lab_file
        vad_class.log_vad()


if __name__ == "__main__":
    main()
