import logging
import os
import time
from functools import wraps

import numpy as np
import soundfile as sf


def audio_stream_from_wav(file_path, chunk_size=5.0, overlap=1.0, sample_rate=16000):
    """
    Reads a WAV file in chunks of 'chunk_size' seconds, with 'overlap' seconds overlap.
    This version ensures proper alignment of chunks like audio_stream_from_wav().

    :param file_path: Path to the WAV file.
    :param chunk_size: Duration of each chunk (default: 5 seconds).
    :param overlap: Overlap between consecutive chunks (default: 1 second).
    :param sample_rate: Expected sample rate (default: 16000Hz).
    :yield: NumPy array (1, time) representing an audio chunk.
    """
    chunk_samples = int(chunk_size * sample_rate)  # Convert chunk size to samples
    step_size = int(
        (chunk_size - overlap) * sample_rate
    )  # Step size for moving to next chunk

    with sf.SoundFile(file_path, "r") as sf_file:
        if sf_file.samplerate != sample_rate:
            raise ValueError(
                f"Sample rate mismatch! Expected {sample_rate}, \
                    but got {sf_file.samplerate}"
            )

        # Start reading in correct step intervals
        position = 0  # Keeps track of the file position

        while position + chunk_samples <= len(sf_file):
            sf_file.seek(position)  # Move to the correct start position
            block = sf_file.read(chunk_samples, dtype="float32")

            # Ensure chunk is the correct size, pad if necessary
            if len(block) < chunk_samples:
                block = np.pad(block, (0, chunk_samples - len(block)), mode="constant")

            yield np.expand_dims(block, axis=0)  # Shape: (1, time)

            # Move position forward by step_size (which accounts for overlap)
            position += step_size

        # Handle the last chunk if any remaining samples exist
        if position < sf_file.frames:
            sf_file.seek(position)
            last_block = sf_file.read(chunk_samples, dtype="float32")

            # Pad the last chunk if it's smaller than expected
            if len(last_block) < chunk_samples:
                last_block = np.pad(
                    last_block, (0, chunk_samples - len(last_block)), mode="constant"
                )

            yield np.expand_dims(last_block, axis=0)  # (1, time)


def time_it(func):
    """
    Decorator to log per-chunk VAD processing time.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        vad_annotation = func(self, *args, **kwargs)  # Run VAD
        processing_time = time.time() - start_time

        # Extract detected speech timestamps
        speech_segments = [
            (seg.start, seg.end) for seg in vad_annotation.itersegments()
        ]
        total_speech_duration = sum(
            seg.end - seg.start for seg in vad_annotation.itersegments()
        )

        # Per-chunk log entry
        log_entry = (
            f"Chunk {self.current_offset:.2f}s | "
            f"Processed in {processing_time:.4f}s | "
            f"Detected Speech: {speech_segments if speech_segments else 'None'}"
        )

        timing_logger = setup_logger(self.output_dir, self.filename, "timing")
        timing_logger.info(log_entry)

        # Save log info
        self.chunk_logs.append(
            {
                "chunk_start_time": round(self.current_offset, 2),
                "processing_time": round(processing_time, 4),
                "detected_speech_segments": speech_segments,
                "speech_duration": round(total_speech_duration, 2),
            }
        )
        return vad_annotation

    return wrapper


def setup_logger(log_dir, filename, log_type):
    """
    Creates a logger for a specific type of logging (timing, memory, results).
    """
    log_path = os.path.join(log_dir, filename, f"{filename}_{log_type}.log")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"{log_type}_{filename}")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Avoid duplicate handlers
    file_handler = logging.FileHandler(log_path, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger
