from typing import Optional

import numpy as np
import torch
from pyannote.core import Annotation, Segment
from silero_vad import VADIterator, get_speech_timestamps, load_silero_vad

from utils import time_it
from vad_models.vad_base import VADBase


class SileroVAD(VADBase):
    def __init__(
        self,
        chunk_size: float,
        overlap: float,
        sample_rate: int,
        label_file_path: Optional[str] = None,
    ):
        super().__init__(chunk_size, overlap, sample_rate, label_file_path)

        # Load Silero VAD model and initialize VADIterator
        self.model = load_silero_vad()
        self.vad_iterator = VADIterator(self.model)

    @time_it
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Annotation:
        """
        Processes a single audio chunk using Silero VAD and \
            returns an Annotation object.
        """
        if len(audio_chunk.shape) == 1:  # Convert mono to (1, time)
            audio_chunk = np.expand_dims(audio_chunk, axis=0)

        # Convert to PyTorch tensor
        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)

        # Run Silero VAD using get_speech_timestamps()
        speech_timestamps = get_speech_timestamps(
            audio_tensor, self.model, sampling_rate=self.sample_rate
        )

        vad_annotation = Annotation()

        for ts in speech_timestamps:
            start_time = ts["start"] / self.sample_rate
            end_time = ts["end"] / self.sample_rate

            adjusted_segment = Segment(
                start_time + self.current_offset, end_time + self.current_offset
            )
            vad_annotation[adjusted_segment] = "speech"

        self.vad_segments.append(vad_annotation)

        # Update offset for next chunk
        self.current_offset += self.chunk_size - self.overlap

        return vad_annotation  # Return current chunk's VAD result
