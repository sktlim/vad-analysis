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
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        super().__init__(
            chunk_size, overlap, sample_rate, label_file_path, output_dir, filename
        )

        # Load Silero VAD model and initialize VADIterator
        self.model = load_silero_vad()
        self.vad_iterator = VADIterator(self.model)

        self.frame_probabilities = []

    def get_speech_probabilities(self, total_duration: float) -> np.ndarray:
        expected_frames = int(total_duration / self.frame_duration)
        padded_probs = np.pad(
            self.frame_probabilities,
            (0, max(0, expected_frames - len(self.frame_probabilities))),
            mode="constant",
            constant_values=0,
        )
        return padded_probs[:expected_frames]

    @time_it
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Annotation:
        if len(audio_chunk.shape) == 1:
            audio_chunk = np.expand_dims(audio_chunk, axis=0)

        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)

        # Reset VAD iterator for each chunk (since we’re chunking manually)
        self.vad_iterator.reset_states()

        chunk_probs = []

        for t in range(audio_tensor.shape[1]):
            sample = audio_tensor[:, t : t + 1]
            prob = self.vad_iterator(sample, return_probability=True).item()
            chunk_probs.append(prob)

        self.frame_probabilities.extend(chunk_probs)

        # Binary labels for segmenting speech
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
        self.current_offset += self.chunk_size - self.overlap

        return vad_annotation

