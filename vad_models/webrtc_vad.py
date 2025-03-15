from collections import deque
from typing import Optional

import numpy as np
import webrtcvad
from pyannote.core import Annotation, Segment, Timeline

from vad_models.vad_base import VADBase
from utils import time_it

class WebRTCVAD(VADBase):
    def __init__(
        self,
        chunk_size: float,
        overlap: float,
        sample_rate: int,
        label_file_path: Optional[str] = None,
        aggressiveness: int = 2,
    ):
        super().__init__(chunk_size, overlap, sample_rate, label_file_path)

        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(
            aggressiveness
        )  # Aggressiveness: 0 (least) to 3 (most aggressive)

        # Frame settings
        self.frame_duration_ms = 30  # WebRTC VAD supports 10ms, 20ms, or 30ms frames
        self.frame_size = int(
            (self.frame_duration_ms / 1000) * self.sample_rate
        )  # Convert ms to samples

        # Store detected speech segments across chunks
        self.vad_segments: deque[Annotation] = deque()

    @time_it
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Annotation:
        """
        Processes a single audio chunk using WebRTC VAD and appends results to self.vad_segments.
        """
        if len(audio_chunk.shape) == 1:  # Convert mono to (1, time)
            audio_chunk = np.expand_dims(audio_chunk, axis=0)

        # Convert float32 audio to 16-bit PCM (WebRTC VAD requires int16)
        audio_pcm = (audio_chunk * 32768).astype(np.int16).flatten()

        vad_annotation = Annotation()

        # Iterate through frames (30ms each)
        num_frames = len(audio_pcm) // self.frame_size
        for i in range(num_frames):
            start_sample = i * self.frame_size
            end_sample = start_sample + self.frame_size

            frame = audio_pcm[start_sample:end_sample].tobytes()
            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if is_speech:
                start_time = (start_sample / self.sample_rate) + self.current_offset
                end_time = (end_sample / self.sample_rate) + self.current_offset
                vad_annotation[Segment(start_time, end_time)] = "speech"

        # Store results for aggregation
        self.vad_segments.append(vad_annotation)

        # Update offset for next chunk
        self.current_offset += self.chunk_size - self.overlap

        return vad_annotation

    def aggregate_results(self):
        """Aggregates overlapped VAD results into a single annotation object."""
        aggregated_timeline = Timeline()
        for vad_result in self.vad_segments:
            aggregated_timeline = aggregated_timeline | vad_result.get_timeline()

        self.vad_result = Annotation()
        for segment in aggregated_timeline.support():
            self.vad_result[segment] = "speech"
