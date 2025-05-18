import librosa
import numpy as np

class ChoppingEngine:
    """
    A class to handle chopping of audio based on silence detection or beat tracking.
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def chop_by_silence(self, audio, top_db=20):
        """
        Chop audio into segments based on silence detection.

        Args:
            audio (numpy.ndarray): Audio time series.
            top_db (int): The threshold (in decibels) below reference to consider as silence.

        Returns:
            list: A list of tuples representing start and end times of segments.
        """
        intervals = librosa.effects.split(audio, top_db=top_db)
        return [(start / self.sample_rate, end / self.sample_rate) for start, end in intervals]

    def chop_by_beats(self, audio):
        """
        Chop audio into segments based on beat tracking.

        Args:
            audio (numpy.ndarray): Audio time series.

        Returns:
            list: A list of beat times in seconds.
        """
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        return beat_times
