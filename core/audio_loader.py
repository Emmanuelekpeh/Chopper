import librosa
import numpy as np

class AudioLoader:
    """
    A class to handle loading and preprocessing of audio files.
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def load_audio(self, file_path):
        """
        Load an audio file and return the audio time series and sample rate.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            tuple: (audio time series, sample rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None

    def get_audio_features(self, audio):
        """
        Extract basic audio features such as duration and RMS energy.

        Args:
            audio (numpy.ndarray): Audio time series.

        Returns:
            dict: A dictionary of audio features.
        """
        features = {
            "duration": librosa.get_duration(y=audio, sr=self.sample_rate),
            "rms": np.mean(librosa.feature.rms(y=audio))
        }
        return features
