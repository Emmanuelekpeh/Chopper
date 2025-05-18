import os
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Tuple
import concurrent.futures

class ProcessingPipeline:
    """
    A processing pipeline to prepare and transform audio samples for model training.
    Includes preprocessing, augmentation, and batch generation capabilities.
    """
    
    def __init__(self, sample_rate=44100, n_mels=128, hop_length=512, 
                 segment_duration=2.0, output_dir="data/processed"):
        """
        Initialize the processing pipeline.
        
        Args:
            sample_rate: Target sample rate
            n_mels: Number of mel bands
            hop_length: Hop length for STFT
            segment_duration: Duration in seconds for fixed-length segments
            output_dir: Directory to save processed files
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sample_rate)
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/mel", exist_ok=True)
        os.makedirs(f"{output_dir}/segments", exist_ok=True)
    
    def process_file(self, audio_path: str) -> Dict:
        """
        Process a single audio file.
        
        Returns:
            Dict with metadata and paths to processed files
        """
        try:
            # Extract filename without extension
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Load and resample audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Apply preprocessing
            audio = self._preprocess_audio(audio)
            
            # Segment the audio
            segments = self._segment_audio(audio)
            
            # Create mel spectrograms
            mel_specs = [self._create_mel_spectrogram(seg) for seg in segments]
            
            # Save segments and spectrograms
            segment_paths = []
            mel_paths = []
            
            for i, (segment, mel_spec) in enumerate(zip(segments, mel_specs)):
                # Save audio segment
                segment_path = f"{self.output_dir}/segments/{filename}_{i}.wav"
                sf.write(segment_path, segment, self.sample_rate)
                segment_paths.append(segment_path)
                
                # Save mel spectrogram
                mel_path = f"{self.output_dir}/mel/{filename}_{i}.npy"
                np.save(mel_path, mel_spec)
                mel_paths.append(mel_path)
            
            return {
                "original": audio_path,
                "filename": filename,
                "duration": len(audio) / self.sample_rate,
                "n_segments": len(segments),
                "segment_paths": segment_paths,
                "mel_paths": mel_paths
            }
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return {
                "original": audio_path,
                "error": str(e)
            }
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply preprocessing to audio"""
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Apply high-pass filter to remove rumble
        audio = librosa.effects.preemphasis(audio)
        
        return audio
    
    def _segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """Split audio into fixed-length segments"""
        segments = []
        
        # If audio is shorter than segment duration, pad it
        if len(audio) < self.segment_samples:
            audio = np.pad(audio, (0, self.segment_samples - len(audio)))
            segments.append(audio)
        else:
            # Split into segments
            for i in range(0, len(audio) - self.segment_samples, self.segment_samples // 2):  # 50% overlap
                segment = audio[i:i + self.segment_samples]
                if len(segment) == self.segment_samples:  # Ensure full length
                    segments.append(segment)
        
        return segments
    
    def _create_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Create mel spectrogram from audio segment"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=2048
        )
        
        # Convert to log scale (dB)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel
    
    def apply_augmentations(self, audio: np.ndarray) -> List[np.ndarray]:
        """Apply audio augmentations to increase dataset variety"""
        augmented = []
        
        # Time stretching (0.9x and 1.1x speed)
        augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
        augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
        
        # Pitch shifting (+/- 2 semitones)
        augmented.append(librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=2))
        augmented.append(librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=-2))
        
        return augmented
    
    def process_batch(self, audio_paths: List[str], n_workers: int = 4) -> List[Dict]:
        """
        Process multiple audio files in parallel.
        
        Args:
            audio_paths: List of paths to audio files
            n_workers: Number of parallel workers
            
        Returns:
            List of metadata dictionaries for all processed files
        """
        results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self.process_file, path) for path in audio_paths]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Worker process failed: {e}")
        
        return results
    
    def create_dataset_metadata(self, results: List[Dict]) -> Dict:
        """
        Compile metadata about the processed dataset
        
        Args:
            results: List of processing results
            
        Returns:
            Dictionary with dataset metadata
        """
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        # Flatten segment paths
        all_segments = []
        all_mels = []
        for result in successful:
            all_segments.extend(result["segment_paths"])
            all_mels.extend(result["mel_paths"])
        
        metadata = {
            "total_files_processed": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_segments": len(all_segments),
            "total_duration_hours": sum(r["duration"] for r in successful) / 3600,
            "segment_paths": all_segments,
            "mel_paths": all_mels,
            "failed_files": [r["original"] for r in failed]
        }
        
        # Save metadata
        np.save(f"{self.output_dir}/dataset_metadata.npy", metadata)
        
        return metadata
