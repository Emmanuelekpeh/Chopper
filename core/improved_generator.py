import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os

class AudioMelDataset(Dataset):
    """Dataset for loading audio files and converting to mel spectrograms"""
    
    def __init__(self, audio_files, sequence_length=128, hop_length=512, n_mels=128, 
                 sample_rate=44100, transforms=None):
        """
        Initialize the dataset.
        
        Args:
            audio_files (list): List of audio file paths
            sequence_length (int): Length of sequences to extract (time dimension)
            hop_length (int): Hop length for STFT
            n_mels (int): Number of mel bands
            sample_rate (int): Sample rate to resample audio to
            transforms (callable): Optional transforms to apply
        """
        self.audio_files = audio_files
        self.sequence_length = sequence_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.transforms = transforms
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_files[idx]
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        
        # Convert to log scale (dB)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1]
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        log_mel = np.clip(log_mel, -1, 1)
        
        # Pad or trim to sequence length
        if log_mel.shape[1] < self.sequence_length:
            pad_width = self.sequence_length - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        elif log_mel.shape[1] > self.sequence_length:
            # Take a random sequence of required length
            start = np.random.randint(0, log_mel.shape[1] - self.sequence_length)
            log_mel = log_mel[:, start:start + self.sequence_length]
        
        # Add channel dimension
        log_mel = log_mel.reshape(1, self.n_mels, self.sequence_length)
        
        # Apply transforms if any
        if self.transforms:
            log_mel = self.transforms(log_mel)
        
        return torch.FloatTensor(log_mel)


class AudioGAN(nn.Module):
    """
    An improved GAN-based model for generating audio samples,
    more suitable for detailed audio textures like drum loops.
    """
    
    def __init__(self, latent_dim=100, n_mels=128, sequence_length=128):
        super(AudioGAN, self).__init__()
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.sequence_length = sequence_length
        
        # Generator
        self.generator = nn.Sequential(
            # Project and reshape latent vector
            nn.Linear(latent_dim, 128 * 16 * 8),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, 16, 8)),  # (batch, 128, 16, 8)
            
            # Layer 1: Upsample to (128, 32, 16)
            nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 2: Upsample to (64, 64, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: Upsample to (32, 128, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 4: Final upsample to (1, n_mels, sequence_length) - adjust for exact dimensions
            nn.ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=(1, 2), padding=(1, 1)),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Flatten and classify
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),  # Adjust size based on the exact dimensions
            nn.Sigmoid()
        )
    
    def generate(self, z):
        """Generate a mel spectrogram from latent vector z"""
        return self.generator(z)
    
    def discriminate(self, mel_spec):
        """Discriminate between real and fake mel spectrograms"""
        return self.discriminator(mel_spec)


class ImprovedSampleGenerator:
    """
    An improved sample generator with better training procedures,
    mel spectrogram conversion, and audio reconstruction.
    """
    
    def __init__(self, n_mels=128, sequence_length=128, latent_dim=100, 
                 model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_mels = n_mels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.device = device
        self.model = AudioGAN(latent_dim, n_mels, sequence_length).to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
    
    def train(self, audio_files, batch_size=16, epochs=100, lr=0.0002, save_path='models/audiogan.pt'):
        """
        Train the GAN model on audio files
        
        Args:
            audio_files (list): List of audio file paths
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            lr (float): Learning rate
            save_path (str): Path to save the trained model
        """
        # Create directory for model if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create dataset and dataloader
        dataset = AudioMelDataset(audio_files, self.sequence_length, n_mels=self.n_mels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Optimizers
        optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Training loop
        for epoch in range(epochs):
            for i, real_mels in enumerate(dataloader):
                batch_size = real_mels.size(0)
                real_mels = real_mels.to(self.device)
                
                # Generate latent vectors
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                # Real samples
                real_pred = self.model.discriminate(real_mels)
                real_loss = F.binary_cross_entropy(real_pred, torch.ones(batch_size, 1, device=self.device))
                
                # Fake samples
                fake_mels = self.model.generate(z)
                fake_pred = self.model.discriminate(fake_mels.detach())
                fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros(batch_size, 1, device=self.device))
                
                # Total discriminator loss
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                
                # Generate fake samples again (for generator training)
                fake_pred = self.model.discriminate(fake_mels)
                g_loss = F.binary_cross_entropy(fake_pred, torch.ones(batch_size, 1, device=self.device))
                
                g_loss.backward()
                optimizer_G.step()
                
                # Print progress
                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                          f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
            
            # Save model after each epoch
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    def generate_sample(self, n_samples=1):
        """
        Generate mel spectrograms
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            torch.Tensor: Generated mel spectrograms
        """
        # Generate latent vectors
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        
        # Generate mel spectrograms
        self.model.eval()
        with torch.no_grad():
            mels = self.model.generate(z)
        
        return mels
    
    def mel_to_audio(self, mel_spec, sr=44100, hop_length=512):
        """
        Convert mel spectrogram to audio using Griffin-Lim algorithm
        
        Args:
            mel_spec (torch.Tensor): Mel spectrogram (1, n_mels, time)
            sr (int): Sample rate
            hop_length (int): Hop length
            
        Returns:
            numpy.ndarray: Audio signal
        """
        # Convert to numpy and squeeze dimensions
        mel = mel_spec.cpu().squeeze().numpy()
        
        # Rescale from [-1, 1] to original scale
        mel = librosa.db_to_power(mel * 80 - 10)
        
        # Griffin-Lim reconstruction
        audio = librosa.feature.inverse.mel_to_audio(
            mel, sr=sr, hop_length=hop_length, n_fft=2048
        )
        
        return audio
    
    def save_audio(self, audio_signal, filepath, sr=44100):
        """Save audio signal to file"""
        import soundfile as sf
        sf.write(filepath, audio_signal, sr)
    
    def generate_and_save_audio(self, filepath, sr=44100):
        """Generate a sample and save it as audio file"""
        mel_spec = self.generate_sample(1)
        audio = self.mel_to_audio(mel_spec, sr)
        self.save_audio(audio, filepath, sr)
        return filepath
        
    def use_rl_optimization(self, target_audio_path=None, n_steps=50, sr=44100):
        """
        Use Reinforcement Learning to optimize sample generation
        
        Args:
            target_audio_path: Optional path to target audio to mimic
            n_steps: Number of optimization steps
            sr: Sample rate
            
        Returns:
            Generated audio optimized with RL
        """
        from core.rl_generator import RLSampleGeneratorAgent
        
        # Initialize RL agent with our generator
        rl_agent = RLSampleGeneratorAgent(self.model.generator, 
                                          latent_dim=self.latent_dim, 
                                          device=self.device)
        
        # Load target features if provided
        target_features = None
        if target_audio_path:
            target_audio, _ = librosa.load(target_audio_path, sr=sr)
            target_features = librosa.feature.melspectrogram(
                y=target_audio, sr=sr, n_mels=self.n_mels, hop_length=512
            )
        
        # Generate sample using RL guidance
        audio, best_latent = rl_agent.generate_sample_with_rl(
            n_steps=n_steps, target_features=target_features, sr=sr
        )
        
        return audio, best_latent
