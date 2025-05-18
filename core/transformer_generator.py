import torch
import torch.nn as nn
import numpy as np
import librosa

class AudioTransformerGenerator(nn.Module):
    """
    Transformer-based architecture for generating audio samples.
    Combines autoregressive transformer modeling with GAN-based generation.
    """
    
    def __init__(self, latent_dim=100, n_mels=128, sequence_length=128, 
                 nhead=8, num_layers=6, d_model=512):
        super(AudioTransformerGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Initial projection from latent space to sequence
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        
        # Final projection to mel spectrogram
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, n_mels),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, z):
        """Transform latent vector into mel spectrogram sequence"""
        batch_size = z.shape[0]
        
        # Project latent vector
        x = self.latent_projection(z)
        
        # Expand to sequence
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Project to mel bands
        mel = self.output_projection(x)  # [batch, seq_len, n_mels]
        
        # Transpose to expected format [batch, 1, n_mels, seq_len]
        mel = mel.transpose(1, 2).unsqueeze(1)
        
        return mel


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class HybridTransformerGAN(nn.Module):
    """
    Hybrid model combining transformer-based generation with GAN discriminator.
    """
    
    def __init__(self, latent_dim=100, n_mels=128, sequence_length=128):
        super(HybridTransformerGAN, self).__init__()
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.sequence_length = sequence_length
        
        # Transformer-based generator
        self.generator = AudioTransformerGenerator(
            latent_dim=latent_dim,
            n_mels=n_mels,
            sequence_length=sequence_length
        )
        
        # CNN-based discriminator
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
            nn.Linear(256 * 8 * 8, 1),  # Adjust size based on the dimensions
            nn.Sigmoid()
        )
    
    def generate(self, z):
        """Generate a mel spectrogram from latent vector z"""
        return self.generator(z)
    
    def discriminate(self, mel_spec):
        """Discriminate between real and fake mel spectrograms"""
        return self.discriminator(mel_spec)


class TransformerSampleGenerator:
    """
    Transformer-based sample generator with improved architecture and training
    for more unique, structured, and coherent audio generation.
    """
    
    def __init__(self, n_mels=128, sequence_length=128, latent_dim=100, 
                 model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_mels = n_mels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.device = device
        self.model = HybridTransformerGAN(latent_dim, n_mels, sequence_length).to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
    
    def train(self, audio_files, batch_size=16, epochs=100, lr=0.0002, save_path='models/transformer_gan.pt'):
        """
        Train the transformer-based GAN model
        
        Args:
            audio_files (list): List of audio file paths
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            lr (float): Learning rate
            save_path (str): Path to save the trained model
        """
        import os
        from torch.utils.data import DataLoader
        from improved_generator import AudioMelDataset
        import torch.nn.functional as F
        
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
                
                # Add feature matching loss for better quality
                real_features = self.extract_features(real_mels)
                fake_features = self.extract_features(fake_mels)
                feature_loss = F.mse_loss(fake_features, real_features)
                
                # Combined loss
                combined_loss = g_loss + 10 * feature_loss
                combined_loss.backward()
                optimizer_G.step()
                
                # Print progress
                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                          f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
            
            # Save model after each epoch
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    def extract_features(self, mel_spec):
        """Extract intermediate features from discriminator for feature matching loss"""
        # Get features from intermediate layer of discriminator
        # This is a simplified version - actual implementation would access an intermediate layer
        return self.model.discriminator[0:8](mel_spec)
    
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
    
    def generate_and_save_audio(self, filepath, sr=44100):
        """Generate a sample and save it as audio file"""
        import soundfile as sf
        import os
        
        mel_spec = self.generate_sample(1)
        audio = self.mel_to_audio(mel_spec, sr)
        
        # Save the audio
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        sf.write(filepath, audio, sr)
        
        return filepath
