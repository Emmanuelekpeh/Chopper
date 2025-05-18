import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import librosa

class RLSampleGeneratorAgent:
    """
    Reinforcement Learning agent for generating audio samples with desired characteristics.
    Uses Deep Q-Learning to optimize a latent space exploration strategy.
    """
    
    def __init__(self, generator, latent_dim=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the RL agent with a pre-trained generator.
        
        Args:
            generator: Trained generator model (e.g., AudioGAN's generator)
            latent_dim: Dimensionality of the latent space
            device: Computation device (CPU/GPU)
        """
        self.generator = generator
        self.latent_dim = latent_dim
        self.device = device
        
        # Create Q-network for RL
        self.q_network = self._build_q_network().to(device)
        self.target_network = self._build_q_network().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Feature extractors for reward calculation
        self.mel_extractor = lambda audio, sr: librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, hop_length=512
        )
        
    def _build_q_network(self):
        """Build the Q-network for RL decision making"""
        # Simple MLP for Q-value prediction from latent vectors
        model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim * 2)  # Actions: move in positive/negative direction for each dimension
        )
        return model
    
    def get_action(self, state):
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: Current state (latent vector)
            
        Returns:
            Action vector to modify the latent space
        """
        if np.random.rand() <= self.epsilon:
            # Exploration: random action
            return torch.randn(self.latent_dim).to(self.device) * 0.1
        else:
            # Exploitation: use Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze()
            
            # Convert Q-values to actions
            actions = torch.zeros(self.latent_dim).to(self.device)
            for i in range(self.latent_dim):
                # For each dimension, choose positive or negative movement
                pos_idx = i
                neg_idx = i + self.latent_dim
                
                if q_values[pos_idx] > q_values[neg_idx]:
                    actions[i] = 0.1  # Small positive step
                else:
                    actions[i] = -0.1  # Small negative step
            
            return actions
    
    def calculate_reward(self, audio, sr, target_features=None):
        """
        Calculate reward based on audio quality and target features
        
        Args:
            audio: Generated audio
            sr: Sample rate
            target_features: Target audio features to match (optional)
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Basic reward based on audio quality
        # 1. Avoid silence
        energy = np.mean(np.abs(audio))
        reward += 2.0 if energy > 0.01 else -1.0
        
        # 2. Avoid clipping
        if np.max(np.abs(audio)) > 0.99:
            reward -= 1.0
        
        # 3. Reward rhythmic structure (using onset strength)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        # Reward higher onset variance (indicates rhythm)
        onset_var = np.var(onset_env)
        reward += np.clip(onset_var * 10, 0, 2.0)
        
        # 4. If target features provided, reward similarity
        if target_features is not None:
            mel_spec = self.mel_extractor(audio, sr)
            
            # If sizes don't match, resize for comparison
            if mel_spec.shape != target_features.shape:
                min_time = min(mel_spec.shape[1], target_features.shape[1])
                mel_spec = mel_spec[:, :min_time]
                target_features = target_features[:, :min_time]
            
            # Calculate cosine similarity
            mel_flat = mel_spec.flatten()
            target_flat = target_features.flatten()
            similarity = np.dot(mel_flat, target_flat) / (np.linalg.norm(mel_flat) * np.linalg.norm(target_flat))
            
            # Map similarity [0,1] to reward [-1,5]
            reward += (similarity * 6) - 1
        
        return reward
    
    def store_transition(self, state, action, reward, next_state):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state))
    
    def train(self, batch_size=64):
        """Train the Q-network using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample random batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # Get Q-values for current states and actions
        q_values = self.q_network(states)
        
        # Get target Q-values from target network
        with torch.no_grad():
            target_q = rewards + self.gamma * torch.max(self.target_network(next_states), dim=1, keepdim=True)[0]
        
        # Calculate loss
        loss = F.mse_loss(q_values, target_q)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if np.random.rand() < 0.01:  # Soft update occasionally
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def generate_sample_with_rl(self, n_steps=50, target_features=None, sr=44100):
        """
        Generate audio sample using RL-guided latent space exploration
        
        Args:
            n_steps: Number of RL steps
            target_features: Target audio features to match (optional)
            sr: Sample rate
            
        Returns:
            The best generated audio and its latent vector
        """
        # Initial state (random latent vector)
        state = torch.randn(self.latent_dim).to(self.device)
        best_state = state.clone()
        best_reward = -float('inf')
        best_audio = None
        
        # RL optimization loop
        for step in range(n_steps):
            # Generate audio from current state
            with torch.no_grad():
                latent = state.unsqueeze(0)
                mel_spec = self.generator(latent)
                audio = self._convert_mel_to_audio(mel_spec.squeeze(), sr)
            
            # Calculate reward
            reward = self.calculate_reward(audio, sr, target_features)
            
            # Keep track of best result
            if reward > best_reward:
                best_reward = reward
                best_state = state.clone()
                best_audio = audio
            
            # Get action from policy
            action = self.get_action(state.cpu().numpy())
            
            # Apply action to get next state
            next_state = state + action
            next_state = torch.clamp(next_state, -2.0, 2.0)  # Keep in reasonable range
            
            # Store transition in replay buffer
            self.store_transition(state.cpu().numpy(), action.cpu().numpy(), 
                                 reward, next_state.cpu().numpy())
            
            # Move to next state
            state = next_state
            
            # Train Q-network occasionally
            if step % 5 == 0 and len(self.replay_buffer) >= 64:
                self.train(batch_size=64)
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step}/{n_steps}, Current reward: {reward:.4f}, Best reward: {best_reward:.4f}")
        
        return best_audio, best_state
    
    def _convert_mel_to_audio(self, mel_spec, sr=44100, hop_length=512):
        """Convert mel spectrogram to audio"""
        # Convert from tensor to numpy
        if isinstance(mel_spec, torch.Tensor):
            mel_spec = mel_spec.detach().cpu().numpy()
        
        # Reshape if needed
        if len(mel_spec.shape) == 3 and mel_spec.shape[0] == 1:
            mel_spec = mel_spec[0]
        
        # Rescale from [-1, 1] (tanh output) to appropriate range for librosa
        mel_spec = (mel_spec + 1) / 2  # Scale to [0, 1]
        mel_spec = librosa.db_to_power(mel_spec * 80 - 10)  # Convert to linear scale
        
        # Griffin-Lim reconstruction
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec, sr=sr, hop_length=hop_length, n_fft=2048
        )
        
        return audio
    
    def save(self, path):
        """Save the RL model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load the RL model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']
