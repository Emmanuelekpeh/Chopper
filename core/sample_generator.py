import numpy as np
from tensorflow.keras import layers, models

class SampleGenerator:
    """
    A class to handle AI-based sample generation using a simple Variational Autoencoder (VAE).
    """

    def __init__(self, input_shape=(128, 128, 1), latent_dim=16):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder, self.decoder, self.vae = self._build_vae()

    def _build_vae(self):
        """
        Build the VAE model (encoder, decoder, and combined VAE).

        Returns:
            tuple: (encoder model, decoder model, VAE model)
        """
        # Encoder
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(128, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = np.random.normal(size=(self.latent_dim,))
            return z_mean + np.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling)([z_mean, z_log_var])

        # Decoder
        decoder_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(128, activation='relu')(decoder_input)
        x = layers.Dense(np.prod(self.input_shape), activation='sigmoid')(x)
        outputs = layers.Reshape(self.input_shape)(x)

        encoder = models.Model(inputs, z, name="encoder")
        decoder = models.Model(decoder_input, outputs, name="decoder")
        vae = models.Model(inputs, decoder(encoder(inputs)), name="vae")

        return encoder, decoder, vae

    def train(self, data, epochs=50, batch_size=16):
        """
        Train the VAE model.

        Args:
            data (numpy.ndarray): Training data.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        self.vae.compile(optimizer='adam', loss='mse')
        self.vae.fit(data, data, epochs=epochs, batch_size=batch_size)

    def generate_sample(self):
        """
        Generate a new sample by sampling from the latent space.

        Returns:
            numpy.ndarray: Generated sample.
        """
        latent_space = np.random.normal(size=(1, self.latent_dim))
        generated_sample = self.decoder.predict(latent_space)
        return generated_sample
