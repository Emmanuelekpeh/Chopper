from core.improved_generator import ImprovedSampleGenerator
from core.transformer_generator import TransformerSampleGenerator
from core.rl_generator import RLSampleGeneratorAgent
import torch
import numpy as np
import argparse
import os
import json

def generate_samples_with_models(output_dir="output", count=5):
    """
    Generate samples using all available model architectures for comparison.
    
    Args:
        output_dir: Directory to save generated samples
        count: Number of samples to generate per model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {count} samples with each model architecture...")
    
    # GAN-based generator
    print("\n=== GAN-Based Generator ===")
    if os.path.exists("models/audiogan.pt"):
        gan_generator = ImprovedSampleGenerator(model_path="models/audiogan.pt")
        for i in range(count):
            output_path = f"{output_dir}/gan_sample_{i}.wav"
            gan_generator.generate_and_save_audio(output_path)
            print(f"Generated: {output_path}")
    else:
        print("GAN model not found at models/audiogan.pt - skipping")
    
    # Transformer-based generator
    print("\n=== Transformer-Based Generator ===")
    if os.path.exists("models/transformer_gan.pt"):
        transformer_generator = TransformerSampleGenerator(model_path="models/transformer_gan.pt")
        for i in range(count):
            output_path = f"{output_dir}/transformer_sample_{i}.wav"
            transformer_generator.generate_and_save_audio(output_path)
            print(f"Generated: {output_path}")
    else:
        print("Transformer model not found at models/transformer_gan.pt - skipping")
    
    # RL-enhanced generation (using base GAN model)
    print("\n=== RL-Enhanced Generator ===")
    if os.path.exists("models/audiogan.pt"):
        gan_generator = ImprovedSampleGenerator(model_path="models/audiogan.pt")
        for i in range(count):
            output_path = f"{output_dir}/rl_enhanced_sample_{i}.wav"
            audio, _ = gan_generator.use_rl_optimization(n_steps=30)
            gan_generator.save_audio(audio, output_path)
            print(f"Generated: {output_path}")
    else:
        print("GAN model not found for RL enhancement - skipping")

def train_transformer_model(audio_files, epochs=100):
    """
    Train the transformer-based generator model.
    
    Args:
        audio_files: List of audio files to train on
        epochs: Number of training epochs
    """
    generator = TransformerSampleGenerator()
    generator.train(audio_files, epochs=epochs, save_path="models/transformer_gan.pt")
    print("Transformer model training complete.")
    return generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Sample Generation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate samples with all models')
    generate_parser.add_argument('--count', '-c', type=int, default=3,
                             help='Number of samples to generate per model')
    
    # Train transformer command
    train_parser = subparsers.add_parser('train', help='Train transformer model')
    train_parser.add_argument('--epochs', '-e', type=int, default=100,
                           help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_samples_with_models(count=args.count)
    elif args.command == 'train':
        # Load processed dataset paths from metadata
        try:
            with open('data/processed/dataset_metadata.json', 'r') as f:
                metadata = json.load(f)
            train_transformer_model(metadata['segment_paths'], args.epochs)
        except (FileNotFoundError, KeyError) as e:
            print(f"Error loading dataset metadata: {e}")
            print("No processed dataset found. Run 'python app.py process' first.")
    else:
        parser.print_help()
