import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
import traceback

# Load our core modules
from core.audio_loader import AudioLoader
from core.chopping_engine import ChoppingEngine
from core.improved_generator import ImprovedSampleGenerator
from core.splice_scraper import SpliceLoopScraper
from core.playwright_scraper import PlaywrightScraper
from core.processing_pipeline import ProcessingPipeline

# Load environment variables from .env file if it exists
load_dotenv()

def setup_directories():
    """Create necessary project directories"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/processed/mel",
        "data/processed/segments",
        "models",
        "output"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Created project directories: {', '.join(dirs)}")

def download_samples(source="looperman", query="drum loop", count=50):
    """Download samples from the specified source"""
    print(f"Downloading {count} samples matching '{query}' from {source}...")
    
    try:
        if source.lower() == "splice":
            scraper = SpliceLoopScraper(download_dir="data/raw")
            samples = scraper.bulk_download(query, count)
            print(f"Downloaded {len(samples)} samples from Splice")
            return [item["local_path"] for item in samples]
            
        elif source.lower() == "looperman":
            scraper = PlaywrightScraper(download_dir="data/raw")
            samples = scraper.bulk_download(query, pages=count // 15 + 1)
            print(f"Downloaded {len(samples)} samples from Looperman")
            return samples
            
        else:
            print(f"Unknown source: {source}")
            return []
            
    except Exception as e:
        print(f"Error downloading samples: {e}")
        traceback.print_exc()
        return []

def process_samples(sample_paths):
    """Process downloaded samples for ML training"""
    print(f"Processing {len(sample_paths)} samples...")
    
    pipeline = ProcessingPipeline()
    results = pipeline.process_batch(sample_paths)
    
    metadata = pipeline.create_dataset_metadata(results)
    print(f"Processed {metadata['successful']} files successfully")
    print(f"Created {metadata['total_segments']} segments")
    
    if metadata['failed'] > 0:
        print(f"Failed to process {metadata['failed']} files")
    
    return metadata

def train_model(dataset_paths, epochs=100):
    """Train the generator model on processed samples"""
    print(f"Training generator model on {len(dataset_paths)} samples for {epochs} epochs...")
    
    generator = ImprovedSampleGenerator()
    generator.train(dataset_paths, epochs=epochs, save_path="models/generator.pt")
    
    print("Model training complete")
    return generator

def generate_samples(count=5):
    """Generate new samples using the trained model"""
    print(f"Generating {count} new samples...")
    
    generator = ImprovedSampleGenerator(model_path="models/generator.pt")
    
    for i in range(count):
        output_path = f"output/generated_sample_{i}.wav"
        generator.generate_and_save_audio(output_path)
        print(f"Generated: {output_path}")

def generate_with_rl(target_file=None, steps=50):
    """Generate new samples using RL optimization"""
    print(f"Generating sample with RL optimization...")
    
    generator = ImprovedSampleGenerator(model_path="models/generator.pt")
    
    # Generate with RL
    audio, _ = generator.use_rl_optimization(target_audio_path=target_file, n_steps=steps)
    
    # Save the result
    output_path = "output/rl_optimized_sample.wav"
    generator.save_audio(audio, output_path)
    print(f"Generated RL-optimized sample: {output_path}")
    
    return output_path

def chop_audio(file_path, method="beat"):
    """Chop an audio file using the specified method"""
    print(f"Chopping audio file: {file_path} using method: {method}")
    
    loader = AudioLoader()
    audio, sr = loader.load_audio(file_path)
    
    if audio is None:
        print("Failed to load audio file")
        return []
    
    chopper = ChoppingEngine(sample_rate=sr)
    
    if method == "beat":
        segments = chopper.chop_by_beats(audio)
        print(f"Found {len(segments)} beat segments")
    else:
        segments = chopper.chop_by_silence(audio)
        print(f"Found {len(segments)} silence-based segments")
    
    return segments

def main():
    parser = argparse.ArgumentParser(description='AI Sample Chopper')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up project structure')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download samples')
    download_parser.add_argument('--source', '-s', default='looperman',
                              choices=['looperman', 'splice'],
                              help='Source for samples')
    download_parser.add_argument('--query', '-q', default='drum loop',
                              help='Search query')
    download_parser.add_argument('--count', '-c', type=int, default=50,
                              help='Number of samples to download')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process downloaded samples')
    process_parser.add_argument('--dir', '-d', default='data/raw',
                             help='Directory containing samples')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train generator model')
    train_parser.add_argument('--epochs', '-e', type=int, default=100,
                           help='Number of training epochs')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate new samples')
    generate_parser.add_argument('--count', '-c', type=int, default=5,
                              help='Number of samples to generate')
    
    # Chop command
    chop_parser = subparsers.add_parser('chop', help='Chop an audio sample')
    chop_parser.add_argument('file', help='Audio file to chop')
    chop_parser.add_argument('--method', '-m', default='beat',
                          choices=['beat', 'silence'],
                          help='Chopping method')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'setup':
        setup_directories()
    
    elif args.command == 'download':
        download_samples(args.source, args.query, args.count)
    
    elif args.command == 'process':
        sample_dir = Path(args.dir)
        sample_paths = [str(f) for f in sample_dir.glob('*') if f.is_file() and f.suffix.lower() in ('.wav', '.mp3', '.ogg')]
        process_samples(sample_paths)
    
    elif args.command == 'train':
        # Load processed dataset paths from metadata
        try:
            import numpy as np
            metadata = np.load('data/processed/dataset_metadata.npy', allow_pickle=True).item()
            train_model(metadata['segment_paths'], args.epochs)
        except (FileNotFoundError, KeyError):
            print("No processed dataset found. Run 'process' command first.")
    
    elif args.command == 'generate':
        generate_samples(args.count)
    
    elif args.command == 'chop':
        segments = chop_audio(args.file, args.method)
        print(f"Chop points (seconds): {segments}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
