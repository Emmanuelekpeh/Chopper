# AI-Powered Samples Chopper

A Python tool that uses AI to chop audio samples and generate new ones.

## Features
- **Audio Loading**: Load audio files with proper resampling and preprocessing
- **Chopping Engine**: Chop samples based on silence detection or beat tracking
- **Sample Generation**: Generate new audio samples using AI models
- **Sample Scrapers**: Multiple scrapers for training data collection
- **Processing Pipeline**: Prepare and transform audio for ML training

## Quick Start

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup project directories
python app.py setup
```

### Download & Process Samples
```bash
# Download samples from Looperman (default)
python app.py download --query "drum loop" --count 50

# Download samples from Splice (requires login)
python app.py download --source splice --query "drum loop" --count 50

# Process downloaded samples for training
python app.py process
```

### Train & Generate
```bash
# Train the generator model
python app.py train --epochs 100

# Generate new samples
python app.py generate --count 5
```

### Chop Samples
```bash
# Chop an audio file using beat detection
python app.py chop path/to/sample.wav --method beat

# Or use silence detection
python app.py chop path/to/sample.wav --method silence
```

## Project Structure
```
chopper/
├── core/                       # Core modules
│   ├── audio_loader.py         # Audio loading and preprocessing
│   ├── chopping_engine.py      # Audio chopping methods
│   ├── improved_generator.py   # ML model for sample generation
│   ├── processing_pipeline.py  # Audio processing for training
│   ├── playwright_scraper.py   # Web scraping for Looperman
│   └── splice_scraper.py       # Web scraping for Splice
├── data/                       # Data directories
│   ├── raw/                    # Raw audio samples
│   └── processed/              # Processed samples and features
├── models/                     # Trained ML models
├── output/                     # Generated samples
├── app.py                      # Command-line interface
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements
- Python 3.8+
- Libraries: librosa, torch, tensorflow, playwright, numpy, soundfile
- For Splice scraping: Account credentials (set as environment variables)

## Future Integrations
- FL Studio plugin compatibility
- Real-time audio processing
- UI for sample visualization and manipulation
