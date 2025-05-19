# AI-Powered Samples Chopper

A Python tool that uses AI to chop audio samples and generate new ones with transformers and GANs.

## Features
- **Audio Loading**: Load audio files with proper resampling and preprocessing
- **Chopping Engine**: Chop samples based on silence detection or beat tracking
- **Sample Generation**: Generate new audio samples using AI models
- **Sample Scrapers**: Multiple scrapers for training data collection
- **Processing Pipeline**: Prepare and transform audio for ML training
- **Transformer Models**: Advanced audio generation with transformer architectures

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Setup project directories
python app.py setup

# Download samples from Looperman (default)
python app.py download --query "drum loop" --count 50

# Process downloaded samples for training
python app.py process

# Train the generator model
python app.py train --epochs 100

# Generate new samples
python app.py generate --count 5
```

### Streamlit Cloud Deployment
1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your repo.
3. Set `streamlit_app.py` as the entry point.
4. The app will build and deploy automatically.

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Or manually with Docker
docker build -t chopper-app .
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models chopper-app
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
├── streamlit_app.py            # Streamlit web app
├── Dockerfile                  # Docker build file
├── docker-compose.yml          # Docker Compose config
└── README.md                   # This file
```

## Requirements
- Python 3.8+
- Libraries: librosa, torch, numpy, soundfile, streamlit, playwright, etc.
- For Splice scraping: Account credentials (set as environment variables)

## Future Integrations
- FL Studio plugin compatibility
- Real-time audio processing
- UI for sample visualization and manipulation

## License
MIT License (or specify your license here)
