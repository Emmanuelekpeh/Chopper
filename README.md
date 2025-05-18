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

## Collaborative Development & Online Deployment

This section explains how to collaborate with others on training, inference, and continuous tweaking.

### Option 1: Google Colab (Easiest)

1. Open the included `chopper_colab.ipynb` notebook in Google Colab:
   - Upload the notebook to Colab
   - Share it directly with your collaborators

2. The notebook will:
   - Clone the repository
   - Install dependencies
   - Allow sample uploading/downloading
   - Enable training and inference
   - Save models to Google Drive or push back to GitHub

3. Sharing models:
   - Save to Google Drive and share the folder
   - Push trained models to the GitHub repo

### Option 2: Streamlit Web App

Run the included Streamlit app locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or deploy it to a cloud service:

1. **Streamlit Cloud**:
   - Sign up at https://streamlit.io/cloud
   - Connect your GitHub repo
   - Select streamlit_app.py as the entry point

2. **Heroku**:
   - Add a Procfile: `web: streamlit run streamlit_app.py --server.port=$PORT --server.headless=true`
   - Deploy through Heroku CLI or GitHub integration

### Option 3: Docker Deployment

For more control over the environment:

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or manually with Docker
docker build -t chopper-app .
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models chopper-app
```

### Option 4: Hugging Face Hub

For model and demo sharing:

1. Install the Hugging Face Hub client:
   ```bash
   pip install huggingface-hub
   ```

2. Upload your model:
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   
   # Login (run once)
   api.login()
   
   # Upload model
   api.upload_file(
       path_or_fileobj="models/transformer_gan.pt",
       path_in_repo="transformer_gan.pt",
       repo_id="your-username/chopper",
       repo_type="model"
   )
   ```

3. Create a Hugging Face Space with the Streamlit app for online inference

### Workflow for Continuous Collaboration

1. **Version Control**: All code changes through Git
   ```bash
   git pull  # Get latest changes
   git add .  # Stage your changes
   git commit -m "Description of changes"
   git push  # Share your changes
   ```

2. **Model Sharing**: Store iterations on Hugging Face or share through Google Drive
   
3. **Regular Sync**: Schedule calls to discuss improvements and coordinate training tasks

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
