# Voice Pitch Trainer

Realtime voice and singing trainer with pitch detection, melody analysis and AI vocal separation.

# Features

- Realtime microphone pitch detection
- Voice training mode
- Singing training mode with realtime accuracy tracking
- Automatic singing latency compensation
- Intelligent microphone noise filtering
- Melody analysis and pitch visualization
- AI vocal separation using Demucs
- Export vocals and instrumental separately
- YouTube song import support
- Recent songs and melody cache
- Singing rank system (S+ to E)
- Singing history and progress tracking
- Modern PyQt6 interface with SVG icons

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Build EXE

```bash
pyinstaller --onefile --windowed --name VoicePitchTrainer main.py
```
