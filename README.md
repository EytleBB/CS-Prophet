# CS Prophet

CS2 round-outcome prediction using a Transformer trained on demo event sequences.

## Project Structure

```
cs-prophet/
├── data/
│   ├── raw/            # Raw .dem demo files
│   └── processed/      # Parsed Parquet datasets
├── src/
│   ├── parser/         # Demo → DataFrame parsing
│   ├── features/       # Feature engineering
│   ├── model/          # Transformer architecture
│   └── inference/      # Real-time predictor
├── notebooks/          # EDA & prototyping
├── tests/              # Unit tests
├── configs/            # Training YAML configs
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Parse demos
python -m src.parser.demo_parser --input data/raw/ --output data/processed/

# Train
python train.py --config configs/train_config.yaml

# Run tests
pytest tests/
```

## Model

A Transformer encoder trained to predict CT/T win probability from a sequence of
in-round events (kills, utility, economy). The first-token output is passed through
a 2-class linear head.
