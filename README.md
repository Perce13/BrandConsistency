# Brand Consistency Checker

This application analyzes images for their consistency with corporate brand imagery guidelines by examining:
- Brightness
- Saturation
- Contrast
- Sharpness (Blur)

## Features
- Multiple image upload
- Real-time analysis
- Visual comparison with reference values
- Downloadable detailed report
- Summary statistics

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
streamlit run brandcheck.py
```

## Project Structure
```
├── brandcheck.py        # Main application code
├── requirements.txt     # Python dependencies
├── packages.txt        # System dependencies
└── README.md          # This file
```

## Requirements
- Python 3.9+
- OpenCV
- Streamlit
- Other dependencies listed in requirements.txt
