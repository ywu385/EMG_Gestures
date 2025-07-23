"""
EMG Gesture Recognition Library

A comprehensive library for EMG signal processing, real-time streaming, 
and gesture classification using machine learning techniques.  This is meant to work with
BITalino Revolution.

Main Components:
- Signal Processing: Filtering, rectification, envelope detection
- Real-time Streaming: BITalino device integration and data pipelines  
- Feature Extraction: Wavelet transforms and statistical features
- Machine Learning: Classification models for gesture recognition
"""

# Core signal processing components
from .processors import (
    SignalProcessor,        # Base signal processing class
    FiveChannels,          # 5-channel EMG processor
    ZeroChannelRemover,    # Remove inactive channels
    DCRemover,             # DC offset removal
    NotchFilter,           # 60Hz noise filtering
    RectificationProcessor, # Signal rectification
    EnvelopeProcessor,     # Envelope extraction
    RealTimeButterFilter   # Real-time Butterworth filtering
)

# Streaming and data collection
from .stream_processor_bit import (
    BitaStreamer,          # BITalino device streaming
    EMGPipeline,          # Signal processing pipeline
    TXTStreamer           # File-based streaming
)

# Post-processing and machine learning
from .post_processing import (
    SignalBuffer,          # Real-time signal buffering
    ModelProcessor,        # Base ML model interface
    WideModelProcessor,    # Wide neural network processor
    WaveletFeatureExtractor, # DWT feature extraction
    WaveletProcessor,      # Wavelet transform processing
    LGBMProcessor,         # LightGBM classifier
    IntensityProcessor,    # Signal intensity analysis
    FeatureUtils,          # Feature engineering utilities
    BaggedRF              # Random Forest ensemble
)

__version__ = "0.1.0"
__author__ = "John Wu"
__email__ = "jywu86@icloud.com"

# For convenience - commonly used classes
# __all__ = [
#     "BitaStreamer", "EMGPipeline", "SignalProcessor", 
#     "WaveletFeatureExtractor", "LGBMProcessor", "TXTStreamer", "WaveletProcessor"
# ]