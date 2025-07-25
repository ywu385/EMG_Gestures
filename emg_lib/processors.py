import numpy as np
import time
from typing import Generator, List, Tuple
from scipy import signal
import os
import csv
import time
from statistics import mode
from abc import ABC, abstractmethod

# Abstract base class for processors
class SignalProcessor(ABC):
    def initialize(self, data: np.ndarray) -> None:
        """Optional initialization step. Default does nothing."""
        pass

    @abstractmethod 
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Required processing implementation.
        Args:
            data: Numpy array of shape (channels, samples)
        Returns:
            Processed data
        """
        pass

class FiveChannels(SignalProcessor):
    def __init__(self, indices = [4,5,6,7,8]):
        self.channels = indices

    def process(self, data:np.ndarray):
        return data[self.channels]

class ZeroChannelRemover(SignalProcessor):
    def __init__(self, threshold=1):  # Add threshold parameter
        self.active_channels = None
        self.threshold = threshold
        self.name = 'Zero Channel'
        
    def initialize(self, data: np.ndarray):
        # Calculate the mean absolute value for each channel
        channel_activity = np.mean(np.abs(data), axis=1)
        # Mark channels as active only if they have significant activity
        self.active_channels = channel_activity > self.threshold
        print(f"Channel activity levels: {channel_activity}")
        print(f"Active channels: {np.where(self.active_channels)[0]}")
        
    def process(self, data: np.ndarray) -> np.ndarray:
        if self.active_channels is None:
            self.initialize(data)
        return data[self.active_channels]


class DCRemover(SignalProcessor):
    def __init__(self):
        self.name = 'DCRemover'
    def process(self, data: np.ndarray) -> np.ndarray:
        # Remove DC offset from each channel
        return data - np.mean(data, axis=1, keepdims=True)

class NotchFilter(SignalProcessor):
    def __init__(self, notch_freqs: List[float], sampling_rate: int, quality_factor: float = 30.0):
        """
        Create notch filters for removing specific frequencies
        Args:
            notch_freqs: List of frequencies to remove (e.g., [60, 120, 180])
            sampling_rate: Signal sampling frequency in Hz
            quality_factor: Quality factor for notch filter (higher = narrower notch)
        """
        self.notch_freqs = notch_freqs
        self.sampling_rate = sampling_rate
        self.quality_factor = quality_factor
        self.b_filters = []
        self.a_filters = []
        self.name = 'Notch Filter'
        
        # Create filter coefficients for each frequency
        for freq in notch_freqs:
            b, a = signal.iirnotch(freq, quality_factor, sampling_rate)
            self.b_filters.append(b)
            self.a_filters.append(a)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply all notch filters in sequence"""
        filtered = data.copy()
        for b, a in zip(self.b_filters, self.a_filters):
            filtered = signal.filtfilt(b, a, filtered, axis=1)
        return filtered
    
class AdaptiveMaxNormalizer(SignalProcessor):
    def __init__(self, decay_factor=0.999):
        self.max_values = None
        self.decay_factor = decay_factor  # For gradually reducing max over time
    
    def process(self, window):
        if self.max_values is None:
            self.max_values = np.max(np.abs(window), axis=0)
        else:
            # Decay previous max slightly to allow adaptation over time
            self.max_values = self.max_values * self.decay_factor
            # Update max values where new signals exceed them
            current_max = np.max(np.abs(window), axis=0)
            self.max_values = np.maximum(self.max_values, current_max)
        
        # Normalize by max values (add small epsilon to avoid division by zero)
        return window / (self.max_values + 1e-8)

class RealTimeButterFilter(SignalProcessor):
    def __init__(self, cutoff, sampling_rate, filter_type='bandpass', order=4):
        """
        Create a real-time Butterworth filter for EMG signal processing
        Args:
            cutoff: Cutoff frequency or frequencies (Hz)
                - For lowpass/highpass: single value, e.g., 200
                - For bandpass/bandstop: list/tuple of [low, high], e.g., [20, 200]
            sampling_rate: Signal sampling frequency in Hz
            filter_type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'
            order: Filter order (higher = steeper roll-off, but more ripple)
        """
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate
        self.filter_type = filter_type
        self.order = order
        self.name = 'Real-Time Butterworth Filter'
        
        # Normalize the cutoff frequency
        nyquist = 0.5 * sampling_rate
        if isinstance(cutoff, (list, tuple)):
            self.normalized_cutoff = [cf / nyquist for cf in cutoff]
        else:
            self.normalized_cutoff = cutoff / nyquist
            
        # Get filter coefficients
        self.b, self.a = signal.butter(
            order,
            self.normalized_cutoff,
            btype=filter_type,
            analog=False
        )
        
        # Initialize filter states (one per channel)
        self.zi = None
        
    def initialize(self, data: np.ndarray) -> None:
        """Initialize filter states based on channel count"""
        n_channels = data.shape[0]
        # Create initial filter states (zeroed)
        self.zi = [signal.lfilter_zi(self.b, self.a) * 0 for _ in range(n_channels)]
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply Butterworth filter to the signal in real-time"""
        # Initialize filter states if needed
        if self.zi is None or len(self.zi) != data.shape[0]:
            self.initialize(data)
            
        # Create output array
        filtered = np.zeros_like(data)
        
        # Process each channel separately, maintaining filter state
        for i in range(data.shape[0]):
            filtered[i], self.zi[i] = signal.lfilter(
                self.b, self.a, data[i], zi=self.zi[i]
            )
            
        return filtered

class ButterFilter(SignalProcessor):
    def __init__(self, cutoff, sampling_rate, filter_type='bandpass', order=4):
        """
        Create a Butterworth filter for EMG signal processing
        
        Args:
            cutoff: Cutoff frequency or frequencies (Hz)
                - For lowpass/highpass: single value, e.g., 200
                - For bandpass/bandstop: list/tuple of [low, high], e.g., [20, 200]
            sampling_rate: Signal sampling frequency in Hz
            filter_type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'
            order: Filter order (higher = steeper roll-off, but more ripple)
        """
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate
        self.filter_type = filter_type
        self.order = order
        self.name = 'Butterworth Filter'
        
        # Normalize the cutoff frequency
        nyquist = 0.5 * sampling_rate
        if isinstance(cutoff, (list, tuple)):
            self.normalized_cutoff = [cf / nyquist for cf in cutoff]
        else:
            self.normalized_cutoff = cutoff / nyquist
            
        # Get filter coefficients
        self.b, self.a = signal.butter(
            order, 
            self.normalized_cutoff, 
            btype=filter_type, 
            analog=False
        )
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply Butterworth filter to the signal"""
        filtered = signal.filtfilt(self.b, self.a, data, axis=1)
        return filtered
   
class RectificationProcessor(SignalProcessor):
    def __init__(self):
        self.name = 'Rectification'
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply full-wave rectification (absolute value) to the signal
        This converts all negative values to positive
        
        Args:
            data: Numpy array of shape (channels, samples)
        Returns:
            Rectified data (all positive values)
        """
        return np.abs(data)

class EnvelopeProcessor(SignalProcessor):
    def __init__(self, cutoff_freq=10, sampling_rate=1000, method='butter'):
        """
        Create envelope by smoothing rectified EMG signal
        
        Args:
            cutoff_freq: Low-pass filter cutoff frequency in Hz (typically 5-15 Hz for EMG envelope)
            sampling_rate: Signal sampling frequency in Hz
            method: 'butter' for Butterworth filter or 'moving_avg' for moving average
        """
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.method = method
        self.name = f'Envelope ({method})'
        
        if method == 'butter':
            # Create low-pass Butterworth filter for envelope
            nyquist = 0.5 * sampling_rate
            self.normalized_cutoff = cutoff_freq / nyquist
            self.b, self.a = signal.butter(2, self.normalized_cutoff, btype='low', analog=False)
            self.zi = None  # Filter states for real-time processing
        elif method == 'moving_avg':
            # Calculate window size for moving average (should be odd)
            self.window_size = int(sampling_rate / cutoff_freq)
            if self.window_size % 2 == 0:
                self.window_size += 1  # Make it odd
    
    def initialize(self, data: np.ndarray) -> None:
        """Initialize filter states for Butterworth method"""
        if self.method == 'butter':
            n_channels = data.shape[0]
            self.zi = [signal.lfilter_zi(self.b, self.a) * 0 for _ in range(n_channels)]
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply envelope detection (smoothing) to rectified signal
        
        Args:
            data: Rectified numpy array of shape (channels, samples)
        Returns:
            Smoothed envelope signal
        """
        if self.method == 'butter':
            # Initialize filter states if needed
            if self.zi is None or len(self.zi) != data.shape[0]:
                self.initialize(data)
            
            # Create output array
            envelope = np.zeros_like(data)
            
            # Process each channel separately with Butterworth filter
            for i in range(data.shape[0]):
                envelope[i], self.zi[i] = signal.lfilter(
                    self.b, self.a, data[i], zi=self.zi[i]
                )
            
            return envelope
            
        elif self.method == 'moving_avg':
            # Apply moving average smoothing
            envelope = np.zeros_like(data)
            
            for i in range(data.shape[0]):
                # Use numpy convolve for moving average
                kernel = np.ones(self.window_size) / self.window_size
                # Use 'same' mode to maintain original signal length
                envelope[i] = np.convolve(data[i], kernel, mode='same')
            
            return envelope


class MovingAverageProcessor(SignalProcessor):
    def __init__(self, window_size=50):
        """
        Alternative smoothing processor using moving average
        
        Args:
            window_size: Number of samples to average (higher = more smoothing)
        """
        self.window_size = window_size
        self.name = f'Moving Average (window={window_size})'
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing"""
        smoothed = np.zeros_like(data)
        kernel = np.ones(self.window_size) / self.window_size
        
        for i in range(data.shape[0]):
            smoothed[i] = np.convolve(data[i], kernel, mode='same')
            
        return smoothed

######################################################## Normalizing features  ######################################################################

class MaxNormalizer(SignalProcessor):
    """
    SignalProcessor that normalizes EMG data by dividing by the maximum value
    for each channel, continuously updating the maximum if larger values are encountered.
    """
    def __init__(self, initial_max_values=None, epsilon=1e-8):
        """
        Initialize the max normalizer
        
        Args:
            initial_max_values: Optional array of initial maximum values for each channel
            epsilon: Small value to avoid division by zero
        """
        self.max_values = initial_max_values  # Will be initialized if None
        self.epsilon = epsilon
        self.name = 'MaxNormalizer'
        self.initialized = False
    
    def initialize(self, data: np.ndarray) -> None:
        """Initialize max values from initial data"""
        n_channels = data.shape[0]
        
        # If max values weren't provided, initialize from data
        if self.max_values is None:
            self.max_values = np.max(np.abs(data), axis=1)
        # If max values were provided but not the right shape, reshape
        elif isinstance(self.max_values, (int, float)):
            self.max_values = np.ones(n_channels) * self.max_values
        # Ensure max_values is the right shape
        elif len(self.max_values) != n_channels:
            raise ValueError(f"Expected {n_channels} max values, got {len(self.max_values)}")
        
        self.initialized = True
    
    def set_max_values(self, max_values):
        """Manually set the maximum values for each channel"""
        self.max_values = np.array(max_values)
        self.initialized = True
    
    def update_max_values(self, data: np.ndarray) -> None:
        """Update max values if larger values are found"""
        current_max = np.max(np.abs(data), axis=1)
        self.max_values = np.maximum(self.max_values, current_max)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data by dividing by the maximum value for each channel
        
        Args:
            data: EMG data of shape (channels, samples)
            
        Returns:
            Normalized data with values between -1 and 1
        """
        # Initialize if not already
        if not self.initialized:
            self.initialize(data)
        
        # Update max values if larger values are found
        self.update_max_values(data)
        
        # Normalize by dividing by max values (with epsilon to avoid division by zero)
        normalized_data = data / (self.max_values[:, np.newaxis] + self.epsilon)
        
        return normalized_data
    
class AdaptiveMaxNormalizer(SignalProcessor):
    """
    Adaptive MaxNormalizer that can handle changes in channel count and detect user fatigue.
    """
    def __init__(self, 
                 initial_max_values=None, 
                 epsilon=1e-8,
                 fatigue_threshold=0.3,    # How much drop to consider fatigue
                 fatigue_history=50,       # How many chunks to look back for fatigue
                 adapt_rate=0.95):         # How quickly to adjust max values
        """
        Initialize the adaptive max normalizer
        
        Args:
            initial_max_values: Optional array of initial maximum values for each channel
            epsilon: Small value to avoid division by zero
            fatigue_threshold: Fraction decrease to trigger fatigue adaptation
            fatigue_history: Number of chunks to maintain for RMS history
            adapt_rate: How quickly to adjust max values when fatigue detected
        """
        self.max_values = initial_max_values
        self.epsilon = epsilon
        self.name = 'AdaptiveMaxNormalizer'
        self.initialized = False
        
        # For channel count adaptation
        self.data_shape = None
        
        # For fatigue detection
        self.fatigue_threshold = fatigue_threshold
        self.fatigue_history = fatigue_history
        self.adapt_rate = adapt_rate
        self.rms_history = []
        self.baseline_max = None
        self.fatigue_detected = False
        self.chunk_counter = 0
    
    def initialize(self, data: np.ndarray) -> None:
        """Initialize for the current data shape"""
        n_channels = data.shape[0]
        self.data_shape = data.shape
        
        # If max values weren't provided, initialize from data
        if self.max_values is None:
            self.max_values = np.max(np.abs(data), axis=1)
        # If max values were provided but not the right shape, reshape
        elif isinstance(self.max_values, (int, float)):
            self.max_values = np.ones(n_channels) * self.max_values
        # If shape changed, reinitialize
        elif len(self.max_values) != n_channels:
            print(f"Channel count changed from {len(self.max_values)} to {n_channels}, reinitializing")
            self.max_values = np.max(np.abs(data), axis=1)
            
        # Store baseline max for reference
        self.baseline_max = self.max_values.copy()
        self.rms_history = []  # Reset RMS history
        self.fatigue_detected = False
        self.initialized = True
    
    def _check_for_fatigue(self, data: np.ndarray) -> None:
        """Check for signs of user fatigue"""
        # Calculate RMS for current chunk
        rms_values = np.sqrt(np.mean(data**2, axis=1))
        avg_rms = np.mean(rms_values)
        
        # Add to history
        self.rms_history.append(avg_rms)
        if len(self.rms_history) > self.fatigue_history:
            self.rms_history.pop(0)
            
        # Only check for fatigue if we have enough history
        if len(self.rms_history) >= 20:
            # Compare early activity to recent activity
            early_rms = np.mean(self.rms_history[:10])
            recent_rms = np.mean(self.rms_history[-10:])
            
            if early_rms > 0 and recent_rms / early_rms < (1 - self.fatigue_threshold):
                if not self.fatigue_detected:
                    print(f"Fatigue detected! Signal strength ratio: {recent_rms/early_rms:.2f}")
                    self.fatigue_detected = True
            # If signal increases again, reset fatigue detection
            elif recent_rms / early_rms > 0.9:
                if self.fatigue_detected:
                    print("Signal strength recovered, resetting fatigue detection")
                    self.fatigue_detected = False
    
    def update_max_values(self, data: np.ndarray) -> None:
        """Update max values with fatigue compensation"""
        # Calculate current max
        current_max = np.max(np.abs(data), axis=1)
        
        # Periodically check for fatigue
        self.chunk_counter += 1
        if self.chunk_counter % 10 == 0:
            self._check_for_fatigue(data)
        
        # Update max values
        if self.fatigue_detected:
            # When fatigued, gradually reduce max values to adapt
            adaptive_ceiling = np.maximum(current_max, self.baseline_max * self.adapt_rate)
            self.max_values = np.minimum(self.max_values, adaptive_ceiling)
        else:
            # Standard behavior - always keep highest values seen
            self.max_values = np.maximum(self.max_values, current_max)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data by dividing by the maximum value for each channel,
        with adaptation for channel count changes and user fatigue
        
        Args:
            data: EMG data of shape (channels, samples)
            
        Returns:
            Normalized data with values between -1 and 1
        """
        # If not initialized or data shape changed, reinitialize
        if not self.initialized or data.shape[0] != (self.data_shape[0] if self.data_shape else None):
            self.initialize(data)
        
        # Update max values with fatigue compensation
        self.update_max_values(data)
        
        # Normalize by dividing by max values
        normalized_data = data / (self.max_values[:, np.newaxis] + self.epsilon)
        
        return normalized_data

######################################################## OLD IMPLEMENTATIONS ######################################################################

######################################################## OLD IMPLEMENTATION ######################################################################
# Specific processor for removing zero channels
# class ZeroChannelRemover(SignalProcessor):
#     def __init__(self):
#         self.active_channels = None

#     def initialize(self, data: np.ndarray): # initializes channels that are non-zero.  This allows for less flucation when streaming
#         self.active_channels = np.any(data != 0, axis=1)
        
#     def process(self, data: np.ndarray) -> np.ndarray:
#         if self.active_channels is None:
#             self.initialize(data)
#         return data[self.active_channels]
######################################################## OLD IMPLEMENTATION ######################################################################