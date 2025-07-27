import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Data_Labeler:
    def __init__(self, data, gesture_name, subject_name):
        """
        Initialize labeler with EMG data
        
        Args:
            data: numpy array of shape (samples, channels) 
            gesture_name: str like 'right', 'upward', 'downward', 'left'
            subject_name: str like 'john_retest', 'rina'
        """
        self.raw_data = data
        self.gesture_name = gesture_name
        self.subject_name = subject_name
        self.normalized_data = None
        self.labels = None
        self.normalization_params = {}
        
        print(f"Initialized labeler for {subject_name} - {gesture_name}")
        print(f"Data shape: {data.shape} ({data.shape[0]/1000:.1f} seconds)")
    
    def normalize_channel(self, channel_data, percentiles=(5, 95)):
        """
        Normalize a single channel using percentile-based clipping
        
        Args:
            channel_data: 1D numpy array
            percentiles: tuple of (low, high) percentiles for clipping
            
        Returns:
            normalized_data: 1D array normalized to [0, 1]
            params: tuple of (p_low, p_high) used for normalization
        """
        p_low, p_high = np.percentile(channel_data, percentiles)
        clipped = np.clip(channel_data, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low)
        
        return normalized, (p_low, p_high)
    
    def normalize_all(self, percentiles=(5, 95)):
        """Normalize all channels"""
        n_channels = self.raw_data.shape[1]
        self.normalized_data = np.zeros_like(self.raw_data)
        
        print("Normalizing all channels:")
        for ch in range(n_channels):
            channel_data = self.raw_data[:, ch]
            normalized, params = self.normalize_channel(channel_data, percentiles)
            
            self.normalized_data[:, ch] = normalized
            self.normalization_params[f'channel_{ch+1}'] = params
            
            print(f"  Channel {ch+1}: {params[0]:.2f} to {params[1]:.2f} -> [0, 1]")
    
    
    def plot_comparison(self, start_sec=0, duration_sec=10):
        """
        Plot raw vs normalized data side by side for each channel
        
        Args:
            start_sec: Start time in seconds
            duration_sec: Duration to plot in seconds
        """
        if self.normalized_data is None:
            self.normalize_all()
        
        start_idx = int(start_sec * 1000)
        end_idx = int((start_sec + duration_sec) * 1000)
        
        time_axis = np.arange(start_idx, end_idx) / 1000
        colors = ['steelblue', 'forestgreen', 'darkorange', 'crimson']
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle(f'{self.subject_name} - {self.gesture_name} (Raw vs Normalized)', fontsize=14)
        
        for ch in range(self.raw_data.shape[1]):
            # Raw data (left column)
            raw_data = self.raw_data[start_idx:end_idx, ch]
            axes[ch, 0].plot(time_axis, raw_data, color=colors[ch], linewidth=1.5)
            axes[ch, 0].set_title(f'Channel {ch+1} - Raw')
            axes[ch, 0].set_ylabel('Raw Amplitude')
            axes[ch, 0].grid(True, alpha=0.3)
            
            # Show normalization bounds
            p_low, p_high = self.normalization_params[f'channel_{ch+1}']
            axes[ch, 0].axhline(y=p_low, color='red', linestyle='--', alpha=0.7, label=f'{p_low:.1f}')
            axes[ch, 0].axhline(y=p_high, color='red', linestyle='--', alpha=0.7, label=f'{p_high:.1f}')
            axes[ch, 0].legend(fontsize=8)
            
            # Normalized data (right column)
            norm_data = self.normalized_data[start_idx:end_idx, ch]
            axes[ch, 1].plot(time_axis, norm_data, color=colors[ch], linewidth=1.5)
            axes[ch, 1].set_title(f'Channel {ch+1} - Normalized')
            axes[ch, 1].set_ylabel('Normalized [0-1]')
            axes[ch, 1].set_ylim(-0.1, 1.1)
            axes[ch, 1].grid(True, alpha=0.3)
            
            # Add range info
            range_val = p_high - p_low
            axes[ch, 1].text(0.02, 0.95, f'Range: {range_val:.1f}', 
                            transform=axes[ch, 1].transAxes, fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Add x-axis labels only to bottom plots
        axes[3, 0].set_xlabel('Time (seconds)')
        axes[3, 1].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.show()

    def analyze_channel_cleanliness(self, exclude_channels=None):
        """
        Analyze frequency characteristics to determine signal 'cleanliness'
        for each channel
        
        Args:
            exclude_channels: list of channel indices to exclude (0-based)
                                e.g. [3] to exclude channel 4
        """
        if self.normalized_data is None:
            self.normalize_all()
        
        if exclude_channels is None:
            exclude_channels = []
        
        channel_metrics = {}
        
        print(f"Frequency Analysis for {self.subject_name} - {self.gesture_name}")
        if exclude_channels:
            excluded_names = [f"Channel {ch+1}" for ch in exclude_channels]
            print(f"Excluding: {', '.join(excluded_names)}")
        print("="*60)
        
        for ch in range(self.normalized_data.shape[1]):
            if ch in exclude_channels:
                print(f"Channel {ch+1}: EXCLUDED")
                continue
                
            channel_data = self.normalized_data[:, ch]
            
            # FFT analysis
            fft_data = np.fft.fft(channel_data)
            power_spectrum = np.abs(fft_data)**2
            freqs = np.fft.fftfreq(len(channel_data), 1/1000)
            
            # Only use positive frequencies
            pos_freq_mask = freqs >= 0
            freqs_pos = freqs[pos_freq_mask]
            power_pos = power_spectrum[pos_freq_mask]
            
            # Calculate cleanliness metrics
            total_power = np.sum(power_pos)
            
            # Low frequency power (0-5 Hz) = "clean" sustained signals
            low_freq_mask = (freqs_pos >= 0) & (freqs_pos <= 5)
            low_freq_power = np.sum(power_pos[low_freq_mask])
            
            # High frequency power (5-50 Hz) = "messy" oscillatory signals  
            high_freq_mask = (freqs_pos > 5) & (freqs_pos <= 50)
            high_freq_power = np.sum(power_pos[high_freq_mask])
            
            # Cleanliness ratio (higher = cleaner)
            cleanliness_ratio = low_freq_power / (high_freq_power + 1e-8)
            
            # Signal-to-noise ratio approximation
            signal_power = np.sum(power_pos[freqs_pos <= 10])  # Relevant signal
            noise_power = np.sum(power_pos[freqs_pos > 50])    # High freq noise
            snr = signal_power / (noise_power + 1e-8)
            
            # Peak frequency
            peak_freq = freqs_pos[np.argmax(power_pos)]
            
            # Signal variance (another cleanliness measure)
            signal_variance = np.var(channel_data)
            
            channel_metrics[f'channel_{ch+1}'] = {
                'cleanliness_ratio': cleanliness_ratio,
                'snr': snr,
                'low_freq_power': low_freq_power,
                'high_freq_power': high_freq_power,
                'peak_frequency': peak_freq,
                'signal_variance': signal_variance
            }
            
            print(f"Channel {ch+1}:")
            print(f"  Cleanliness Ratio: {cleanliness_ratio:.3f}")
            print(f"  Signal-to-Noise:   {snr:.3f}")
            print(f"  Signal Variance:   {signal_variance:.3f}")
            print(f"  Peak Frequency:    {peak_freq:.2f} Hz")
            print()
        
        # Determine cleanest channel (only from non-excluded channels)
        if channel_metrics:
            cleanest_channel = max(channel_metrics.keys(), 
                                    key=lambda x: channel_metrics[x]['cleanliness_ratio'])
            
            print(f"CLEANEST CHANNEL (excluding {exclude_channels}): {cleanest_channel}")
            print(f"Cleanliness Ratio: {channel_metrics[cleanest_channel]['cleanliness_ratio']:.3f}")
        
        return channel_metrics