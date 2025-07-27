#%%
from stream_processor_bit import *
from processors import *
#%%

emg_data = 'data/new_raw_files/*'

import glob
import numpy as np
files = glob.glob(emg_data)

p_files = {}
files
#%%
for f in files:
    subject_name = f.split('_2025')[0].split('/')[-1]
    gesture_name = f.split('_')[-1].split('.txt')[0]
    name = f'{subject_name}_{gesture_name}'
    streamer = TXTStreamer(f)
    pipeline = EMGPipeline()
    pipeline.add_processor(DCRemover())
    pipeline.add_processor(ButterFilter([20, 450], sampling_rate=1000, filter_type='bandpass'))
    pipeline.add_processor(NotchFilter([60, 120], sampling_rate=1000))
    pipeline.add_processor(RectificationProcessor())
    pipeline.add_processor(EnvelopeProcessor(cutoff_freq=10, sampling_rate=1000))
    streamer.add_pipeline(pipeline)
    processed_data = streamer.process_all()
    processed_data = processed_data.T
    p_files[name] = processed_data
    # f = f.split('/')[-1].split('.txt')[0]
    f_name = f'data/processed_files/{name}_processed.csv'
    np.savetxt(f_name, processed_data, delimiter=',')



#%%
######################################################## LOOKING AT SIGNALS FIRST ######################################################################

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load your processed files
processed_files = {}
processed_dir = Path('data/processed_files')

for csv_file in processed_dir.glob('*_processed.csv'):
    name = csv_file.stem.replace('_processed', '')
    data = np.loadtxt(csv_file, delimiter=',')
    processed_files[name] = data
    print(f"Loaded {name}: shape {data.shape}, duration {data.shape[0]/1000:.2f}s")

print(f"\nTotal files: {len(processed_files)}")
#%%

# Pick one file to examine closely
file_name = list(processed_files.keys())[1]
data = processed_files[file_name]

print(f"\nExamining: {file_name}")
print(f"Shape: {data.shape}")
print(f"Data range: {data.min():.4f} to {data.max():.4f}")

# Plot it
plt.figure(figsize=(15, 8))
time_axis = np.arange(data.shape[0]) / 1000

if len(data.shape) > 1:
    for ch in range(data.shape[1]):
        plt.subplot(data.shape[1], 1, ch+1)
        plt.plot(time_axis, data[:, ch])
        plt.title(f'Channel {ch+1}')
        plt.ylabel('Amplitude')
        if ch == data.shape[1]-1:
            plt.xlabel('Time (seconds)')
else:
    plt.plot(time_axis, data)
    plt.title('Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set up seaborn styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Focus on forearm channels (1, 2, 3)
forearm_data = data[:, :3]
time_axis = np.arange(forearm_data.shape[0]) / 1000

# Let's start by looking at Channel 3 to identify gesture periods
channel_3 = forearm_data[:, 2]  # Channel 3 (index 2)

# First, let's plot Channel 3 cleanly to identify timing
plt.figure(figsize=(16, 6))
plt.plot(time_axis, channel_3, linewidth=1.5, color='steelblue')
plt.title('Channel 3 - Identify Gesture Periods', fontsize=14)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.show()

print("Look at the plot above and identify gesture start/end times.")
print("Then run the next component with your timing estimates...")
#%%
import matplotlib.pyplot as plt
import numpy as np

# Keep the detection function
def detect_gesture_periods_hybrid(signal, derivative_threshold=1.5, amplitude_threshold=1.0):
    derivative = np.abs(np.gradient(signal))
    deriv_mean = np.mean(derivative)
    deriv_std = np.std(derivative)
    deriv_thresh = deriv_mean + derivative_threshold * deriv_std
    rapid_changes = derivative > deriv_thresh
    
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    amp_thresh = signal_mean + amplitude_threshold * signal_std
    high_amplitude = signal > amp_thresh
    
    combined_mask = rapid_changes | high_amplitude
    
    return derivative, deriv_thresh, rapid_changes, amp_thresh, high_amplitude, combined_mask

# Test on Channel 3
channel_3 = forearm_data[:, 2]
derivative, deriv_thresh, rapid_mask, amp_thresh, amp_mask, combined_mask = detect_gesture_periods_hybrid(channel_3)

def plot_window(start_sec=0, duration_sec=10):
    """Plot a specific time window"""
    start_idx = int(start_sec * 1000)
    end_idx = int((start_sec + duration_sec) * 1000)
    
    window_time = time_axis[start_idx:end_idx]
    window_signal = channel_3[start_idx:end_idx]
    window_combined = combined_mask[start_idx:end_idx]
    window_deriv = derivative[start_idx:end_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Top: Signal with detection
    ax1.plot(window_time, window_signal, color='steelblue', linewidth=2, label='Channel 3')
    ax1.fill_between(window_time, 0, np.max(window_signal), 
                     where=window_combined, alpha=0.3, color='red', label='Detected Gestures')
    ax1.axhline(y=amp_thresh, color='green', linestyle='--', alpha=0.7, label=f'Amp Threshold: {amp_thresh:.2f}')
    ax1.set_title(f'Channel 3: {start_sec}s to {start_sec+duration_sec}s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Derivative
    ax2.plot(window_time, window_deriv, color='purple', linewidth=1, label='|Derivative|')
    ax2.axhline(y=deriv_thresh, color='orange', linestyle='--', label=f'Deriv Threshold: {deriv_thresh:.3f}')
    ax2.set_xlabel('Time (seconds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Start with first 10 seconds
plot_window(10, )

#%%

def overlay_gesture_timing_across_channels(start_sec=0, duration_sec=10):
    """Overlay Channel 3 gesture timing on all channels to check alignment"""
    start_idx = int(start_sec * 1000)
    end_idx = int((start_sec + duration_sec) * 1000)
    
    window_time = time_axis[start_idx:end_idx]
    window_combined = combined_mask[start_idx:end_idx]  # From Channel 3 detection
    
    # Include all 4 channels this time
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    channel_names = ['Channel 1 (Forearm)', 'Channel 2 (Forearm)', 'Channel 3 (Forearm)', 'Channel 4 (Thumb)']
    colors = ['steelblue', 'forestgreen', 'darkorange', 'crimson']
    
    for ch in range(4):
        window_signal = data[start_idx:end_idx, ch]
        
        # Plot the channel signal
        axes[ch].plot(window_time, window_signal, color=colors[ch], linewidth=2, 
                     label=channel_names[ch])
        
        # Overlay Channel 3's gesture timing as red shading
        axes[ch].fill_between(window_time, 0, np.max(window_signal), 
                             where=window_combined, alpha=0.3, color='red', 
                             label='Ch3 Gesture Timing')
        
        # Add some stats
        mean_val = np.mean(window_signal)
        max_val = np.max(window_signal)
        axes[ch].axhline(y=mean_val, color='gray', linestyle=':', alpha=0.7, 
                        label=f'Mean: {mean_val:.1f}')
        
        axes[ch].set_title(f'{channel_names[ch]} with Channel 3 Gesture Timing')
        axes[ch].legend(loc='upper right')
        axes[ch].grid(True, alpha=0.3)
        
        if ch == 3:  # Last subplot
            axes[ch].set_xlabel('Time (seconds)')
        
        axes[ch].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    # Print alignment statistics
    print(f"\nAlignment Analysis for {start_sec}s to {start_sec+duration_sec}s:")
    print("="*50)
    
    for ch in range(4):
        window_signal = data[start_idx:end_idx, ch]
        
        # Calculate how much of the gesture periods have elevated activity
        gesture_periods = window_signal[window_combined]
        non_gesture_periods = window_signal[~window_combined]
        
        if len(gesture_periods) > 0 and len(non_gesture_periods) > 0:
            gesture_mean = np.mean(gesture_periods)
            rest_mean = np.mean(non_gesture_periods)
            ratio = gesture_mean / rest_mean if rest_mean > 0 else float('inf')
            
            print(f"Channel {ch+1}:")
            print(f"  Gesture periods mean: {gesture_mean:.2f}")
            print(f"  Rest periods mean: {rest_mean:.2f}")
            print(f"  Gesture/Rest ratio: {ratio:.2f}")
            print(f"  {'✓ Good alignment' if ratio > 1.5 else '⚠ Weak alignment'}")
            print()

# Test the overlay
overlay_gesture_timing_across_channels(35, 10)

#%%
def detect_baseline_departures(signal, baseline_window=1000, threshold_multiplier=2.0, min_gesture_duration=0.3):
    """
    Detect gestures as departures from baseline activity
    """
    # Step 1: Estimate baseline using a rolling window approach
    # Use the minimum values in rolling windows as baseline estimate
    from scipy import ndimage
    
    # Calculate rolling minimum to estimate baseline
    baseline = ndimage.minimum_filter1d(signal, size=baseline_window)
    
    # Smooth the baseline estimate
    baseline_smooth = ndimage.uniform_filter1d(baseline, size=baseline_window//4)
    
    # Step 2: Calculate distance from baseline
    distance_from_baseline = signal - baseline_smooth
    
    # Step 3: Set threshold for "significant departure"
    baseline_noise = np.std(distance_from_baseline[distance_from_baseline < np.percentile(distance_from_baseline, 25)])
    departure_threshold = threshold_multiplier * baseline_noise
    
    # Step 4: Find periods above threshold
    above_baseline = distance_from_baseline > departure_threshold
    
    # Step 5: Find gesture start/end points
    transitions = np.diff(above_baseline.astype(int))
    gesture_starts = np.where(transitions == 1)[0] + 1  # Rising edges
    gesture_ends = np.where(transitions == -1)[0] + 1   # Falling edges
    
    # Handle edge cases
    if len(above_baseline) > 0 and above_baseline[0]:
        gesture_starts = np.concatenate([[0], gesture_starts])
    if len(above_baseline) > 0 and above_baseline[-1]:
        gesture_ends = np.concatenate([gesture_ends, [len(above_baseline)-1]])
    
    # Step 6: Filter by minimum duration
    min_samples = int(min_gesture_duration * 1000)
    valid_gestures = []
    
    for start, end in zip(gesture_starts, gesture_ends):
        if (end - start) >= min_samples:
            valid_gestures.append((start, end))
    
    return baseline_smooth, distance_from_baseline, departure_threshold, valid_gestures

# Test baseline approach
baseline, distance, threshold, gesture_periods = detect_baseline_departures(channel_3)

def plot_baseline_detection(start_sec=0, duration_sec=10):
    start_idx = int(start_sec * 1000)
    end_idx = int((start_sec + duration_sec) * 1000)
    
    window_time = time_axis[start_idx:end_idx]
    window_signal = channel_3[start_idx:end_idx]
    window_baseline = baseline[start_idx:end_idx]
    window_distance = distance[start_idx:end_idx]
    
    # Create gesture mask
    gesture_mask = np.zeros_like(window_signal, dtype=bool)
    for start, end in gesture_periods:
        if start_idx <= start < end_idx or start_idx <= end < end_idx:
            local_start = max(0, start - start_idx)
            local_end = min(len(window_signal), end - start_idx)
            if local_start < local_end:
                gesture_mask[local_start:local_end] = True
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Top: Original signal with baseline
    axes[0].plot(window_time, window_signal, color='steelblue', linewidth=2, label='Original Signal')
    axes[0].plot(window_time, window_baseline, color='gray', linewidth=2, linestyle='--', label='Estimated Baseline')
    axes[0].fill_between(window_time, 0, np.max(window_signal), where=gesture_mask, 
                        alpha=0.3, color='red', label='Detected Gestures')
    axes[0].set_title('Signal with Baseline Estimation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Middle: Distance from baseline
    axes[1].plot(window_time, window_distance, color='purple', linewidth=1.5, label='Distance from Baseline')
    axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Departure Threshold: {threshold:.2f}')
    axes[1].fill_between(window_time, 0, np.max(window_distance), where=gesture_mask, 
                        alpha=0.3, color='red')
    axes[1].set_title('Distance from Baseline Analysis')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Bottom: Final result
    axes[2].plot(window_time, window_signal, color='steelblue', linewidth=2)
    axes[2].fill_between(window_time, 0, np.max(window_signal), where=gesture_mask, 
                        alpha=0.4, color='red', label='Complete Gesture Periods')
    
    # Mark onset/offset points
    for start, end in gesture_periods:
        if start_idx <= start < end_idx:
            axes[2].axvline(x=start/1000, color='green', linestyle='-', linewidth=3, alpha=0.7, label='Onset' if start == gesture_periods[0][0] else "")
        if start_idx <= end < end_idx:
            axes[2].axvline(x=end/1000, color='red', linestyle='-', linewidth=3, alpha=0.7, label='Offset' if end == gesture_periods[0][1] else "")
    
    axes[2].set_title('Final Gesture Periods with Onset/Offset Markers')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_baseline_detection(0, 10)
print(f"Detected {len(gesture_periods)} gesture periods")
print(f"Average gesture duration: {np.mean([(end-start)/1000 for start, end in gesture_periods]):.2f} seconds")

#%%
# Simple visualization of what we detected
def plot_detected_gestures(start_sec=0, duration_sec=10):
    start_idx = int(start_sec * 1000)
    end_idx = int((start_sec + duration_sec) * 1000)
    
    window_time = time_axis[start_idx:end_idx]
    window_signal = channel_3[start_idx:end_idx]
    
    # Create gesture mask from our 33 detected periods
    gesture_mask = np.zeros_like(window_signal, dtype=bool)
    for start, end in gesture_periods:
        if start_idx <= start < end_idx or start_idx <= end < end_idx:
            local_start = max(0, start - start_idx)
            local_end = min(len(window_signal), end - start_idx)
            if local_start < local_end:
                gesture_mask[local_start:local_end] = True
    
    plt.figure(figsize=(15, 6))
    plt.plot(window_time, window_signal, color='steelblue', linewidth=2, label='Channel 3')
    plt.fill_between(window_time, 0, np.max(window_signal), where=gesture_mask, 
                     alpha=0.4, color='red', label=f'Detected "Right" Gestures')
    
    plt.title(f'Detected Right Gesture Periods: {start_sec}s to {start_sec+duration_sec}s')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Count gestures in this window
    gesture_count = 0
    for start, end in gesture_periods:
        if start_idx <= start < end_idx:
            gesture_count += 1
    print(f"Gestures in this window: {gesture_count}")

plot_detected_gestures(0, 10)

#%%

for i, v in p_files.items():
    print(i, v.shape)
#%%

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


#%%
johnR_labeler = Data_Labeler(p_files['john_retest_right'], gesture_name = 'right', subject_name='John') 

#%%
johnR_labeler.normalize_all(percentiles=[1,95])
johnR_labeler.plot_comparison(start_sec = 15, duration_sec=30)

#%%
rinaR_labeler = Data_Labeler(p_files['rina_right'], gesture_name = 'right', subject_name ='rina')
rinaR_labeler.normalize_all(percentiles = [1,95])
rinaR_labeler.plot_comparison(start_sec = 15, duration_sec=30)
#%%
johnL_labeler = Data_Labeler(p_files['john_retest_left'], gesture_name= 'left',subject_name = 'john')
johnL_labeler.normalize_all(percentiles = [1,95])
johnL_labeler.plot_comparison(start_sec=20,duration_sec=30)

#%%
johnU_labeler = Data_Labeler(p_files['john_retest_upward'], gesture_name= 'up',subject_name = 'john')
johnU_labeler.normalize_all(percentiles = [1,95])
johnU_labeler.plot_comparison(start_sec=20,duration_sec=30)

#%%
rinaU_labeler = Data_Labeler(p_files['rina_upward'], gesture_name = 'up', subject_name ='rina')
rinaU_labeler.normalize_all(percentiles = [1,95])
rinaU_labeler.plot_comparison(start_sec = 15, duration_sec=30)