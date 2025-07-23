import multiprocessing
import numpy as np
import time
import os
import signal
import atexit
from datetime import datetime
from stream_processor_bit import *
from processors import *
from post_processing import *

# Try importing the EMG modules at global scope
try:
    from revolution_api.bitalino import *
    EMG_MODULES_AVAILABLE = True
    print("All EMG modules loaded successfully")
except ImportError as e:
    print(f"Error importing EMG modules: {e}")
    EMG_MODULES_AVAILABLE = False

# Global variables for configuration
USE_BITALINO = True
INPUT_FILE = './data/combined.txt'
mac_address = "/dev/tty.BITalino-3C-C2"

# Global queue for data transfer
emg_queue = multiprocessing.Queue()

# List to keep track of recorded files
recorded_files = []

def convert_npy_to_txt(input_file, delete_npy=True):
    """Convert a .npy file to a .txt file and optionally delete the NPY file"""
    output_file = os.path.splitext(input_file)[0] + '.txt'
    
    try:
        data = np.load(input_file, allow_pickle=True)
        
        if data.ndim == 1:
            np.savetxt(output_file, data, fmt='%.6f')
            print(f"Converted 1D array with {data.shape[0]} samples")
        elif data.ndim == 2:
            if data.shape[0] > data.shape[1]:
                np.savetxt(output_file, data, fmt='%.6f', delimiter='\t')
                print(f"Converted 2D array with shape {data.shape} (rows as time points)")
            else:
                np.savetxt(output_file, data.T, fmt='%.6f', delimiter='\t')
                print(f"Converted 2D array with {data.shape[0]} channels, {data.shape[1]} samples each")
        else:
            print(f"Warning: Array has {data.ndim} dimensions with shape {data.shape}")
            reshaped = data.reshape(-1, data.shape[-1])
            np.savetxt(output_file, reshaped, fmt='%.6f', delimiter='\t')
        
        print(f"Saved to: {output_file}")
        
        if delete_npy:
            try:
                os.remove(input_file)
                print(f"Deleted NPY file: {input_file}")
            except Exception as e:
                print(f"Error deleting NPY file: {e}")
                
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
    
    return output_file

def convert_all_files():
    """Convert all recorded NPY files to TXT - called on any exit"""
    if recorded_files:
        print("\n" + "="*50)
        print("CONVERTING NPY FILES TO TXT FORMAT")
        print("="*50)
        
        for i, npy_file in enumerate(recorded_files):
            if os.path.exists(npy_file):  # Check if file still exists
                print(f"[{i+1}/{len(recorded_files)}] Converting {os.path.basename(npy_file)}")
                try:
                    convert_npy_to_txt(npy_file, delete_npy=True)
                except Exception as e:
                    print(f"Error converting {npy_file}: {e}")
        
        print("\nFile conversion complete!")

# Register cleanup function to run on any exit
atexit.register(convert_all_files)

def stream_emg_data(chunk_queue, use_bitalino=True, input_file=None):
    """Stream EMG data from either BiTalino or a text file"""
    try:
        if use_bitalino and EMG_MODULES_AVAILABLE:
            print('Creating BiTalino streamer within process')
            try:
                device = BITalino(mac_address)
                device.battery(10)
                print("BiTalino device connected")
                
                streamer = BitaStreamer(device)
                print("BiTalino streamer created")
            except Exception as e:
                print(f"Error initializing BiTalino: {e}")
                print("Falling back to text streamer")
                streamer = TXTStreamer(input_file, simple=False)
        else:
            print('Creating text streamer within process')
            streamer = TXTStreamer(input_file, simple=False)
        
        pipeline = EMGPipeline()
        pipeline.add_processor(ZeroChannelRemover())
        streamer.add_pipeline(pipeline)
        
        print("Starting data streaming process")
        for chunk in streamer.stream_processed():
            try:
                chunk_queue.put((chunk), block=False)
            except Exception as e:
                print(f"Error putting chunk in queue: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Stream processing error: {e}")
    finally:
        print("Stream processing stopped.")
        if use_bitalino and EMG_MODULES_AVAILABLE and 'device' in locals():
            try:
                device.close()
                print("BiTalino device closed")
            except:
                print("Error closing BiTalino device")

def record_gesture(queue, duration, output_file):
    """Record a single gesture"""
    print(f"Recording gesture for {duration} seconds to {output_file}")
    
    all_data = []
    start_time = time.time()
    end_time = start_time + duration
    
    # Clear any backlog in the queue
    while not queue.empty():
        queue.get(block=False)
    
    print("Queue cleared, starting to record fresh data...")
    
    try:
        while time.time() < end_time:
            try:
                if not queue.empty():
                    chunk = queue.get(block=False)
                    all_data.append(chunk)
                    
                    elapsed = time.time() - start_time
                    print(f"\rRecording: {elapsed:.1f}/{duration} seconds - {int(elapsed/duration*100)}% complete", end="")
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"\nError during recording: {e}")
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    
    print("\nRecording complete. Processing data...")
    
    try:
        if len(all_data) > 0:
            if isinstance(all_data[0], np.ndarray):
                if all_data[0].ndim == 2:
                    num_channels = all_data[0].shape[0]
                    concat_data = np.hstack(all_data)
                    np.save(output_file, concat_data)
                    print(f"Saved {concat_data.shape[1]} samples for {num_channels} channels to {output_file}")
                    print(f"Data shape: {concat_data.shape}")
                else:
                    concat_data = np.concatenate(all_data)
                    np.save(output_file, concat_data)
                    print(f"Saved {len(concat_data)} samples to {output_file}")
            else:
                data_array = np.array(all_data)
                np.save(output_file, data_array)
                print(f"Saved data with shape {data_array.shape} to {output_file}")
            
            recorded_files.append(output_file)
            
        else:
            print("No data collected during recording!")
    except Exception as e:
        print(f"Error saving data: {e}")

def choose_collection_mode():
    """Let user choose between full collection or starting from specific gesture"""
    print("\nCollection Mode Options:")
    print("1. Full collection (upward → downward → left → right)")
    print("2. Start from specific gesture")
    print("3. Record custom gestures only")
    
    while True:
        try:
            choice = input("Choose mode (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            raise

def get_starting_gesture():
    """Get which gesture to start from"""
    gestures = ["upward", "downward", "left", "right"]
    
    print("\nAvailable gestures:")
    for i, gesture in enumerate(gestures):
        print(f"{i+1}. {gesture}")
    
    while True:
        try:
            choice = input("Start from which gesture (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice) - 1, gestures
            else:
                print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            raise

def main():
    base_output_dir = "emg_recordings"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create and start the streaming process
    streamer_process = multiprocessing.Process(
        target=stream_emg_data, 
        args=(emg_queue, USE_BITALINO, INPUT_FILE),
        daemon=True
    )
    streamer_process.start()
    
    print("\n" + "="*50)
    print("Starting EMG recording system")
    print("="*50 + "\n")
    
    time.sleep(2)
    
    try:
        # Get collection mode
        mode = choose_collection_mode()
        
        # Get recording parameters
        duration = float(input("Enter recording duration in seconds for each gesture: "))
        if duration <= 0:
            raise ValueError("Duration must be positive")
        name = str(input("Enter Name of participant: ")).strip()
        
        # Create user-specific directory
        # Clean the name for file system compatibility
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')  # Replace spaces with underscores
        if not safe_name:
            safe_name = "unknown_participant"
        
        user_output_dir = os.path.join(base_output_dir, safe_name)
        os.makedirs(user_output_dir, exist_ok=True)
        print(f"Recording to: {user_output_dir}")
        
        if mode == 1:
            # Full collection
            gestures = ["upward", "downward", "left", "right"]
            start_idx = 0
        elif mode == 2:
            # Start from specific gesture
            start_idx, gestures = get_starting_gesture()
        else:
            # Custom gestures only
            gestures = []
            start_idx = 0
        
        # Record predefined gestures (if any)
        if mode in [1, 2]:
            for i in range(start_idx, len(gestures)):
                gesture = gestures[i]
                print(f"\n--- Preparing to record {gesture} ({name}) ---")
                
                input(f"Press Enter when ready to record {gesture}...")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{user_output_dir}/{safe_name}_{timestamp}_{gesture}.npy"
                
                record_gesture(emg_queue, duration, filename)
                print(f"Finished recording {gesture}")
                
                # Ask to continue (except for last gesture)
                if i < len(gestures) - 1:
                    cont = input(f"Continue to {gestures[i+1]}? (y/n): ").strip().lower()
                    if cont != 'y':
                        print("Gesture recording session ended early by user")
                        break
        
        # Always offer custom gestures
        while True:
            another = input("\nRecord additional custom gestures? (y/n): ").strip().lower()
            if another == 'y':
                try:
                    gesture_name = input("Enter gesture name: ").strip()
                    if not gesture_name:
                        gesture_name = "Custom_Gesture"
                    
                    custom_duration = input(f"Duration for {gesture_name} (Enter for {duration}s): ").strip()
                    if custom_duration:
                        custom_duration = float(custom_duration)
                    else:
                        custom_duration = duration
                    
                    if custom_duration <= 0:
                        raise ValueError("Duration must be positive")
                    
                    input(f"Press Enter when ready to record {gesture_name}...")
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{user_output_dir}/{safe_name}_{gesture_name}_{timestamp}.npy"
                    
                    record_gesture(emg_queue, custom_duration, filename)
                    
                except ValueError as e:
                    print(f"Invalid input: {e}")
            elif another == 'n':
                break
            else:
                print("Please enter 'y' or 'n'")
        
        print("\nRecording session completed!")
        
    except (KeyboardInterrupt, ValueError) as e:
        if isinstance(e, ValueError):
            print(f"Invalid input: {e}")
        else:
            print("\nSession interrupted by user")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        print("\nRecording session ended")
        
        # Clean up
        if streamer_process.is_alive():
            print("Terminating streamer process...")
            streamer_process.terminate()
            streamer_process.join(timeout=1.0)
            print("Streamer process terminated")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()