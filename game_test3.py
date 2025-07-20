# Add these imports at the top if not already present
import time
import csv
from datetime import datetime
from collections import deque

#%%
import pygame
import multiprocessing
import time
import traceback
import numpy as np
import pickle
import glob
import atexit
import sys
import random
from stream_processor_bit import *
from processors import *
from post_processing import *

# Import game classes
from target_game import TargetGame
from gamemanager2 import GameManager
from spriralgame import GridSpiralChallenge

# Add this global variable near your other global variables
latency_log = []
# Moving average window for latency components (last N measurements)
latency_window_size = 10
t_acq_values = deque(maxlen=latency_window_size)
t_feat_values = deque(maxlen=latency_window_size)
t_pred_values = deque(maxlen=latency_window_size)
t_ui_values = deque(maxlen=latency_window_size)
total_latency_values = deque(maxlen=latency_window_size)


import argparse

def parse_arguments():
    """Parse command line arguments for model selection and intensity scaling"""
    parser = argparse.ArgumentParser(description='EMG Processing with model selection')
    
    # Add model selection group
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model1', action='store_true', 
                        help='Use the first model (LGBM.pkl)')
    model_group.add_argument('--model2', action='store_true', 
                        help='Use the second model (lgb.pkl)')
    model_group.add_argument('--model3', action='store_true', 
                        help='Use the third model (lgb.pkl)')
    
    # Add intensity scaling
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scaling factor for movement intensity (default: 1.0)')
    
    # Add optional max intensity cap
    parser.add_argument('--max-intensity', type=float, default=3.0,
                        help='Maximum intensity value (default: 3.0)')
    
    return parser.parse_args()

# grabbing arguments
args = parse_arguments()

########################################################  Adding speed scaling ######################################################################

manual_intensity_scale = args.scale
max_allowed_intensity = args.max_intensity
print(f"Using intensity scaling factor: {manual_intensity_scale}")
print(f"Maximum intensity capped at: {max_allowed_intensity}")
#%%
# BITA = True
# Flag for model switching
if args.model1: 
    model_path = './working_models/LGBM_simple.pkl'
    print('Base Model loaded as {model_path}')
elif args.model2:
    model_path = './working_models/LGBM.pkl'
    print('Experimental Model Loaded {model_path}') 
elif args.model3:
    model_path ='./working_models/LGBM_model3.pkl'
    print('Experimental (zona) Model Loaded {model_path}') 
else:
    model_path = './working_models/LGBM_model3.pkl'
    print(f'No model selected.  Defaulting to {model_path}') 

# Import your custom EMG modules
try:    
    from revolution_api.bitalino import *
    EMG_MODULES_AVAILABLE = True
    print("All EMG modules loaded successfully")
except ImportError as e:
    print(f"Error importing EMG modules: {e}")
    EMG_MODULES_AVAILABLE = False

# Define queue at the top BEFORE it's used elsewhere
# Small queue for real-time communication - only keeps most recent predictions
emg_queue = multiprocessing.Queue(maxsize=5)

# Global variables and initialization
print("Initializing EMG components at global level...")
#%%
# Find the model path
# model_paths = glob.glob('./working_models/LGBM.pkl')
# model_paths = glob.glob('./working_models/lgb.pkl')
if model_path:
    # model_path = model_path[0]
    # print(f"Found model: {model_path}")
    
    # Load model
    with open(model_path, 'rb') as file:
        # model, label_encoder = pickle.load(file)
        models = pickle.load(file)
    print("Model loaded at global level")
else:
    print("No model files found")
    model = None
#%%
# BITalino MAC address
mac_address = "/dev/tty.BITalino-3C-C2"  # Update with your device's address


# Initialize device and streamer
if EMG_MODULES_AVAILABLE:
    try:
        # if BITA:
        # Setup device
        # device = BITalino(mac_address)
        # device.battery(10)
        # print("BITalino connected at global level")
        # streamer = BitaStreamer(device)

        # Setup streamer
        
        print("Created BITalino streamer at global level")
        # else:
        import glob
        files = glob.glob('./data/zona*')
        streamer = TXTStreamer(filepath = files[0])
        
        # Setup pipeline
        # pipeline = EMGPipeline()
        # pipeline.add_processor(ZeroChannelRemover())
        # pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
        # pipeline.add_processor(DCRemover())
        # # bandpass = ButterFilter(cutoff=[20, 450], sampling_rate=1000, filter_type='bandpass', order=4)
        # # pipeline.add_processor(bandpass)
        # pipeline.add_processor(AdaptiveMaxNormalizer())
        # streamer.add_pipeline(pipeline)
        pipeline = EMGPipeline()
        pipeline.add_processor(ZeroChannelRemover())
        pipeline.add_processor(NotchFilter([60], sampling_rate=1000)) 
        pipeline.add_processor(DCRemover())
        emg_bandpass = RealTimeButterFilter(
                            cutoff=[20, 450],  # Target the 20-450 Hz frequency range for EMG
                            sampling_rate=1000,  # Assuming 1000 Hz sampling rate
                            filter_type='bandpass',
                            order=4  # 4th order provides good balance between sharpness and stability
                        )
        pipeline.add_processor(emg_bandpass)
        # pipeline.add_processor(AdaptiveMaxNormalizer())
        # pipeline.add_processor(MaxNormalizer())
        streamer.add_pipeline(pipeline)
        print("Pipeline added to streamer at global level")
        
        # Setup model processor
        if args.model2 or args.model3:
            model_processor = LGBMProcessor(
                models=models,
                window_size=250,
                overlap=0.5,
                sampling_rate=1000,
                n_predictions=5,
                wavelets  = ['sym5']
                # label_encoder=label_encoder
            )
        elif args.model1:
            model_processor = LGBMProcessor(
                models=models,
                window_size=250,
                overlap=0.5,
                sampling_rate=1000,
                n_predictions=5,
                # wavelets  = ['sym5']
                # label_encoder=label_encoder
            )
        # model_processor = ModelProcessor(
        #     model = model,
        #     window_size = 250,
        #     overlap=0.5,

        # )

        
        # Setup buffer and intensity processor
        buffer = SignalBuffer(window_size=250, overlap=0.5)
        intensity_processor = IntensityProcessor(scaling_factor=1.5)
        
        print("All EMG components initialized at global level")
        
        # Flag to indicate if EMG is initialized
        emg_initialized = True
    except Exception as e:
        print(f"Error initializing EMG components: {e}")
        traceback.print_exc()
        emg_initialized = False
else:
    emg_initialized = False


# Function to log latency metrics to CSV file
def save_latency_log():
    if latency_log:
        filename = f"latency_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'T_acq', 'T_feat', 'T_pred', 'T_ui', 'Total_Latency'])
            for entry in latency_log:
                writer.writerow([
                    entry['timestamp'],
                    entry['t_acq'],
                    entry['t_feat'],
                    entry['t_pred'], 
                    entry['t_ui'],
                    entry['total']
                ])
        print(f"Latency log saved to {filename}")

# Register this function to run at exit
atexit.register(save_latency_log)

# Modify your process_emg_data function to include all timing components
def process_emg_data(model_processor, chunk_queue):
    counter = 0
    print("Starting to process EMG data...")
    
    # Add rate limiting variables
    current_intensity = 0
    ramp_factor = 0.1
    max_intensity_limit = 3.5

    while True:  # Outer loop for reconnection
        try:
            print("Connecting to stream...")
            for chunk in streamer.stream_processed():
                # Record acquisition timestamp - this is as close as we can get to 
                # when the signal was actually acquired
                t_acq_start = time.time()
                
                # Start feature extraction timing
                t_feat_start = time.time()
                windows = buffer.add_chunk(chunk)
                intensity_value = None
                prediction = None
                
                for w in windows:
                    # Start prediction timing
                    t_pred_start = time.time()
                    prediction = model_processor.process(w)
                    t_pred_end = time.time()
                    t_pred_duration = t_pred_end - t_pred_start
                    
                    # Continue with intensity processing
                    i_metrics = intensity_processor.process(w)
                    
                    metric_att = 'rms_values'
                    if i_metrics[metric_att] is not None and len(i_metrics[metric_att]) > 0:
                        min_speed, max_speed = 0, 1
                        norm_rms = i_metrics['overall_normalized_rms']
                        intensity_value = min_speed + (norm_rms * (max_speed - min_speed))
                        intensity_value = intensity_value * manual_intensity_scale
                        intensity_value = min(intensity_value, max_allowed_intensity)
                    
                    # End feature extraction timing (includes everything up to prediction)
                    t_feat_end = time.time()
                    t_feat_duration = t_feat_end - t_feat_start - t_pred_duration  # Exclude prediction time
                    
                    # Calculate acquisition time (time from physical signal to data arrival)
                    # This is an approximation, as we can't know exactly when the muscle moved
                    t_acq_duration = t_acq_start - (t_acq_start - 0.01)  # Assuming ~10ms acquisition delay
                    
                    # Only when model buffer has enough data
                    if prediction is not None:
                        # Handle queue
                        if chunk_queue.full():
                            try:
                                chunk_queue.get_nowait()
                            except:
                                pass
                        
                        # Create timestamp for UI to calculate its latency
                        prediction_timestamp = time.time()
                        
                        # Add newest prediction with timestamp for UI latency calculation
                        # chunk_queue.put((prediction, intensity_value, prediction_timestamp), block=False)
                        
                        chunk_queue.put((
                                    prediction, 
                                    intensity_value, 
                                    t_acq_duration,  # Pass all timing values
                                    t_feat_duration,
                                    t_pred_duration,
                                    prediction_timestamp
                                ), block=False)
                        # Add to moving average
                        # t_acq_values.append(t_acq_duration)
                        # t_feat_values.append(t_feat_duration)
                        # t_pred_values.append(t_pred_duration)
                        
                        # Log component latencies (UI latency will be added in game loop)
                        print(f"Prediction {counter}: {prediction}, intensity={intensity_value:.2f}")
                        print(f"  T_acq: {t_acq_duration*1000:.2f}ms, T_feat: {t_feat_duration*1000:.2f}ms, T_pred: {t_pred_duration*1000:.2f}ms")
                        
                        counter += 1
                    
        except Exception as e:
            print(f"Error processing EMG data: {e}")
            traceback.print_exc()
            print("Will attempt to reconnect in 3 seconds...")
            time.sleep(3)  # Wait before retrying

# Process for running EMG
emg_process = None

# Function to shutdown EMG processing
def shutdown_emg():
    global emg_process, device
    
    if emg_process is not None and emg_process.is_alive():
        print("Shutting down EMG processing...")
        emg_process.terminate()
        emg_process.join(timeout=1.0)
        print("EMG processing terminated")
    
    if 'device' in globals():
        try:
            device.close()
            print("BITalino device closed")
        except:
            print("Error closing BITalino device")

# Register the shutdown function
atexit.register(shutdown_emg)

# Helper function to safely clear the queue
def clear_queue():
    """Clear all items from the queue"""
    count = 0
    while not emg_queue.empty():
        try:
            emg_queue.get_nowait()
            count += 1
        except:
            break
    if count > 0:
        print(f"Cleared {count} items from the queue")

# Now modify the main function to calculate UI latency
def main():
    # Initialize pygame
    pygame.init()
    
    # Create game manager
    screen_width, screen_height = 800, 600
    game_manager = GameManager(screen_width, screen_height)
    
    # Start EMG processing
    global emg_process
    
    if emg_initialized:
        print("Starting EMG processing")
        clear_queue()
        
        emg_process = multiprocessing.Process(
            target=process_emg_data,
            args=(model_processor, emg_queue)
        )
        emg_process.start()
    else:
        print("EMG components not initialized. Using keyboard controls only.")
    
    # Mapping of EMG predictions to game directions
    prediction_mapping = {
        'upward': 'up',
        'downward': 'down',
        'inward': 'left',
        'outward': 'right',
        'rest':'rest',
        'Upward':'up',
        'Downward':'down',
        'Left':'right',
        'Right':'left',
    }
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    frame_counter = 0
    
    # For latency display
    font = pygame.font.Font(None, 24)
    
    while running:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # For latency reporting
        frame_start_time = time.time()
        ui_update_performed = False
        
        # Check for new EMG data from queue
        if not emg_queue.empty():
            try:
                # Get prediction, intensity, and timestamp from the queue
                prediction, intensity, t_acq_val, t_feat_val, t_pred_val, prediction_timestamp = emg_queue.get_nowait()

                
                # UI update starts here
                t_ui_start = time.time()
                t_acq_values.append(t_acq_val)
                t_feat_values.append(t_feat_val)
                t_pred_values.append(t_pred_val)
                # Map the prediction to a game direction
                game_direction = prediction_mapping.get(prediction, prediction)
                
                # Pass data to the game manager
                game_manager.latest_prediction = game_direction
                game_manager.latest_intensity = intensity
                
                ui_update_performed = True
                print(f"Game using: {game_direction}, intensity={intensity:.2f}")
            except Exception as e:
                print(f"Error processing EMG data in game: {e}")
        
        # Update and render the game
        game_manager.handle_input()
        game_manager.update()
        game_manager.render()
        
        # If a UI update was performed this frame, calculate and log UI latency
        if ui_update_performed:
            t_ui_end = time.time()
            t_ui_duration = t_ui_end - t_ui_start
            
            # Add UI latency to moving average
            t_ui_values.append(t_ui_duration)
            
            # Calculate total latency - use moving averages for each component
            avg_t_acq = sum(t_acq_values) / len(t_acq_values) if t_acq_values else 0
            avg_t_feat = sum(t_feat_values) / len(t_feat_values) if t_feat_values else 0
            avg_t_pred = sum(t_pred_values) / len(t_pred_values) if t_pred_values else 0
            avg_t_ui = sum(t_ui_values) / len(t_ui_values) if t_ui_values else 0
            total_latency = avg_t_acq + avg_t_feat + avg_t_pred + avg_t_ui
            
            # Add to moving average of total latency
            total_latency_values.append(total_latency)
            
            # Log latency components
            latency_entry = {
                'timestamp': datetime.now().isoformat(),
                't_acq': avg_t_acq,
                't_feat': avg_t_feat,
                't_pred': avg_t_pred,
                't_ui': avg_t_ui,
                'total': total_latency
            }
            latency_log.append(latency_entry)
            
            # Display current latency on screen
            if frame_counter % 10 == 0:  # Update display every 10 frames
                print(f"Latency Components (ms): T_acq: {avg_t_acq*1000:.1f}, T_feat: {avg_t_feat*1000:.1f}, "
                      f"T_pred: {avg_t_pred*1000:.1f}, T_ui: {avg_t_ui*1000:.1f}")
                print(f"Total Latency: {total_latency*1000:.1f}ms")
            
            # Render latency information on screen
            latency_info = f"Latency: {total_latency*1000:.1f}ms"
            latency_surface = font.render(latency_info, True, (255, 255, 255))
            game_manager.screen.blit(latency_surface, (10, 10))
            
            # Additional detailed latency breakdown
            detail_y = 40
            for name, value in [('T_acq', avg_t_acq), ('T_feat', avg_t_feat), 
                              ('T_pred', avg_t_pred), ('T_ui', avg_t_ui)]:
                detail_text = f"{name}: {value*1000:.1f}ms"
                detail_surface = font.render(detail_text, True, (200, 200, 200))
                game_manager.screen.blit(detail_surface, (10, detail_y))
                detail_y += 25
            
            # Update display
            pygame.display.flip()
        
        # Increment frame counter
        frame_counter += 1
        
        # Cap the frame rate
        clock.tick(60)
    
    # Clean up when exiting
    pygame.quit()
    shutdown_emg()
    save_latency_log()  # Make sure to save latency data before exit
    sys.exit()

if __name__ == '__main__':
    main()