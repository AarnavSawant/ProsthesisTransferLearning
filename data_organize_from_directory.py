"""
U-Net: Convolutional Networks for BiomedicalImage Segmentation (Ronneberger 2015)
    - https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    
Image-to-Image Translation with Conditional Adversarial Networks (Isola 2018)
    - Conditional GANs. Used shape of Unet with PatchGAN Classifier
    - https://github.com/phillipi/pix2pix
    
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Zhu 2017)
    - Unpaired Cyclic GANs + L1 Loss. PatchGAN Classifier
    - https://github.com/junyanz/CycleGAN
    
Human Activity Recognition Based on Motion Sensor Using U-Net (Zhang 2019)
    - HAR with pixel-to-pixel classification avoiding sliding window. Modified U-net.
    - https://github.com/zhangzhao156/Human-Activity-Recognition-Codes-Datasets
    
AdaptNet: Human Activity Recognition via Bilateral Domain Adaptation 
Using Semi-Supervised Deep Translation Networks (An 2021)
    - HAR with AdaptNet for domain adaptation. U-net shape Generator and Classification Discriminator
    - https://github.com/ast0414/AdaptNet
    
Generating synthetic gait patterns based on benchmark datasets for controlling prosthetic legs
    - Adapts GAN from pix2pix (second reference from above)
    

---------------------------------------------------------------
Real2Sim Prostheses Walking Speed Estimation in the Osim domain
    
                             + Reverse U-net
Data (Prosthesis Domain) -> U-net (Generator) -> Simulated Data (Osim Domain) -> PatchGAN (Discriminator)
            '------------------------------------------------------------------------^
            
Prosthesis to Osim Domain Transfer
    - Use cases
        - Sensors differ in quality and placement
        - Differences in mechanical design influence data
    - Pros
        - New prosthesis only need to train 1 Generator to access all ML models
        - New prosthesis does not need to collect multi-speed, multi-slope, multi-mode dataset for a baseline model
    - Cons
        - Attaining Osim data (kinematics + simulated IMUs). Fixed by cheaper ways to do so.
    
Analyses and Comparisons
    - RealOSL vs SimOSL Data
    - RealPK vs SimPK Data
    - RealOSL ML vs SimOSL ML (similar performance): baseline perfomance across domains (real vs sim)
    - SimOSL ML vs SimPK ML (similar performance): baseline performance across prostheses (osl vs pk)
    - Unpaired training of real2sim data
    

Steps
    1. Process OSL, PK, and Osim data
    2. Train non-cyclic GAN
    3. Train cyclic GAN
    4. Optimize GAN
    5. Test translations with models trained with Osim data
        - OSL real on RealML / vs / OSL sim on SimML (similar)    
        - OSL sim on Sim ML/ vs / PK sim on Sim ML (similar)
    6. Real-time testing
---------------------------------------------------------------
PK2OSL Walking Speed Estimation
    - Train a translator from OSL to PK
    - Only focus on stance phase

The generator consists of:
Input-C16-C32-C64-T32-T16-T4-D200-Output

The discriminator consists of:
Input-C16-C32-C64-L80-Output

Here, Ck denotes a one-dimensional (1D) convolution
layer with k filters, and Tk denotes a transposed 1D con-
volution layer with k filters. All convolution layers have the
same padding, a kernel size of 2, a stride length of 2, and
the LeakyReLU activation function. Dk denotes a dense
layer of k units with sigmoid activation.

Here, Lk denotes an LSTM layer of k units with
LeakyReLU activation.
The GAN was trained for 500 epochs ( [22] of 1000,
with a batch size of 64, the ADAM optimizer with a learn-
ing rate of 0.01, and with the mean absolute error as the
loss function).

"""
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
import math
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dense, LeakyReLU, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from matplotlib.table import table


pd.options.mode.chained_assignment = None  # None means no warning will be printed

device_colors = {
    'OSL':'b',
    'PK': 'r',
    'AB':'k',
    'Generated':'g'
    }

subject_mapping = {
    'PK01':'R', 
    'PK02':'R', 
    'PK03':'R', 
    'PK04':'R', 
    'PK05':'L', 
    'PK06':'L', 
    'PK07':'L', 
    'PK08':'R', 
    'PK09':'R',
    'PK10':'R',
    'AB06':'R',
    'AB30':'R'
    }

# PK --> OSL Shank IMU (ankle motor)
pk_to_osl_mapping = {
    'shank_accelX':'-shank_accelY',
    'shank_accelY':'shank_accelZ',
    'shank_accelZ':'shank_accelX',
    'shank_gyroX':'shank_gyroY',
    'shank_gyroY':'shank_gyroZ',
    'shank_gyroZ':'shank_gyroX',
    
    'thigh_accelX':'thigh_accelX',
    'thigh_accelY':'thigh_accelY',
    'thigh_accelZ':'thigh_accelZ',
    'thigh_gyroX':'thigh_gyroX',
    'thigh_gyroY':'thigh_gyroY',
    'thigh_gyroZ':'thigh_gyroZ',
    }


    # Global Sensors (OSL Frame):
    #     Shank IMU in deg/sec - x posterior, y inferior, z lateral
    #     Thigh IMU in rad/sec - x inferior, y medial, z posterior
    #     Knee angle in deg
    #     knee vel in deg/sec
    #     GRF in N
    
# AB to OSL/PK signals
camargo_to_osl_mapping = {
    'Header':'header',
    'thigh_Accel_X':'thigh_accelX', # Good
    'thigh_Accel_Y':'thigh_accelY', # Good
    'thigh_Accel_Z':'thigh_accelZ', # Good
    'thigh_Gyro_X':'thigh_gyroX', # Good
    'thigh_Gyro_Y':'thigh_gyroY', # Good
    'thigh_Gyro_Z':'thigh_gyroZ', # Good
    'shank_Accel_X':'-shank_accelY',
    'shank_Accel_Y':'shank_accelZ',
    'shank_Accel_Z':'shank_accelX',
    'shank_Gyro_X':'shank_gyroZ',
    'shank_Gyro_Y':'shank_gyroY',
    'shank_Gyro_Z':'shank_gyroX',
    'Treadmill_R_vy':'forceZ',
    'knee_sagittal':'-knee_theta',
    'knee_sagittal_velocity':'-knee_thetadot'
    }


# OSL Data
osl_data_dict = {5:'Trial5_filled',
              6:'Trial6_filled',
              7:'Trial7_filled',
              8:'Trial8_filled'}

# Shared signals
signals = ['knee_theta', 'knee_thetadot', 'forceZ',
            'shank_accelX', 'shank_accelY', 'shank_accelZ', 
            'shank_gyroX', 'shank_gyroY', 'shank_gyroZ', 
            'thigh_accelX','thigh_accelY', 'thigh_accelZ', 
            'thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ',
            ]



def get_stack(paths, device='OSL'):
    stack = []
    stack_speeds = []
    for path in paths:
        # Read
        bag_df = pd.read_csv(path)
        
        # Sync
        if 'sync' in bag_df.columns:
            bag_df = bag_df[(bag_df.sync == 1).idxmax():]
            bag_df.header = bag_df.header - bag_df.header.iloc[0]
        
        # Flip signals, rename, or convert values to match OSL
        # MICROSTRAINS Read in G and rad/sec
        if device == 'PK':
            bag_df_copy = bag_df.copy()
            for col in pk_to_osl_mapping.keys():
                swap_col = pk_to_osl_mapping[col]
                if swap_col[0] == '-':
                    multiplier = -1
                    swap_col = swap_col[1:]
                else:
                    multiplier = 1
                
                # G to m/s^2
                # if 'accel' in swap_col: 
                #     multiplier *= 9.8

                # Rad/sec to deg/sec
                if 'gyro' in swap_col:
                    multiplier *= 180/np.pi

                bag_df[col] = multiplier * bag_df_copy[swap_col]
                
        # OSL reads G and deg/sec at Shank DEPHY IMU
        elif device == 'OSL':
            bag_df_copy = bag_df.copy()
            for col in bag_df.columns:
                multiplier = 1

                # Meant to match the shank accel units of the DEPHY ankle IMU 
                if 'shank_accel' in col: 
                    multiplier *= 9.8
                    
                # Rad/sec to deg/sec, shank IMU is already deg/sec
                if 'thigh_gyro' in col:
                    multiplier *= 180/np.pi
                    
                bag_df[col] = multiplier * bag_df_copy[col]
                
        # Extract gait events
        heel_strike_indices, toe_off_indices = get_gait_events(bag_df)
        if device == 'OSL':
            print(heel_strike_indices)
            print(toe_off_indices)
            print()
        
        for i in range(1,len(heel_strike_indices)):
            
            # Extract start and end of stride
            start_time = bag_df['header'].iloc[heel_strike_indices[i-1]]
            end_time = bag_df['header'].iloc[heel_strike_indices[i]]
            
            elapsed_time = end_time - start_time
            if elapsed_time < 1 or elapsed_time > 3: # Skip abnormaly short or long strides
                continue
            
            # Extract stride
            stride = bag_df[(bag_df.header >= start_time) & (bag_df.header <= end_time)]
            
            # Zero
            stride['header'] = stride['header'] -  stride['header'].iloc[0]
            
            if device == 'OSL':
                if ('forward' in stride.keys()):
                    stack_speeds.append(stride['forward'].mean())
                elif ('speed' in stride.keys()):
                    stack_speeds.append(stride['speed'].mean())
                elif ('Speed' in stride.keys()):
                    stack_speeds.append(stride['Speed'].mean())
            elif device == 'PK':
                stack_speeds.append(stride['speed'].mean())
            else:
                stack_speeds.append([0])
            
            # Add to stack
            stride = stride[signals].apply(lambda x: resample(x, 200))
            stack += [stride.values]
            
    return np.array(stack), np.array(stack_speeds)

def get_gait_events(df, min_elapsed_time = 0.5, heel_percentage = 0.20, toe_percentage = 0.20, time_col='header', fz_col = 'forceZ'):
    
        # Required columns
        time = df[time_col]
        vertical_load = df[fz_col]
        
        # Load thresholds
        heel_load_threshold = vertical_load.max()*heel_percentage
        toe_load_threshold = vertical_load.max()*toe_percentage

        # Derivative
        # diff = vertical_load.diff()
        diff = vertical_load.diff().rolling(window=5, min_periods=1).mean()

        # Below logic alternates between searching for HS or TO
        # If HS, Fz must surpass threshold, be increasing, and be min_elapsed_time ahead of last HS
        # If TO, Fz must is similar, but also checks that it is min_elapsed_time/2 ahead of HS
        heel_strike_indices = []
        toe_off_indices = []
        
        hs_flag = False
        for i in range(len(diff)):
            if vertical_load.iloc[i] > heel_load_threshold and diff.iloc[i] > 0 and hs_flag == True:
                if heel_strike_indices:
                    if time.iloc[i] - time.iloc[heel_strike_indices[-1]] > min_elapsed_time:
                        heel_strike_indices.append(i)
                        hs_flag = False                
                else:
                    heel_strike_indices.append(i)
                    hs_flag = False
            elif vertical_load.iloc[i] < toe_load_threshold and diff.iloc[i] < 0 and hs_flag == False:
                if toe_off_indices:
                    if time.iloc[i] - time.iloc[toe_off_indices[-1]] > min_elapsed_time and time.iloc[i] - time.iloc[heel_strike_indices[-1]] > min_elapsed_time/2:
                        toe_off_indices.append(i)
                        hs_flag = True                
                else:
                    toe_off_indices.append(i)
                    hs_flag = True 
        
        return heel_strike_indices, toe_off_indices
    
def plot_stacks(stacks, channel_names, epoch=-1, plot_all_strides=False):
    '''
    Plot stack
    
    Global Sensors (OSL Frame):
        Shank IMU in deg/sec - x posterior, y inferior, z lateral
        Thigh IMU in rad/sec - x inferior, y medial, z posterior
        Knee angle in deg
        knee vel in deg/sec
        GRF in N
    
    Power Knee Sensors:
        Shank Microstrain IMU - reads in rad/sec, requires orientation swaps and rad/sec -> deg/sec
        Thigh Microstrain IMU - reads in deg
        PK Knee Angle - reads in deg
        PK Knee Angular Velocity - reads in deg/sec, noisy, not filtered
        PK Vertical GRF - reads in N/A, seems to be capped or need scaling
        
    OSL Sensors
        Shank DEPHY IMU (ankle) - reads in deg/sec, appears to remove gravity from signals
        Thigh Microstrain IMU - reads in rad/sec
        Knee Angle - reads in deg
        Knee Angular Velocity - filtered within MAV of length 3, deg/sec
        Loadcell Fz - filtered with 4th order filter, Newtons
    
    Camargo Sensors
        Shank IMUs - 
        Thigh IMU - 
        GON Knee Angle - 
        GON Knee Angular Velocity - 
        FP Fz - 
    
    
    '''
    for i, signal in enumerate(channel_names):
        plt.figure(figsize=(10, 4))
    
        if plot_all_strides:
            for stack_name, stack in stacks.items():
                print(stack_name,np.shape(stack))
                legend_added = False
                n, s, c = np.shape(stack)
                color = device_colors[stack_name]
            
                for j in range(n):
                    y = stack[j,:,i]
                    x = np.linspace(0, 99, len(y))
                    plt.plot(x, y, color=color)
        
        for stack_name, stack in stacks.items():
            color = device_colors[stack_name]
            if plot_all_strides:
                color = 'k' 
            
            average_signal = np.mean(stack,axis=0)[:,i]
            std_signal = np.std(stack,axis=0)[:,i]
            x = np.linspace(0, 99, len(average_signal))
            plt.plot(x, np.mean(stack,axis=0)[:,i], color=color, label=stack_name)
            plt.fill_between(x, average_signal - std_signal, average_signal + std_signal, color=color, alpha=0.25)  # Standard deviation shading
        
        plt.title(f'{signal}')
        plt.xlabel('Gait Cycle %')
        plt.ylabel(signal)
        plt.legend()
        # plt.show()
        if (epoch != -1):
            plt.savefig(f'{signal}-epoch{epoch}.jpg')


def generate_signal_mae_tables(real_stack, generated_stack, signalGroup, channel_names, label, epoch):
    '''
    Generate Signal MAE Tables (for one individua)
    
    Real_Stack: Strides we are attempting to emulate
    
    '''
    tableVals = [channel_names]
    values = []
    for i, signal in enumerate(channel_names):
         gen_stride = generated_stack[:, :, i]
         print("Gen Stride Shape", gen_stride.shape)
         real_stride = real_stack[:, :, i]
         mae = mean_absolute_error(real_stride, gen_stride)
         values.append(round(mae, 4))
    tableVals.append(values)
    # Create the table
    fig, ax = plt.subplots()
    ax.set_title("%s Signal Group %d MAE Epoch %d" % (label, signalGroup, epoch))
    table = ax.table(cellText=tableVals, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # Adjust table size
    ax.axis('off')  # Turn off axis

    plt.savefig("%ssignal-group%depoch%d.jpg" % (label, signalGroup, epoch))

def generate_signal_mae_tables_all_signals(real_stack, generated_stack, channel_groups, label):
    '''
    Generate Signal MAE Tables (for a set of strides)
    
    Real_Stack: Strides we are attempting to emulate (NUM_STRIDES x 200 x NUM_SIGNALS)
    Generated_Stack: Strides generated by our generator (NUM_STRIDES x 200 x NUM_SIGNALS)
    Channel_Groups: The corresponding signals and their groups (2D List)
    Label: String to pass that will be displayed in title

    The output will be a table that contains the MAE for each signal
    '''
    tableVals = []
    print(generated_stack.shape)
    print(real_stack.shape)
    print(channel_groups)
    tableVals.append(["Signal", "MAE (Gen vs Real)"])
    for j, group in enumerate(channel_groups):
        values = []
        for i, signal in enumerate(group):
            gen_stride = generated_stack[:, :, j * 3 + i]
            print("Gen Stride Shape", gen_stride.shape)
            real_stride = real_stack[:, :, j * 3 + i]
            mae = mean_absolute_error(real_stride, gen_stride)
            print(mae)
            # values.append(round(mae, 4))
            tableVals.append([signal, round(mae, 4)])
        # Create the table
    fig, ax = plt.subplots()
    print(tableVals)
        # ax.title("%s MAE" % (label))
    val = 0
    for i in range(1, len(tableVals)):
        val += tableVals[i][1]
    val = val / 15.0
    tableVals.append(["Average", round(val, 4)])
    table = ax.table(cellText=tableVals, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    ax.set_title("%s MAE of Generated Signals versus Real Signals" % label)
    table.scale(1.2, 1.2)  # Adjust table size
    # Set column widths to fit 15 columns
    ax.axis('off')  # Turn off axis

    plt.savefig("%s.jpg" % (label))





        
# Get normalizition values
def min_max_normalization(stacks):
    min_max_vals = []
    num_channels = np.shape(stacks[list(stacks.keys())[0]])[2]
    for i in range(num_channels):
        global_min, global_max = math.inf, -math.inf
        for stack_name, stack in stacks.items():
            n, s, c = np.shape(stack)
            for j in range(n):
                stride_data = stack[j,:,i]
                local_min = min(stride_data)
                local_max = max(stride_data)
                
                global_min = min(global_min,local_min)
                global_max = max(global_max,local_max)
            
        min_max_vals.append([global_min, global_max])
    
    norm_stacks = {}
    stacks_copy = stacks.copy()
    for stack_name, stack in stacks_copy.items():
        for i in range(num_channels):
            norm_min, norm_max = min_max_vals[i]
            n, s, c = np.shape(stack)
            for j in range(n):
                stride_data = stack[j,:,i]
                normalized_stride = ((stride_data - norm_min) / (norm_max - norm_min))
                stack[j,:,i] = normalized_stride
        norm_stacks[stack_name] = stack
        
    return norm_stacks

def main():
    
    #PK Subject Path
    subject_path = "/Users/aarnavsawant/Documents/EPICLab/GANProject/PK/CSV/TF25_03_05_24"
    folders = os.listdir(subject_path)
    # Get PK paths to CSVs
    PK_paths = []
    for trial_number in folders:
        if trial_number.isnumeric(): #Remove .DS_Store and Key.csv from appearing
            new_path = subject_path + "/" + trial_number
            print(os.listdir(new_path))
            PK_paths.append(new_path + "/" + trial_number + ".csv")


    # Get OSL paths to CSVs
    main_directory = '/Users/aarnavsawant/Documents/EPICLab/GANProject/OSL'
    csv_directory = f'{main_directory}/CSV/'
    OSL_paths = [] 
    for subject in os.listdir(csv_directory): 
        new_path = csv_directory + subject
        folders = os.listdir(new_path)
        for trial_number in folders:
                if trial_number.isnumeric() and (trial_number != '9' or trial_number != '10'):
                    bag_path = f'{new_path}/{trial_number}/{trial_number}.csv'
                    OSL_paths.append(bag_path)    

        
    # Get stack of strides. Shape is (N, S, C)
    OSL_stack, OSL_speeds = get_stack(OSL_paths)
    PK_stack, PK_speeds = get_stack(PK_paths, device='PK')



    # Get only the signals we need
    # channels = ['knee_theta', 'knee_thetadot','forceZ','shank_accelY']
    channels = signals
    indices = []
    for c in channels:
        indices.append(signals.index(c))
    PK_stack = PK_stack[:,:,indices]
    OSL_stack = OSL_stack[:,:,indices]

    with open('OSL_speeds.pkl', 'wb') as f:
        pickle.dump(OSL_speeds, f)

    with open('PK_speeds.pkl', 'wb') as f:
        pickle.dump(PK_speeds, f)
    
    # Plot stacks
    stacks = {'OSL':OSL_stack, 'PK':PK_stack}

    
    # Plot normalized stacks
    norm_stacks = min_max_normalization(stacks)
    plot_stacks(norm_stacks, channels, plot_all_strides=True)
    
    # # Trim stacks to match N
 
    
    # Save
    with open('norm_stacks.pkl', 'wb') as f:
        pickle.dump(norm_stacks, f)
    
    
    # Assuming OSL_stack and OSL_speeds are already defined
    # OSL_stack shape: (Number Strides, Stride Length, Sensor Channel)
    # OSL_speeds shape: (Number Strides, Stride Length)
    grouped_norm_stacks = {}
    temp_dict = {'OSL':OSL_speeds,'PK':PK_speeds}
    for device in temp_dict.keys():
        
        stack = norm_stacks[device]
        speeds = temp_dict[device]
        min_speed = np.min(speeds)
        max_speed = np.max(speeds)
        
        # Step 2: Define speed thresholds
        speed_range = max_speed - min_speed
        speed_thresholds = {
            '0.3': 0.35,
            '0.4': 0.45,
            '0.5': 0.55,
            '0.6': 0.65,
            '0.7': 0.75,
            '0.8': 0.85,
            '0.9': 0.95
        }
        print(device, speed_thresholds)
        
        # Step 3: Categorize Each Stride
        stride_categories = ['0.3' if speed <= speed_thresholds['0.3'] else
                             '0.4' if speed <= speed_thresholds['0.4'] else
                             '0.5' if speed <= speed_thresholds['0.5'] else
                             '0.6' if speed <= speed_thresholds['0.6'] else
                             '0.7' if speed <= speed_thresholds['0.7'] else
                             '0.8' if speed <= speed_thresholds['0.8'] else
                             '0.9' for speed in speeds]
        
        # Step 5: Group Strides in stack Based on These Categories
        # Store speed corresponding to each stride for classification experiments later on
        grouped_strides = {'0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [],'0.9': [],
                           '0.3Speeds': [], '0.4Speeds': [], '0.5Speeds': [],'0.6Speeds': [], '0.7Speeds': [], '0.8Speeds': [],'0.9Speeds': [],}

        grouped_strides_with_speeds = grouped_strides

        for idx, category in enumerate(stride_categories):
            if "Speeds" not in category:
                grouped_strides_with_speeds[category].append(stack[idx])
                grouped_strides_with_speeds[category + 'Speeds'].append(speeds[idx])
        
        # Convert lists to numpy arrays for easier manipulation later if needed
        for category in grouped_strides:
            if "Speeds" not in category:
                grouped_strides_with_speeds[category] = np.array(grouped_strides_with_speeds[category])
                print("Category", category, "Shape", grouped_strides_with_speeds[category].shape)
            
        grouped_norm_stacks[device] = grouped_strides_with_speeds
    

    #Upsample so that we can do 1-1 pairing later on
    # max_sizes = {'slow': -float('inf'), 'medium': -float('inf'), 'fast': -float('inf')}
    # for device, speed_levels in grouped_norm_stacks.items():
    #    for speed, array in speed_levels.items():
    #         if speed in max_sizes.keys():
    #             current_size = array.shape[0]
    #             if current_size > max_sizes[speed]:
    #                 max_sizes[speed] = current_size
    # for device, speed_levels in grouped_norm_stacks.items():
    #     for speed, array in speed_levels.items():
    #         if speed in max_sizes.keys():
    #             speeds = grouped_norm_stacks[device][speed + "Speeds"]
    #             print(array.shape[0])
    #             print(max_sizes[speed])

    #             #max_sizes is the maximum amount of data out of PK/OSL for a particular speed group
    #             while (array.shape[0] < max_sizes[speed]):
    #                 print(array.shape)

    #                 #find the difference between the maximum and the current array (that's how much we need to append)
    #                 diff = max_sizes[speed] - array.shape[0]

    #                 #if the difference is greater than the entries in the array, add the whole array
    #                 if (diff > array.shape[0]):
    #                     array = np.concatenate((array, array[:, :, :]), axis=0)
    #                     speeds += speeds
    #                 else:
                        
    #                     #else add the difference
    #                     array = np.concatenate((array, array[:diff, :, :]), axis=0)
    #                     speeds += speeds[:diff]

                        
    #             grouped_norm_stacks[device][speed] = array
    #             grouped_norm_stacks[device][speed + "Speeds"] = speeds

    # print(grouped_norm_stacks['OSL']['fast'].shape)
    # print(grouped_norm_stacks['PK']['fast'].shape) 
    # print(grouped_norm_stacks['OSL']['slow'].shape)
    # print(grouped_norm_stacks['PK']['slow'].shape)
    # print(grouped_norm_stacks['OSL']['medium'].shape)
    # print(grouped_norm_stacks['PK']['medium'].shape)  
    # print(len(grouped_norm_stacks['OSL']['fastSpeeds']))
    # print(len(grouped_norm_stacks['PK']['fastSpeeds']))
    # print(len(grouped_norm_stacks['OSL']['slowSpeeds']))
    # print(len(grouped_norm_stacks['PK']['slowSpeeds']))
    # print(len(grouped_norm_stacks['OSL']['mediumSpeeds']))
    # print(len(grouped_norm_stacks['PK']['mediumSpeeds']))                     

              
    # Trim
    # min_sizes = {'slow': float('inf'), 'medium': float('inf'), 'fast': float('inf')}
    
    # for device, speed_levels in grouped_norm_stacks.items():
    #     for speed, array in speed_levels.items():
    #         if speed in min_sizes.keys():
    #             current_size = array.shape[0]
    #             if current_size < min_sizes[speed]:
    #                 min_sizes[speed] = current_size
                
    # for device, speed_levels in grouped_norm_stacks.items():
    #     for speed, array in speed_levels.items():
    #         if speed in min_sizes.keys():
    #             grouped_norm_stacks[device][speed] = array[:min_sizes[speed], :, :]
    #             grouped_norm_stacks[device][speed + "Speeds"] = grouped_norm_stacks[device][speed + "Speeds"][:min_sizes[speed]]
    
    with open('grouped_norm_stacks_untrimmed.pkl', 'wb') as f:
        pickle.dump(grouped_norm_stacks, f)
# main()