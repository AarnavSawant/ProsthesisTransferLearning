
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dense, LeakyReLU, LSTM, Reshape, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import pickle
import numpy as np
from data_organize_from_directory import plot_stacks, generate_signal_mae_tables, generate_signal_mae_tables_all_signals
from sklearn.model_selection import train_test_split
import copy
from tensorflow.keras.models import load_model


def build_generator(num_signals):
    '''
    Build Generator Model
    
    UNet Architecture

    Input: 200 x Num Signals

    Output: 200 x Num Signals

    Used as a translator between Prosthesis Domains
    
    '''
    alpha = 0
    model = tf.keras.Sequential()
    model.add(Input(shape=(200, num_signals)))

    # Convolutional layers with BatchNormalization
    # Output 100,16
    model.add(Conv1D(16, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    # Output 50, 32
    model.add(Conv1D(32, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    # Output 25, 64
    model.add(Conv1D(64, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    # Output 50, 32
    # Convolutional Transpose layers with BatchNormalization
    model.add(Conv1DTranspose(32, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())
    
    # Output 100,16
    model.add(Conv1DTranspose(16, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    # Output 200, 3
    model.add(Conv1DTranspose(num_signals, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(200*num_signals, activation='sigmoid'))
    model.add(Reshape((200, num_signals)))

    return model



'''
    Point this to your untrimmed norm stacks pkl file
'''
with open('/Users/aarnavsawant/Documents/EPICLab/GANProject/grouped_norm_stacks_untrimmed.pkl', 'rb') as f:
    grouped_norm_stacks = pickle.load(f)


'''
    Trimming Data to have 1 -> 1 pair matching between desired domains based on associated speed of signals 
'''
trimmed_norm_stacks = copy.deepcopy(grouped_norm_stacks)
min_sizes = {'0.3': float('inf'), '0.4': float('inf'), '0.5': float('inf'), '0.6': float('inf'), '0.7': float('inf'), '0.8': float('inf'), '0.9' : float('inf')}    
for device, speed_levels in grouped_norm_stacks.items():
    for speed, array in speed_levels.items():
        if speed in min_sizes.keys():
            current_size = array.shape[0]
            if current_size < min_sizes[speed]:
                min_sizes[speed] = current_size
                
for device, speed_levels in grouped_norm_stacks.items():
    for speed, array in speed_levels.items():
        if speed in min_sizes.keys():
            trimmed_norm_stacks[device][speed] = array[:min_sizes[speed], :, :]
            trimmed_norm_stacks[device][speed + "Speeds"] = grouped_norm_stacks[device][speed + "Speeds"][:min_sizes[speed]]






'''
 Helper functions to get particular data we want   
'''
real_device = 'PK' #Device we want to generate signal of
noise_device = 'OSL' #Device we want to use to generate signal


def get_all_data(device):
    '''
       Returns the untrimmed version of all the data for a particular device
       We use this particularly when generating all the synthetic signals aftert training the 
       generator
       (e.g. after training our generators, we use this to gather all the OSL data) 
    '''
    strides = []
    speeds = []
    for speed in grouped_norm_stacks[device]:
        if ("Speeds" not in speed):
            strides.append(grouped_norm_stacks[device][speed])
            speeds.append(grouped_norm_stacks[device][speed + "Speeds"])
    strides = np.concatenate(strides, axis=0)
    speeds = np.concatenate(speeds, axis=0)
    return strides, speeds


def get_real_and_noise_signals(real_device, noise_device, shuffle=False):
    '''
        Returns 1-1 pairings for our training and test sets for our generator training
        Uses the trimmed norm stacks to achieve this
    '''
    real_train = []
    real_train_labels = []
    noise_train = []
    real_test = {}
    noise_test = {}
    for speed in min_sizes.keys():
        real = trimmed_norm_stacks[real_device][speed]
        real_speeds = trimmed_norm_stacks[real_device][speed + "Speeds"]

        split_index = int(real.shape[0] * 0.8)
        if (shuffle):
            # Calculate the split index for 80%
            indices = np.random.permutation(real.shape[0])

            shuffled_real = real[indices]
            shuffled_real_speeds = [real_speeds[i] for i in indices]

            # Extract 80% of real data
            real_data_part = shuffled_real[:split_index]
            real_train.append(real_data_part)
            real_train_labels += shuffled_real_speeds[:split_index]
        


            # Keep the remaining 20% in the dictionary
            real_test[speed] = shuffled_real[split_index:]

            noise = trimmed_norm_stacks[noise_device][speed]
            # Similarly for noise data
            split_index = int(noise.shape[0] * 0.8)
            noise_data_part = noise[:split_index]
            noise_train.append(noise_data_part)
            noise_test[speed] = noise[split_index:]
            # Concatenate the 80% parts
        else:
            # Extract 80% of real data
            real_data_part = real[:split_index]
            real_train.append(real_data_part)
            real_train_labels += real_speeds[:split_index]
        


            # Keep the remaining 20% in the dictionary
            real_test[speed] = real[split_index:]

            noise = trimmed_norm_stacks[noise_device][speed]
            # Similarly for noise data
            split_index = int(noise.shape[0] * 0.8)
            noise_data_part = noise[:split_index]
            noise_train.append(noise_data_part)
            noise_test[speed] = noise[split_index:]
            # Concatenate the 80% parts
    real_train = np.concatenate(real_train, axis=0)
    noise_train = np.concatenate(noise_train, axis=0)

    return real_train, noise_train, real_train_labels, real_test, noise_test


'''
    Main Training Loop
'''
SHOULD_ENABLE_TRAINING = False

#Get the training and test data for generator training
#In this case, labels refers to the corresponding speeds for the PK data
real_train, noise_train, real_train_labels, real_test, noise_test = get_real_and_noise_signals(real_device=real_device, noise_device=noise_device, shuffle=True)


#Get all the noise data for generating all synthetic signals
all_noise_data, all_noise_speeds = get_all_data(noise_device)

epochs = 51 # Number of epochs for training
batch_size = 32
num_batches = int(real_train.shape[0] / batch_size)  # Calculate the number of batches per epoch
#Signal Groupings
signals = [['knee_theta', 'knee_thetadot', 'forceZ'],
            ['shank_accelX', 'shank_accelY', 'shank_accelZ'], 
            ['shank_gyroX', 'shank_gyroY', 'shank_gyroZ'], 
            ['thigh_accelX','thigh_accelY', 'thigh_accelZ'], 
            ['thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ'],
            ]
models = []

if (SHOULD_ENABLE_TRAINING):
    for i in range(len(signals)):
        generator = build_generator(len(signals[i]))
        generator.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.legacy.Adam(0.0001))
        for epoch in range(epochs):
            real_train, noise_train, real_train_labels, _, _ = get_real_and_noise_signals(real_device=real_device, noise_device=noise_device)
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                real_samples = real_train[start_idx:end_idx,:,i * 3 : i * 3 + 3]
                noise_samples = noise_train[start_idx:end_idx, :,i * 3 : i * 3 + 3]
                generated_samples = generator.predict(noise_samples)
            
                # Train the generator on noise samples
                g_loss = generator.train_on_batch(noise_samples, real_samples)
            
            # Optionally, log losses or save models
            print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {g_loss}")

            
            # Optionally, log losses or save models
            if (epoch) % 50 == 0:
                plot_stacks({'Generated':generated_samples, real_device:real_samples, noise_device:noise_samples}, signals[i], epoch=epoch + 1)
                generate_signal_mae_tables(real_samples, generated_samples, i + 1, signals[i],  "Training", epoch + 1)

        generator.save('osl_to_pk_group%d.h5' % i)
        models.append(generator)
else:
    '''
        Loading our Models
    '''
    for i in range(len(signals)):
        models.append(load_model('osl_to_pk_group%d.h5' % i))


'''
    Testing our generators on our test set
'''
results = {} 
all_gen= []
all_real = []
all_noise = []
for speed in real_test.keys():
    results[speed] = []
    for i in range(len(real_test[speed])):
        fullNoiseSignal = []
        fullRealSignal = []
        fullGenSignal = []
        for j in range(len(models)):
            print(real_test[speed][i,:,j*3:j*3 + 3].shape)
            real = real_test[speed][i,:,j*3:j*3 + 3].reshape(1,200,3)
            noise = noise_test[speed][i,:,j*3:j*3 + 3].reshape(1,200,3)
            gen = models[j].predict(noise)
            fullRealSignal.append(real)
            fullNoiseSignal.append(noise)
            fullGenSignal.append(gen)
        fullNoiseSignal = np.concatenate(fullNoiseSignal, axis=2)
        fullRealSignal = np.concatenate(fullRealSignal, axis=2)
        fullGenSignal = np.concatenate(fullGenSignal, axis=2)
        all_gen.append(fullGenSignal)
        all_real.append(fullRealSignal)
        all_noise.append(fullNoiseSignal)
all_gen = np.concatenate(all_gen, axis=0)
all_real = np.concatenate(all_real, axis=0)
all_noise = np.concatenate(all_noise, axis=0)
print("all real", all_real.shape)
generate_signal_mae_tables_all_signals(all_real, all_gen, signals, "Testing")
results[speed].append((fullNoiseSignal, fullGenSignal, fullRealSignal))


'''
    Passing all our noise data through the generators to get our synthetic signals
'''
synthetic_signals = {'data' : [], 'labels' : []}
for i in range(all_noise_data.shape[0]):
    full_gen_signal = []
    for j in range(len(models)):
        noise_signal = all_noise_data[i, :, j * 3 : j * 3 + 3]
        generated_signal = models[j].predict(noise_signal.reshape((1, noise_signal.shape[0], noise_signal.shape[1])))
        full_gen_signal.append(generated_signal)
    full_gen_signal = np.concatenate(full_gen_signal, axis=2)
    synthetic_signals['data'].append(full_gen_signal)
    synthetic_signals['labels'].append(all_noise_speeds[i])

print("Generated Synthetic Signals...")
synthetic_signals['data'] = np.concatenate(synthetic_signals['data'], axis=0)
print(synthetic_signals['data'].shape)


'''
    Uncomment to write the synthetic_signals pkl file
'''
# with open('synthetic_signals.pkl', 'wb') as f:
#     pickle.dump(synthetic_signals, f)
        
        
plot_stacks({'Generated':fullGenSignal, 'PK':fullRealSignal,'OSL':fullNoiseSignal}, ['knee_theta', 'knee_thetadot', 'forceZ',
            'shank_accelX', 'shank_accelY', 'shank_accelZ', 
            'shank_gyroX', 'shank_gyroY', 'shank_gyroZ', 
            'thigh_accelX','thigh_accelY', 'thigh_accelZ', 
            'thigh_gyroX', 'thigh_gyroY', 'thigh_gyroZ',
            ], plot_all_strides=True)


'''
    Uncomment to enable more detailed plotting
'''
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))

# # Defining unique colors for each combination
# colors = {
#     'Gen 0.9': 'blue', 
#     'Real 0.9': 'orange', 
#     'Input 0.9': 'green',
#     'Gen 0.3': 'red', 
#     'Real 0.3': 'purple', 
#     'Input 0.3': 'brown'
# }

# import matplotlib.pyplot as plt

# # Assuming 'results' is a pre-defined dictionary with the necessary data
# # results = {...}

# # Plot 1: Real Fast and Real Slow
# plt.figure(figsize=(10, 6))
# channel = 3
# for speed in ['0.3', '0.9']:
#     for real_signal, _, _ in results[speed]:
#         real_label = f'Real {speed.capitalize()}'
#         plt.plot(real_signal[:, :, channel].flatten(), label=real_label, color=colors[real_label])

# # Removing duplicate labels for plot 1
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.title('Real Signals: Fast vs Slow')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
# plt.show()

# # Plot 2: Gen Fast and Gen Slow
# plt.figure(figsize=(10, 6))
# for speed in ['0.3', '0.9']:
#     for _, _, gen_signal in results[speed]:
#         gen_label = f'Gen {speed.capitalize()}'
#         plt.plot(gen_signal[:, :, channel].flatten(), label=gen_label, color=colors[gen_label])

# # Removing duplicate labels for plot 2
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.title('Generated Signals: Fast vs Slow')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
# plt.show()

# plt.figure(figsize=(10, 6))

# for speed in ['0.3', '0.9']:
#     for _, input_signal, _ in results[speed]:
#         input_label = f'Input {speed.capitalize()}'
#         plt.plot(input_signal[:, :, channel].flatten(), label=input_label, color=colors[input_label])

# # Removing duplicate labels
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())

# plt.title('Input Signals: Fast vs Slow')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
# plt.show()

# # Plot 1: Real Fast vs Gen Fast
# plt.figure(figsize=(10, 6))
# channel = 3
# for speed in ['0.3', '0.9']:
#     for real_signal, _, gen_signal in results[speed]:
#         real_label = f'Real {speed.capitalize()}'
#         gen_label = f'Gen {speed.capitalize()}'
#         plt.plot(real_signal[:, :, channel].flatten(), label=real_label, color=colors[real_label])
#         plt.plot(gen_signal[:, :, channel].flatten(), label=gen_label, color=colors[gen_label])

# # Removing duplicate labels for plot 1
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.title('Comparison: Real Fast vs Gen Fast')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
# plt.show()

# # Plot 2: Real Slow vs Gen Slow
# plt.figure(figsize=(10, 6))
# for speed in ['0.3']:
#     for real_signal, _, gen_signal in results[speed]:
#         real_label = f'Real {speed.capitalize()}'
#         gen_label = f'Gen {speed.capitalize()}'
#         plt.plot(real_signal[:, :, channel].flatten(), label=real_label, color=colors[real_label])
#         plt.plot(gen_signal[:, :, channel].flatten(), label=gen_label, color=colors[gen_label])

# # Removing duplicate labels for plot 2
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.title('Comparison: Real Slow vs Gen Slow')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
# plt.show()

# # Plot 3: All Four (Real Fast, Gen Fast, Real Slow, Gen Slow)
# plt.figure(figsize=(10, 6))
# for speed in ['0.9', '0.3']:
#     for real_signal, _, gen_signal in results[speed]:
#         real_label = f'Real {speed.capitalize()}'
#         gen_label = f'Gen {speed.capitalize()}'
#         plt.plot(real_signal[:, :, channel].flatten(), label=real_label, color=colors[real_label])
#         plt.plot(gen_signal[:, :, channel].flatten(), label=gen_label, color=colors[gen_label])

# # Removing duplicate labels for plot 3
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.title('All Signals: Real Fast, Gen Fast, Real Slow, Gen Slow')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
# plt.show()
