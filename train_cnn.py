import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the processed data
with open('synthetic_signals.pkl', 'rb') as f:
    synthetic_signals = pickle.load(f)


with open('/Users/aarnavsawant/Documents/EPICLab/GANProject/grouped_norm_stacks_untrimmed.pkl', 'rb') as f:
    grouped_norm_stacks = pickle.load(f)

real_device = 'PK' #Device we want to generate signal of
real_test = []
real_test_labels = []
for speed in ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:

    # Extract 80% of real data
    real_data_part = grouped_norm_stacks[real_device][speed]
    real_test.append(real_data_part)
    real_test_labels += grouped_norm_stacks[real_device][speed + "Speeds"]


# X = np.concatenate(real_test, axis=0)
# y = np.array(real_test_labels)

X = synthetic_signals['data']
y = np.array(synthetic_signals['labels'])


print(X.shape)
# Ensure all sequences have the same length for CNN input
# def pad_sequences(sequences, maxlen=None, dtype='float32', padding='pre', truncating='pre', value=0.):
#     lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
#     num_samples = len(sequences)
#     if maxlen is None:
#         maxlen = np.max(lengths)

#     # Initialize the output array
#     x = np.full((num_samples, maxlen, sequences[0].shape[-1]), value, dtype=dtype)
#     for idx, s in enumerate(sequences):
#         if not len(s):
#             continue  # Skip empty sequences
#         if truncating == 'pre':
#             trunc = s[-maxlen:]
#         else:
#             trunc = s[:maxlen]
#         trunc = np.asarray(trunc, dtype=dtype)
#         if padding == 'post':
#             x[idx, :len(trunc)] = trunc
#         else:
#             x[idx, -len(trunc):] = trunc
#     return x

# # Preprocess inputs
# max_stride_length = max(max([len(x) for x in X_train]), max([len(x) for x in X_test]))
# X_train_pad = pad_sequences(X_train, maxlen=max_stride_length)
# X_test_pad = pad_sequences(X_test, maxlen=max_stride_length)

# # Define the CNN model
def create_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, padding='same', activation='linear', input_shape=input_shape),
        LeakyReLU(alpha=0.01),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, padding='same', activation='linear'),
        LeakyReLU(alpha=0.01),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='leaky_relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
    return model

def generate_sample_windows(stack, speed, sample_size=50, stride=2):
    extracted_features = []
    extracted_speeds = []
    for i in range(0, stack.shape[0] - sample_size, stride):
        window = stack[i:i+sample_size, :]
        extracted_features.append(window)
        extracted_speeds.append(speed)
    extracted_features = np.array(extracted_features)
    extracted_speeds = np.array(extracted_speeds)
    return extracted_features, extracted_speeds

real_train_sampled = []
real_train_speeds_sampled = []
for i in range(X.shape[0]):
    sampled_stacks, sampled_speed = generate_sample_windows(X[i], y[i])
    real_train_sampled.append(sampled_stacks)
    real_train_speeds_sampled.append(sampled_speed)

real_train_sampled = np.array(real_train_sampled)
real_train_speeds_sampled = np.array(real_train_speeds_sampled)

real_train_sampled = real_train_sampled.reshape((-1, real_train_sampled.shape[-2], real_train_sampled.shape[-1]))
real_train_speeds_sampled = real_train_speeds_sampled.reshape((-1))

# # Initialize and train the model
model = create_cnn((X.shape[1], X.shape[2]))
model.summary()

print(X.shape)

# history = model.fit(X, y, epochs=250, batch_size=32, validation_split=0.2)

model.save("")

with open('/Users/aarnavsawant/Documents/EPICLab/GANProject/grouped_norm_stacks_untrimmed.pkl', 'rb') as f:
    grouped_norm_stacks = pickle.load(f)

real_device = 'PK' #Device we want to generate signal of
real_test = []
real_test_labels = []
for speed in ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:

    # Extract 80% of real data
    real_data_part = grouped_norm_stacks[real_device][speed]
    real_test.append(real_data_part)
    real_test_labels += grouped_norm_stacks[real_device][speed + "Speeds"]


real_test = np.concatenate(real_test, axis=0)
real_test_labels = np.array(real_test_labels)
model = load_model("cnn.h5")
test_loss = model.evaluate(real_test, real_test_labels)

print("Test Shape", real_test.shape)
pred = model.predict(real_test)
# print(real_test_labels)

plt.plot(pred)
plt.plot(real_test_labels)
plt.show()


# model.save("cnn.h5")
print(f'Test Loss: {test_loss}')

    # Keep the remaining 20% in the dictionary

# # Evaluate the model
# test_loss = model.evaluate(X_test_pad, y_test)
# print(f'Test Loss: {test_loss}')

# # Save the model
# model.save('cnn_model.h5')
# print("Model has been successfully trained and saved.")
