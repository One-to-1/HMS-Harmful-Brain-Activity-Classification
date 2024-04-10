# %% [markdown]
# # HMS - Harmful Brain Activity Classification
# 
# > The objective of this notrbook is to classify seizures and other patterns of harmful brain activity in critically ill patients
# 
# This project has the foollowing objectives
# * Loading the data efficiently using [`tf.data`](https://www.tensorflow.org/guide/data).
# * Creating the model using KerasCV presets.
# * Training the model.
# * Inference and Submission on test data.
# 

# %% [markdown]
# ## Import Libraries

# # %%
# !pip install -q /kaggle/input/kerasv3-lib-ds/keras_cv-0.8.2-py3-none-any.whl --no-deps
# !pip install -q /kaggle/input/kerasv3-lib-ds/tensorflow-2.15.0.post1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps
# !pip install -q /kaggle/input/kerasv3-lib-ds/keras-3.0.4-py3-none-any.whl --no-deps

# %%
import os
os.environ["KERAS_BACKEND"] = "jax" # you can also use tensorflow or torch
import tensorflow as tf
import keras_cv
import keras
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm # python library for progress bar
import joblib
import matplotlib.pyplot as plt
# %%
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasCV:", keras_cv.__version__) # Check for compatibility
# %% [markdown]
# ## Config
# %%
class Config:
    fold = 0 # Which fold to set as validation data
    randSeed = 42  # Random seed
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    batchSize = 64  # Batch size
    verbosity = 2  # Verbosity
    dropRem = True  # Drop incomplete batches
    epochs = 13 # Training epochs
    lrMode = "cos" # LR scheduler mode from one of "cos", "step", "exp"
    numClasses = 6 # Number of classes in the dataset
    imageSize = [400, 300]  # Input image size
    classNames = ['Seizure', 'LPD', 'GPD', 'LRDA','GRDA', 'Other']
    label2name = dict(enumerate(classNames))
    name2label = {v:k for k, v in label2name.items()}
# Set a value for the random seed to be able to reproduce a speciffic outcome.
keras.utils.set_random_seed(Config.randSeed)

# %% [markdown]
# ## Dataset Path
# %%
BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
SPEC_DIR = "/tmp/dataset/hms-hbac"
os.makedirs(SPEC_DIR+'/train_spectrograms', exist_ok=True)
os.makedirs(SPEC_DIR+'/test_spectrograms', exist_ok=True)
## Meta Data
# Train + Valid
df = pd.read_csv(f'{BASE_PATH}/train.csv')
df['eeg_path'] = f'{BASE_PATH}/train_eegs/'+df['eeg_id'].astype(str)+'.parquet'
df['spec_path'] = f'{BASE_PATH}/train_spectrograms/'+df['spectrogram_id'].astype(str)+'.parquet'
df['spec2_path'] = f'{SPEC_DIR}/train_spectrograms/'+df['spectrogram_id'].astype(str)+'.npy'
df['class_name'] = df.expert_consensus.copy()
df['class_label'] = df.expert_consensus.map(Config.name2label)
display(df.head(2))
# Test
test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
test_df['eeg_path'] = f'{BASE_PATH}/test_eegs/'+test_df['eeg_id'].astype(str)+'.parquet'
test_df['spec_path'] = f'{BASE_PATH}/test_spectrograms/'+test_df['spectrogram_id'].astype(str)+'.parquet'
test_df['spec2_path'] = f'{SPEC_DIR}/test_spectrograms/'+test_df['spectrogram_id'].astype(str)+'.npy'
display(test_df.head(2))
# %% [markdown]
# ## Convert `.parquet` to `.npy` 
# To facilitate easier data loading, we will convert the EEG spectrograms from `parquet` to `npy` format. This process involves saving the spectrogram data, and since the content of the files remains the same, no significant changes are made.

# Define a function to process a single eeg_id
def process_spec(spec_id, split="train"):
    spec_path = f"{BASE_PATH}/{split}_spectrograms/{spec_id}.parquet"
    spec = pd.read_parquet(spec_path)
    spec = spec.fillna(0).values[:, 1:].T # fill NaN values with 0, transpose for (Time, Freq) -> (Freq, Time)
    spec = spec.astype("float32")
    np.save(f"{SPEC_DIR}/{split}_spectrograms/{spec_id}.npy", spec)
# Get unique spec_ids of train and valid data
spec_ids = df["spectrogram_id"].unique()
# Parallelize the processing using joblib for training data
_ = joblib.Parallel(n_jobs=-1, backend="loky")(
    joblib.delayed(process_spec)(spec_id, "train")
    for spec_id in tqdm(spec_ids, total=len(spec_ids))
)
# Get unique spec_ids of test data
test_spec_ids = test_df["spectrogram_id"].unique()
# Parallelize the processing using joblib for test data
_ = joblib.Parallel(n_jobs=-1, backend="loky")(
    joblib.delayed(process_spec)(spec_id, "test")
    for spec_id in tqdm(test_spec_ids, total=len(test_spec_ids))
)
# %% [markdown]
# ## DataLoader 
# This DataLoader first reads `npy` spectrogram files and extracts labeled subsamples using specified `offset` values. Then, it converts the spectrogram data into `log spectrogram` and applies the popular signal augmentation `MixUp`.
# Note: we are converting the mono channel signal to a 3-channel signal for using "ImageNet" weights of pretrained model.
# %%
def build_augmenter(dim=Config.imageSize):
    augmenters = [
        keras_cv.layers.MixUp(alpha=2.0),
        keras_cv.layers.RandomCutout(height_factor=(1.0, 1.0),
                                     width_factor=(0.06, 0.1)), # freq-masking
        keras_cv.layers.RandomCutout(height_factor=(0.06, 0.1),
                                     width_factor=(1.0, 1.0)), # time-masking
    ]
    def augment(img, label):
        data = {"images":img, "labels":label}
        for augmenter in augmenters:
            if tf.random.uniform([]) < 0.5:
                data = augmenter(data, training=True)
        return data["images"], data["labels"]
    return augment
def build_decoder(with_labels=True, target_size=Config.imageSize, dtype=32):
    def decode_signal(path, offset=None):
        # Read .npy files and process the signal
        file_bytes = tf.io.read_file(path)
        sig = tf.io.decode_raw(file_bytes, tf.float32)
        sig = sig[1024//dtype:]  # Remove header tag
        sig = tf.reshape(sig, [400, -1])

        # Extract labeled subsample from full spectrogram using "offset"
        if offset is not None:
            offset = offset // 2  # Only odd values are given
            sig = sig[:, offset:offset+300]

            # Pad spectrogram to ensure the same input shape of [400, 300]
            pad_size = tf.math.maximum(0, 300 - tf.shape(sig)[1])
            sig = tf.pad(sig, [[0, 0], [0, pad_size]])
            sig = tf.reshape(sig, [400, 300])

        # Log spectrogram
        sig = tf.clip_by_value(sig, tf.math.exp(-4.0), tf.math.exp(8.0)) # avoid 0 in log
        sig = tf.math.log(sig)

        # Normalize spectrogram
        sig -= tf.math.reduce_mean(sig)
        sig /= tf.math.reduce_std(sig) + 1e-6

        # Mono channel to 3 channels to use "ImageNet" weights
        sig = tf.tile(sig[..., None], [1, 1, 3])
        return sig

    def decode_label(label):
        label = tf.one_hot(label, Config.numClasses)
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [Config.numClasses])
        return label

    def decode_with_labels(path, offset=None, label=None):
        sig = decode_signal(path, offset)
        label = decode_label(label)
        return (sig, label)

    return decode_with_labels if with_labels else decode_signal


def build_dataset(paths, offsets=None, labels=None, batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=False, repeat=True, shuffle=1024,
                  cache_dir="", drop_remainder=False):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter()

    AUTO = tf.data.experimental.AUTOTUNE
    slices = (paths, offsets) if labels is None else (paths, offsets, labels)

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    if shuffle:
        ds = ds.shuffle(shuffle, seed=Config.randSeed)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds

# %% [markdown]
# ## Data Split
# 
# In the following code snippet, the data is divided into `5` folds. Note that, the `groups` argument is used to prevent any overlap of patients between the training and validation sets, thus avoiding potential **data leakage** issues. Additionally, each split is stratified based on the `class_label`, ensuring a uniform distribution of class labels in each fold.

# %%
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=Config.randSeed)

df["fold"] = -1
df.reset_index(drop=True, inplace=True)
for fold, (train_idx, valid_idx) in enumerate(
    sgkf.split(df, y=df["class_label"], groups=df["patient_id"])
):
    df.loc[valid_idx, "fold"] = fold
df.groupby(["fold", "class_name"])[["eeg_id"]].count().T

# %% [markdown]
# ## Build Train & Valid Dataset
# Only first sample for each `spectrogram_id` is used in order to keep the dataset size managable. Feel free to train on full data.

# %%
# Sample from full data
sample_df = df.groupby("spectrogram_id").head(1).reset_index(drop=True)

train_df = sample_df[sample_df.fold != Config.fold]
valid_df = sample_df[sample_df.fold == Config.fold]
print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

# Train
train_paths = train_df.spec2_path.values
train_offsets = train_df.spectrogram_label_offset_seconds.values.astype(int)
train_labels = train_df.class_label.values
train_ds = build_dataset(train_paths, train_offsets, train_labels, batch_size=Config.batchSize,
                         repeat=True, shuffle=True, augment=True, cache=True)

# Valid
valid_paths = valid_df.spec2_path.values
valid_offsets = valid_df.spectrogram_label_offset_seconds.values.astype(int)
valid_labels = valid_df.class_label.values
valid_ds = build_dataset(valid_paths, valid_offsets, valid_labels, batch_size=Config.batchSize,
                         repeat=False, shuffle=False, augment=False, cache=True)

# %% [markdown]
# ## Dataset Check
# 
# Let's visualize some samples from the dataset.

# %%
imgs, tars = next(iter(train_ds))

num_imgs = 8
plt.figure(figsize=(4*4, num_imgs//4*5))
for i in range(num_imgs):
    plt.subplot(num_imgs//4, 4, i + 1)
    img = imgs[i].numpy()[...,0]  # Adjust as per your image data format
    img -= img.min()
    img /= img.max() + 1e-4
    tar = Config.label2name[np.argmax(tars[i].numpy())]
    plt.imshow(img)
    plt.title(f"Target: {tar}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Loss & Metric
# 
# The evaluation metric in this is **KL Divergence**, defined as,
# 
# $$
# D_{\text{KL}}(P \parallel Q) = \sum_{i} P(i) \log\left(\frac{P(i)}{Q(i)}\right)
# $$
# 
# Where:
# - $P$ is the true distribution.
# - $Q$ is the predicted distribution.
# 
# Interestingly, as KL Divergence is differentiable, we can directly use it as our loss function. Thus, we don't need to use a third-party metric like **Accuracy** to evaluate our model. Therefore, `valid_loss` can stand alone as an indicator for our evaluation. In keras, we already have impelementation for KL Divergence loss.

# %%
LOSS = keras.losses.KLDivergence()

# %% [markdown]
# ## Modeling
# 
# We use the `EfficientNetV2 B2` from KerasCV's collection of pretrained models. To change this modify the `preset` in the `Config`. <br>
# [KerasCV website](https://keras.io/api/keras_cv/models/tasks/image_classifier/)

# %%
# Build Classifier
model = keras_cv.models.ImageClassifier.from_preset(Config.preset, num_classes=Config.numClasses)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss=LOSS)

# Model Sumamry
model.summary()

# %% [markdown]
# ## LR Schedule
# 
# A well-structured learning rate schedule is essential for efficient model training, ensuring optimal convergence and avoiding issues such as overshooting or stagnation.

# %%


def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 5e-5, 6e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

# %%
lr_cb = get_lr_callback(Config.batchSize, mode=Config.lrMode, plot=True)

# %% [markdown]
# ## Model Checkpointing

# %%
ckpt_cb = keras.callbacks.ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min')

# %% [markdown]
# ## Training

# %%
history = model.fit(train_ds, epochs=Config.epochs, callbacks=[lr_cb, ckpt_cb], steps_per_epoch=len(train_df)//Config.batchSize, validation_data=valid_ds, verbose=Config.verbosity)

# %% [markdown]
# # Prediction

# %% [markdown]
# ## Load Best Model

# %%
model.load_weights("best_model.keras")

# %% [markdown]
# ## Build Test Dataset

# %%
test_paths = test_df.spec2_path.values
test_ds = build_dataset(test_paths, batch_size=min(Config.batchSize, len(test_df)),
                         repeat=False, shuffle=False, cache=False, augment=False)

# %% [markdown]
# ## Inference

# %%
preds = model.predict(test_ds)

# %% [markdown]
# ## Submission

# %%
pred_df = test_df[["eeg_id"]].copy()
target_cols = [x.lower()+'_vote' for x in Config.classNames]
pred_df[target_cols] = preds.tolist()

sub_df = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
sub_df = sub_df[["eeg_id"]].copy()
sub_df = sub_df.merge(pred_df, on="eeg_id", how="left")
sub_df.to_csv("submission.csv", index=False)
sub_df.head()


