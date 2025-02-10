from tensorflow_model_optimization.python.core.keras.compat import keras #for Quantization Aware Training (QAT)
import tensorflow_model_optimization as tfmot #for Post Training Quantization (PTQ)
from datasets import load_dataset #for downloading the Wake Vision Dataset
import tensorflow as tf #for designing and training the model 

import tensorflow as tf
from keras.layers import *

model_name = 'wv_k_8_c_5'

#some hyperparameters 
#Play with them!
input_shape = (50,50,3)
batch_size = 512
learning_rate = 0.001
epochs = 100




def blazeFace(
    input_shape,
    use_double_block=False,
    activation="relu",
    use_optional_block=True,
    use_resblock=True,
):
    def channel_padding(x):
        return tf.keras.backend.concatenate([x, tf.zeros_like(x)], axis=-1)

    def single_block(x_input, filters, strides, activation):

        x = DepthwiseConv2D(
            kernel_size=5, strides=strides, padding="same", name=f"depthwiseconv2d_{i}_0"
        )(x_input)
        x = Conv2D(
            kernel_size=1, filters=filters, strides=[1, 1], padding="same", name=f"conv2d_{i}_1"
        )(x)
        # x = SeparableConv2D(filters=filters,kernel_size= 5 ,strides=strides,padding= 'same', use_bias=False)(x_input)
        x = BatchNormalization(name=f"batch_normalization_{i}_2")(x)

        # residual_connection
        if strides == 2 and use_resblock:
            input_channels = x_input.shape[-1]
            output_channels = x.shape[-1]
            x1 = MaxPool2D(name=f"max_pool2d_{i}_2_1")(x_input)
            if output_channels - input_channels != 0:
                x1 = Lambda(channel_padding, name=f"lambda_{i}_2_2")(x1)
            out = Add(name=f"add_{i}_2_3")([x, x1])
            output = Activation(activation, name=f"{activation}_{i}_2_4")(out)
            return output
        if use_resblock:
            out = Add(name=f"add_{i}_3")([x, x_input])
        else:
            out = x
        output = Activation(activation, name=f"{activation}_{i}_4")(out)
        return output

    def double_block(x_input, filters_1, filters_2, strides, activation="relu"):
        x = DepthwiseConv2D(
            kernel_size=5, strides=strides, padding="same", name=f"depthwiseconv2d_{i}_0"
        )(x_input)
        x = Conv2D(
            kernel_size=1,
            filters=filters_1,
            strides=[1, 1],
            padding="same",
            name=f"conv2d_{i}_1",
        )(x)
        x = BatchNormalization(name=f"batch_normalization_{i}_2")(x)
        # x = Activation(activation, name=f"{activation}_{i}_3")(x)

        x = DepthwiseConv2D(
            kernel_size=5, strides=[1, 1], padding="same", name=f"depthwiseconv2d_{i}_3"
        )(x)
        x = Conv2D(
            kernel_size=1,
            filters=filters_2,
            strides=[1, 1],
            padding="same",
            name=f"conv2d_{i}_4",
        )(x)
        x = BatchNormalization(name=f"batch_normalization_{i}_5")(x)

        # residual_connection
        if strides == 2 and use_resblock:
            input_channels = x_input.shape[-1]
            output_channels = x.shape[-1]
            x1 = MaxPool2D(name=f"max_pool2d_{i}_5_1")(x_input)

            if output_channels - input_channels != 0:
                x1 = Lambda(channel_padding, name=f"lambda_{i}_5_2")(x1)
            out = Add(name=f"add_{i}_5_3")([x, x1])
            output = Activation(activation, name=f"{activation}_{i}_5_4")(out)
            return output

        if use_resblock:
            out = Add(name=f"add_{i}_6")([x, x_input])
        else:
            out = x
        output = Activation(activation, name=f"{activation}_{i}_7")(out)
        return output

    # model = keras.Sequential(name="GenBlazeFacev2")
    # model.add(InputLayer(input_shape=input_shape, name="input"))
    i = 0
    inputs = Input(shape=input_shape)
    ### First layer
    x0 = Conv2D(filters=24, kernel_size=5, strides=(2, 2), padding="same", name=f"conv2d_{i}_0")(
        inputs
    )
    x0 = BatchNormalization(name=f"batch_normalization_{i}_1")(x0)
    x0 = Activation(activation, name=f"{activation}_{i}_2")(x0)

    ### single blocks
    i += 1
    x1 = single_block(x0, 24, 1, activation)
    i += 1
    x2 = single_block(x1, 24, 1, activation)
    i += 1
    x3 = single_block(x2, 48, 2, activation)  # (downscale stride)
    i += 1
    x4 = single_block(x3, 48, 1, activation)
    i += 1
    x5 = single_block(x4, 48, 1, activation)
    ## double blocks
    if use_double_block:
        i += 1
        x6 = double_block(x5, 24, 96, 2, activation)  # (downscale stride)
        i += 1
        x7 = double_block(x6, 24, 96, 1, activation)
        x8 = x7
        if use_optional_block:
            i += 1
            x8 = double_block(x7, 24, 96, 1, activation)  # optional
        i += 1
        x9 = double_block(x8, 24, 96, 2, activation)  # (downscale stride)
        x10 = x9
        if use_optional_block:
            i += 1
            x10 = double_block(x9, 24, 96, 1, activation)  # optional
        i += 1
        x11 = double_block(x10, 24, 96, 2, activation)  # (downscale stride)
    ### single blocks
    else:
        i += 1
        x6 = single_block(x5, 96, 2, activation)  # (downscale stride)
        i += 1
        x7 = single_block(x6, 96, 1, activation)
        x8 = x7
        if use_optional_block:
            i += 1
            x8 = single_block(x7, 96, 1, activation)  # optional
        i += 1
        x9 = single_block(x8, 96, 2, activation)  # (downscale stride)
        x10 = x9
        if use_optional_block:
            i += 1
            x10 = single_block(x9, 96, 1, activation)  # optional
        i += 1
        x11 = single_block(x10, 96, 2, activation)  # (downscale stride)

    ### head
    i += 1
    x12 = Conv2D(filters=64, kernel_size=1, strides=[1, 1], name=f"conv2d_{i}_0")(x11)
    x13 = BatchNormalization(name=f"batch_normalization_{i}_1")(x12)
    backbone_output = Activation(activation, name=f"{activation}_{i}_2")(x13)

    model = tf.keras.models.Model(inputs=inputs, outputs= backbone_output)
    return model





blazeface_backbone = blazeFace(input_shape = input_shape, use_double_block= True, activation = 'relu', use_optional_block= False, use_resblock=False)
feature = GlobalAveragePooling2D(name="feature")(blazeface_backbone.output)
output = Dense(2, activation='softmax')(feature)

model = tf.keras.Model(inputs = blazeface_backbone.input, outputs = output)

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

#load dataset
ds = load_dataset("Harvard-Edge/Wake-Vision")
    
train_ds = ds['train_quality'].to_tf_dataset(columns='image', label_cols='person')
val_ds = ds['validation'].to_tf_dataset(columns='image', label_cols='person')
test_ds = ds['test'].to_tf_dataset(columns='image', label_cols='person')

#some preprocessing 
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1])])

data_augmentation = tf.keras.Sequential([
    data_preprocessing,
    #add some data augmentation 
    #Play with it!
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)])
    
train_ds = train_ds.shuffle(1000).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(tf.data.AUTOTUNE)

#set validation based early stopping
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".tf",
    monitor='val_sparse_categorical_accuracy',
    mode='max', save_best_only=True)
    
#training
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[model_checkpoint_callback])

#Post Training Quantization (PTQ)
model = tf.keras.models.load_model(model_name + ".tf")

def representative_dataset():
    for data in train_ds.rebatch(1).take(150) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
    
#Test quantized model
interpreter = tf.lite.Interpreter(model_name + ".tflite")
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in test_ds :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       if label.numpy() == interpreter.get_tensor(output['index']).argmax() :
           correct = correct + 1
       else :
           wrong = wrong + 1
print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n\n")
