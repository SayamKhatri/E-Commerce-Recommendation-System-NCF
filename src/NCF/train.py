import tensorflow as tf
from common.config import NCF_PARAMS

def create_dataset(df, batch_size=512):
    user_ids = df['visitorid_encoded'].astype('int32').to_numpy()
    item_ids = df['itemid_encoded'].astype('int32').to_numpy()
    labels = df['interaction'].astype('float32').to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices(((user_ids, item_ids), labels))

    return dataset.cache().shuffle(buffer_size=len(df)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_ncf(model, train_samples, val_data):

    train_dataset = create_dataset(train_samples)
    val_dataset = create_dataset(val_data)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=NCF_PARAMS['learning_rate']),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NCF_PARAMS['epochs'],
        verbose=1
    )

    return history
