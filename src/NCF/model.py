import tensorflow as tf

class NCFModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=64, mlp_layers=[128, 64, 32]):
        super(NCFModel, self).__init__()
        self.user_embeddings_gmf = tf.keras.layers.Embedding(num_users, embedding_size, name='user_embeddings_gmf')
        self.item_embeddings_gmf = tf.keras.layers.Embedding(num_items, embedding_size, name='item_embeddings_gmf')
        self.user_embeddings_mlp = tf.keras.layers.Embedding(num_users, embedding_size, name='user_embeddings_mlp')
        self.item_embeddings_mlp = tf.keras.layers.Embedding(num_items, embedding_size, name='item_embeddings_mlp')

        self.mlp_layers = [tf.keras.layers.Dense(units, activation='relu') for units in mlp_layers]
        self.final_output = tf.keras.layers.Dense(1, activation=None, name='output')
    
    def call(self, inputs):
        user_ids, item_ids = inputs
        user_gmf = self.user_embeddings_gmf(user_ids)
        item_gmf = self.item_embeddings_gmf(item_ids)
        gmf_vector = tf.multiply(user_gmf, item_gmf)
        user_mlp = self.user_embeddings_mlp(user_ids)
        item_mlp = self.item_embeddings_mlp(item_ids)

        mlp_vector = tf.concat([user_mlp, item_mlp], axis=-1)

        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)
            
        combined_vector = tf.concat([gmf_vector, mlp_vector], axis=-1)

        return self.final_output(combined_vector)