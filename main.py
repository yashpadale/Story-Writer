import keras
from keras import layers
import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from keras.models import load_model
import pickle

def number_to_words(predictions, dictionary):
    # Invert the dictionary
    inverted_dictionary = {v: k for k, v in dictionary.items()}

    predicted_sentences = []

    for prediction_row in predictions:
        words_row = []
        for index in prediction_row:
            # Check if the index exists in the inverted dictionary
            word = inverted_dictionary.get(index)
            if word is not None:
                words_row.append(word)
        predicted_sentence = ' '.join(words_row)
        predicted_sentences.append(predicted_sentence)

    return predicted_sentences
def return_order(dict_, content: str):


    tokens = word_tokenize(content)

    order = [dict_[token] for token in tokens if token in dict_]
    return order
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # Apply mask to prevent attending to future tokens during decoding
        mask = tf.linalg.band_part(tf.ones_like(scaled_score), -1, 0)
        scaled_score -= 1e9 * (1 - mask)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)

        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=True)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=True)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config



class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.token_emb = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)
        super().build(input_shape)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.embed_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config
def build_transformer_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, dropout_rate, num_encoders, num_decoders):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)
    encoders_outputs = []
    x = embedding_layer
    for _ in range(int(num_encoders)):
        encoder_output = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
        encoders_outputs.append(encoder_output)

    decoder_inputs = layers.Input(shape=(maxlen,))
    y = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(decoder_inputs)
    for _ in range(int(num_decoders)):
        for encoder_output in encoders_outputs:
            y = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(y)
    outputs = layers.Dense(vocab_size, activation="softmax")(y)

    model = keras.Model(inputs=[inputs, decoder_inputs], outputs=outputs)
    return model

def query_gen_sentences(query, model, dictionary, maxlen):
    # Convert the query to the order of words based on the provided dictionary
    query_order = return_order(dict_=dictionary, content=query)
    u_order = np.array(query_order)

    # Pad the order to match the maximum length
    padding_length = max(0, maxlen - len(u_order))
    padded_u_order = np.pad(u_order, (0, padding_length), mode='constant', constant_values=0)
    padded_u_order = np.reshape(padded_u_order, (1, -1))

    # Generate predictions using the model
    # Assuming x_data_1 and x_data_2 are your input data tensors
    predictions = model.predict([padded_u_order, padded_u_order])
    predicted_classes = np.argmax(predictions, axis=-1)

    # Convert predicted classes to words using the provided dictionary
    words = number_to_words(predictions=predicted_classes, dictionary=dictionary)

    return words


custom_objects = {
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
    'TransformerBlock': TransformerBlock,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding
}
loaded_model = load_model('transformer_model.h5', custom_objects=custom_objects)
with open('dictionary.pkl', 'rb') as f:
    loaded_dictionary = pickle.load(f)

maxlen=15
def generate_text(s1,tokens:int):
    respose=''
    words = query_gen_sentences(query=s1,
                                model=loaded_model, dictionary=loaded_dictionary, maxlen=maxlen)
    w_=words[0].split(' ')
    respose+=' '+w_[-1]+' '
    for i in range(tokens):
        w1 = query_gen_sentences(query=words[-1]
                                 ,
                                 model=loaded_model, dictionary=loaded_dictionary, maxlen=maxlen)
        words.append(w1[0])
        w_ = w1[0].split(' ')
        respose += ' '+w_[-1]+' '

    return respose

def gen(input,num_of_token):
    o=generate_text(s1=input,tokens=num_of_token)
    return o


while True:
    i=input("Enter : ")
    # remember the number of words in ur prompt should be less that 15
    o=generate_text(s1=i,tokens=25)
    print(o)


# Sample prompt- Harry walked through the dark corridors of Hogwarts, his wand lit with a faint