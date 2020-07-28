import tensorflow as tf
import tensorflow_io as tfio


############################################################
#  Model functions
############################################################

def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, sequence length)
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)
  
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
      super(PositionalEncoding, self).__init__()
      self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
      angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
      return position * angles

    def positional_encoding(self, position, d_model):
      angle_rads = (self.get_angles(
          position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
          i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
          d_model=d_model)).numpy()
      
      # apply sin to even indices in the array; 2i
      angle_rads[:, 0::2] = tf.math.sin(angle_rads[:, 0::2])
      
      # apply cos to odd indices in the array; 2i+1
      angle_rads[:, 1::2] = tf.math.cos(angle_rads[:, 1::2])
      pos_encoding = angle_rads[tf.newaxis, ...]

      return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
      return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
  
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': attention,
          'key': attention,
          'value': attention,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = attention + inputs

  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(outputs)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = outputs + attention

  return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(maximum_position_encoding,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,d_model,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = inputs
      
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(maximum_position_encoding, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)
      
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': attention1,
          'key': attention1,
          'value': attention1,
          'mask': look_ahead_mask
      })
  attention1 = tf.keras.layers.Dropout(rate=dropout)(attention1)
  attention1 = attention1 + inputs

  attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1)
  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention2,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = attention2 + attention1

  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2)
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(outputs)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = outputs + attention2

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)
  
def decoder(vocab_size,
            maximum_position_encoding,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(maximum_position_encoding, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)
      
def transformer(vocab_size,
              maximum_position_encoding,
              num_layers_enc,
              num_layers_dec,
              units,
              d_spec,
              d_model,
              num_heads,
              dropout,
              cnn,
              name="transformer"):
    inputs = tf.keras.Input(shape=(None,d_spec), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    if cnn:
        x = tf.expand_dims(inputs, axis=-1)

        #block 1
        x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

        #block2
        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

        x = tf.keras.layers.Reshape((-1,(d_spec//4)*64))(x)
        inputs_strided = tf.keras.layers.MaxPool1D(pool_size=4, strides=4)(inputs)
    else:
        x = inputs
        inputs_strided = inputs
        
    x = tf.keras.layers.Dense(d_model)(x)

    inputs_masks = tf.dtypes.cast(
        #Like our input has a dimension of length X d_model but the masking is applied to a vector
        # We get the sum for each row and result is a vector. So, if result is 0 it is because in that position was masked
        tf.math.reduce_sum(
        inputs_strided,
        axis=2,
        keepdims=False,
        name=None
    ), tf.int32)

    #creating padding mask
    enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),name='enc_padding_mask')(inputs_masks)

    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask,output_shape=(1, None, None),name='look_ahead_mask')(dec_inputs)

    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),name='dec_padding_mask')(inputs_masks)

    enc_outputs = encoder(
        maximum_position_encoding=maximum_position_encoding,
        num_layers=num_layers_enc,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[x, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        maximum_position_encoding=maximum_position_encoding,
        num_layers=num_layers_dec,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
    
    
def get_model(config, checkpoints):
    model = transformer(
        vocab_size=config.VOCAB_SIZE,
        maximum_position_encoding=config.MAX_POSITION_ENCODING,
        num_layers_enc=config.N_ENC,
        num_layers_dec=config.N_DEC,
        units=config.UNITS,
        d_spec=config.D_SPEC,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
        cnn = config.CNN)
    
    model.load_weights(checkpoints)
    
    return model
    
    
############################################################
#  Decoding functions
############################################################


def gready_search_decoding(input_encoder, model, tokenizer, config, verbose=0):

    input_encoder = tf.expand_dims(input_encoder, axis=0)
    input_decoder = tf.expand_dims(config.START_TOKEN, axis=0)

    for i in range(config.MAX_LENGTH):
        if verbose:
          if config.ENCODING == 'subword':
              sys.stdout.write("\r{}".format( tokenizer.decode([j for j in input_decoder.numpy()[0] if j < config.VOCAB_SIZE-2]) ))
          else:
              sys.stdout.write("\r{}".format( tokenizer.sequences_to_texts([[j for j in input_decoder.numpy()[0] if j < config.VOCAB_SIZE-2]]) ))

        predictions = model(inputs=[input_encoder, input_decoder], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, config.END_TOKEN):
            break

        input_decoder = tf.concat([input_decoder, predicted_id], axis=-1)

    if verbose:
        print()

    if config.ENCODING == 'subword':
        return tokenizer.decode([i for i in tf.squeeze(input_decoder, axis=0).numpy() if i < config.VOCAB_SIZE-2])
    else:
        return tokenizer.sequences_to_texts([[i for i in tf.squeeze(input_decoder, axis=0).numpy() if i < config.VOCAB_SIZE-2]])[0][::2]


def beam_search_decoding(input_encoder, model, tokenizer, config, verbose=0):

    input_encoder = tf.expand_dims(input_encoder, axis=0)
    input_decoder = tf.expand_dims(config.START_TOKEN, axis=0)

    k_scores = [0.0]

    for i in range(config.MAX_LENGTH):
        if verbose:
            print('\nStep', i)
            if config.ENCODING == 'subword':
                for k in range(input_decoder.shape[0]):
                    print( tokenizer.decode([j for j in input_decoder.numpy()[k] if j < config.VOCAB_SIZE-2]) )
            else:
                for k in range(input_decoder.shape[0]):
                    print( tokenizer.sequences_to_texts([[j for j in input_decoder.numpy()[k] if j < config.VOCAB_SIZE-2]]) )
        predictions = model(inputs=[input_encoder, input_decoder], training=False)
        predictions = predictions[:, -1:, :]
        values, indices = tf.math.top_k(tf.math.log(tf.nn.softmax(predictions)), config.BEAM_SIZE)

        sequences = []
        scores = []

        for k in range(input_decoder.shape[0]):
            for b in range(config.BEAM_SIZE):
                sequences.append(tf.concat([input_decoder[k], [indices[k,0,b]]], axis=0))
                if i>=config.MAX_REP and len(tf.unique(sequences[-1][-config.MAX_REP:])[0])==1:
                    scores.append(k_scores[k] - float('inf'))
                else:
                    scores.append(k_scores[k] + values[k,0,b])

        values, indices = tf.math.top_k(scores, config.BEAM_SIZE)
          
        k_scores = []
        input_decoder = []
        for k in range(config.BEAM_SIZE):
            k_scores.append(values[k])
            input_decoder.append(sequences[indices[k]])
        input_decoder = tf.stack(input_decoder)

        if input_encoder.shape[0] == 1:
            input_encoder = tf.repeat(input_encoder, config.BEAM_SIZE, axis=0)

        if tf.equal(input_decoder[0,-1], config.END_TOKEN):
            break

    if verbose:
        print()

    if config.ENCODING == 'subword':
        return tokenizer.decode([i for i in input_decoder[0].numpy() if i < config.VOCAB_SIZE-2])
    else:
        return tokenizer.sequences_to_texts([[i for i in input_decoder[0].numpy() if i < config.VOCAB_SIZE-2]])[0][::2]




############################################################
#  Preprocessing functions
############################################################

def audio_to_spec(audio, sr):
    spec = tfio.experimental.audio.spectrogram(tf.cast(audio, tf.float32), nfft=512, window=400, stride=160)
    spec = tfio.experimental.audio.melscale(spec, rate=sr, mels=80, fmin=0, fmax=8000)
    spec = tf.math.log(spec+1e-9)
    spec = (spec - tf.math.reduce_mean(spec))/tf.math.reduce_std(spec)
    return spec
