import tensorflow as tf
import numpy as np

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocabSize, dModel):
        super().__init__()
        self.dModel = dModel
        self.embedding = tf.keras.layers.Embedding(vocabSize, dModel, mask_zero = True)
        self.posEncoding = self.positional_encoding(length = vocabSize, depth = dModel)

    def positional_encoding(self, length, depth):
        depth /= 2

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth

        angleRates = 1 / (10000**depths)
        angleRads = positions * angleRates

        posEncoding = np.concatenate([np.sin(angleRads), np.cos(angleRads)], axis=-1)

        return tf.cast(posEncoding, dtype=tf.float32)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.dModel, tf.float32))
        x = x + self.posEncoding[tf.newaxis, :length, :]

        return x


class AddAndNormalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        return self.layerNorm(tf.add_n(inputs))


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        q, k, v, mask = inputs
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)

        scores = tf.matmul(q, k, transpose_b = True) / tf.math.sqrt(d_k)

        if mask != None:
            scores = scores + (mask * -1e9)
        
        weights = tf.nn.softmax(scores, axis=-1)

        return tf.matmul(weights, v), weights


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, h, dModel, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        self.dModel = dModel
        self.keyDimension = dModel // h
        self.valueDimension = dModel // h

        # Generating the linear layers.
        self.linearQ = tf.keras.layers.Dense(dModel)
        self.linearK = tf.keras.layers.Dense(dModel)
        self.linearV = tf.keras.layers.Dense(dModel)

        self.finalLinearLayer = tf.keras.layers.Dense(dModel)

        # Also pre-preparing our SDPA instance.
        self.SDPAInstance = ScaledDotProductAttention()
    
    def split_heads(self, x, dimension):
        batch = tf.shape(x)[0]
        seqLength = tf.shape(x)[1]

        x = tf.reshape(x, (batch, seqLength, self.h, dimension))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q, k, v, mask = inputs

        # Create outputs for each linear set of linear layers.
        outputs = [
            self.split_heads(self.linearQ(q), self.keyDimension),
            self.split_heads(self.linearK(k), self.keyDimension),
            self.split_heads(self.linearV(v), self.valueDimension)
        ]

        # For each set of inputs, pass it through the scaled dot product attention.
        context, attentionWeights = self.SDPAInstance.call([outputs[0], outputs[1], outputs[2], mask])

        # Concatenate the results.
        context = tf.transpose(context, [0, 2, 1, 3])
        batch, seqLength = tf.shape(context)[0], tf.shape(context)[1]
        context = tf.reshape(context, (batch, seqLength, self.dModel))

        # Return the result of the final linear layer.
        return self.finalLinearLayer(context)


class FFN(tf.keras.layers.Layer):
    def __init__(self, dModel, insideDimension, **kwrags):
        super().__init__(**kwrags)
        self.dModel = dModel
        self.insideDimension = insideDimension

        self.layer_1 = tf.keras.layers.Dense(self.insideDimension)
        self.layer_2 = tf.keras.layers.Dense(self.dModel)
    
    def call(self, x):
        return self.layer_2(tf.keras.activations.gelu(self.layer_1(x)))


class Decoder(tf.keras.layers.Layer):
    def __init__(self, dModel, numHeads, ffnInnerSize, **kwargs):
        super().__init__(**kwargs)

        self.dModel = dModel
        self.ffnInnerSize = ffnInnerSize

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadedAttention(numHeads, dModel)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = FFN(dModel, ffnInnerSize)
    
    def call(self, inputs):
        x, mask = inputs

        norm_1 = self.norm_1(x)
        mha_1 = x + self.mha([norm_1, norm_1, norm_1, mask])

        norm_2 = self.norm_2(mha_1)
        ffn_final = mha_1 + self.ffn(norm_2)

        return ffn_final


class LLM(tf.keras.Model):
    def __init__(self, vocabSize, dModel, numHeads, ffnInnerSize, decoderCount, maxPositions = 2048, **kwrags):
        super().__init__(**kwrags)
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.numHeads = numHeads
        self.ffnInnerSize = ffnInnerSize
        self.decoderCount = decoderCount

        self.tokenEmbedding = tf.keras.layers.Embedding(vocabSize, dModel)
        self.positionalEmbedding = tf.keras.layers.Embedding(maxPositions, dModel)

        self.decoderBlocks = [Decoder(self.dModel, self.numHeads, self.ffnInnerSize) for _ in range(self.decoderCount)]

        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def generate_mask(self, sequenceLength):
        mask = tf.linalg.band_part(tf.ones((sequenceLength, sequenceLength)), -1, 0)
        mask = 1 - mask

        return tf.cast(mask[tf.newaxis, tf.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        batchSize = tf.shape(x)[0]
        sequenceLength = tf.shape(x)[1]

        # Construct initial mask and embeddings.
        mask = self.generate_mask(sequenceLength)

        tokenEmbeddings = self.tokenEmbedding(x)
        positionalEmbeddings = self.positionalEmbedding(tf.tile(tf.range(sequenceLength)[tf.newaxis, :], [batchSize, 1]))

        hiddenEmbeddings = tokenEmbeddings + positionalEmbeddings

        # Now run the embeddings through every decoder block.
        for block in self.decoderBlocks:
            hiddenEmbeddings = block([hiddenEmbeddings, mask])
        
        # Next, perform normalization, and return the output after linking them with the embedding weights.
        normal_output = self.final_norm(hiddenEmbeddings)
        logits = tf.matmul(normal_output, self.tokenEmbedding.weights[0], transpose_b = True)

        return logits