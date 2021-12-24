import tensorflow as tf
from tensorflow import keras
import numpy as np
import math 
import arviz as az
import tensorflow_addons as tfa

class CRNNGAN(tf.keras.model):
	def __init__(self, feature_space, timesteps, rnn_model, **kwargs):
		self.rnn = rnn_model
		self.feature_space = feature_space

		disc_input = keras.layers.Input(shape=(timesteps, self.feature_space))
		disc_rnn = keras.layers.Bidirectional(keras.layers.GRU(64, recurrent_dropout=0.5, dropout=0.5, activity_regularizer='l2'), merge_mode='ave', name='feature_ext')(disc_input)
		disc_output = keras.layers.Dense(1)(disc_rnn)
		disc_model = keras.Model(disc_input, disc_output)
		self.discrim = disc_model

		self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
		self.discrim_loss_tracker = keras.metrics.Mean(name="discrim_loss")

	@property
	def metrics(self):
		return [
			self.gen_loss_tracker,
			self.discrim_loss_tracker
		]

	def compile(self, gen_optimizer, discrim_optimizer):
		super(CRNNGAN, self).compile()
		self.gen_optimizer = gen_optimizer
		self.discrim_optimizer = discrim_optimizer

	def train_step(self, data):
		if isinstance(data, tuple):
			input_data = data[0]
			output_data = data[1]

		with tf.GradientTape(persistent=True) as tape:
			gens = []
			state=None
			for i in range(timesteps):
				x = np.random.uniform(size=3)
				output, state = self.rnn_model(x, initial_state = state, training=True)
				state = [state]
				gens.append(output)
			gens = tf.stack(gens, axis=1)
			
			discrim_output = self.discrim(gens)

			bce = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1, from_logits=True)
			gen_loss = tf.reduce_mean(
				bce(tf.ones_like(discrim_output), disc_output)
			)
		generator_grads = tape.gradient(gen_loss, self.rnn.trainable_weights)
		self.gen_optimizer.apply_gradients(zip(generator_grads, self.rnn.trainable_weights))

        self.gen_loss_tracker.update_state(gen_loss)
        del tape

        with tf.GradientTape(persistent=True) as tape:
        	gens = []
        	for i in range(timesteps):
				x = np.random.uniform(size=3)
				output, state = self.rnn_model(x, initial_state = state, training=True)
				state = [state]
				gens.append(output)
			gens = tf.stack(gens, axis=1)

			discrim_fake = self.discrim(gens)
			discrim_real = self.discrim(output_data)

			bce = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1, from_logits=True)

			fake_loss = tf.reduce_mean(
				bce(tf.zeros_like(discrim_fake), discrim_fake)
			)

			real_loss = tf.reduce_mean(
				bce(tf.ones_like(discrim_real), discrim_real)
			)

			discrim_loss = fake_loss + real_loss

		discrim_grads = tape.gradient(discrim_loss, self.discrim.trainable_weights)
		self.discrim_optimizer.apply_gradients(zip(discrim_grads, self.discrim.trainable_weights))
		self.discrim_loss_tracker.update_state(discrim_loss)
		del tape 

		return {
			'gen_loss': self.gen_loss_tracker.results(),
			'discrim_loss': self.discrim_loss_tracker.results()
		}



