import tensorflow as tf
from tensorflow import keras
import numpy as np


def kl_gauss(posterior_means, prior_means, posterior_log_var, prior_log_var):   
    kl = prior_log_var - posterior_log_var + (tf.exp(posterior_log_var) + 
                                       tf.square(posterior_means - prior_means)) / tf.exp(prior_log_var) - 1
    kl = 0.5 * tf.reduce_sum(kl, axis=[1,2])
    return kl

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)



class VRNNCell(tf.keras.layers.GRUCell):
    # Implementing Variational RNN's and variations by subclassing Keras RNN-type Cells
    def __init__(self, units, output_dim, **kwargs):
        super(VRNNCell, self).__init__(units, **kwargs)
        self.output_dim = output_dim
    

    def build(self, input_shape):
        # Taking most of the standard weight initiaalizations from the base GRU class
        super().build((input_shape[0], input_shape[1] + self.units))

        self.batch_size = input_shape[0]
        
        self.input_kernel = self.add_weight(shape=(input_shape[-1], input_shape[-1]), name="layer", initializer='truncated_normal')
        
        self.prior_kernel = self.add_weight(shape=(self.units, self.units),name="layer", initializer='truncated_normal')
        
        self.pos_kernel = self.add_weight(shape=(input_shape[-1] + self.units,input_shape[-1] + self.units), name="layer", initializer='truncated_normal')
        
        self.encoder_mu_kernel = self.add_weight(shape=(input_shape[-1] + self.units, self.units), name="layer", initializer='truncated_normal')
        
        self.encoder_logvar_kernel = self.add_weight(shape=(input_shape[-1] + self.units, self.units), name="layer", initializer='truncated_normal')
        
        self.prior_mu_kernel = self.add_weight(shape=(self.units, self.units), name="layer", initializer='truncated_normal')
        
        self.prior_logvar_kernel = self.add_weight(shape=(self.units, self.units), name="layer", initializer='truncated_normal')  
        
        self.z_kernel = self.add_weight(shape=(self.units, self.units), name="layer", initializer='truncated_normal')
        
        self.output_mean_kernel = self.add_weight(shape=(self.units + self.units, self.output_dim), name="layer", initializer='truncated_normal')

        self.output_logvar_kernel =self.add_weight(shape=(self.units + self.units, self.output_dim), name="layer", initializer='truncated_normal')


    def sample(self, mu, log_var):
        # Sample from unit Normal
        dims = tf.shape(mu)
        epsilon = tf.random.normal(dims)
        # All element-wise computations
        std = tf.math.exp(0.5 * log_var)
        z = tf.math.multiply(std, epsilon) + mu
        return z
    
    def call(self, inputs, states=None, training=True):
        # Some formulations:
        # Generation:
        # z_t ~ N(mu_(0, t), sigma_(0,t)), w here [mu_(0,t), sigma(0,t)] = phi_prior(h_(t-1))
        # Update: 
        # h_t = f_theta(h_(t-1), z_t, x_t) *recurrence equation
        # Inference:
        # z_t ~ N(mu_z, sigma_z), where [mu_z, sigma_z] = phi_post(x_t, h_(t-1))
        #
        # Let the base RNN cell handle the rest and add loss
        
        if training:
            print("TRAINING")
            x_t = tf.nn.relu(tf.matmul(inputs, self.input_kernel))
            h_prev = states[0]

            prior = tf.matmul(h_prev, self.prior_kernel)
            p_mu = tf.matmul(prior, self.prior_mu_kernel)
            p_logvar = tf.matmul(prior, self.prior_logvar_kernel)
            
            input_state_concat = tf.concat([x_t, h_prev], axis=1)
            
            pos = tf.matmul(input_state_concat, self.pos_kernel)
            q_mu = tf.matmul(pos, self.encoder_mu_kernel)
            q_logvar = tf.matmul(pos, self.encoder_logvar_kernel)
            z_t = self.sample(q_mu, q_logvar)
            phi_z_t = tf.nn.relu(tf.matmul(z_t, self.z_kernel))
            
            inp = tf.concat([x_t, phi_z_t], axis=1)
            _, h_next = super().call(inp, h_prev)
            
            output_mean = tf.matmul(tf.concat([h_prev, phi_z_t], axis=1), self.output_mean_kernel)
            output_log_var = tf.matmul(tf.concat([h_prev, phi_z_t], axis=1), self.output_logvar_kernel)

            output = self.sample(output_mean, output_log_var)

            
            all_output = (output, z_t, q_mu, p_mu, q_logvar, p_logvar, output_mean, output_log_var)
            return all_output, h_next
        
        else:
            print('GENERATION')
            x_t = tf.nn.relu(tf.matmul(inputs, self.input_kernel))
            if states is None:
                h_prev = super().get_initial_state(x_t)
            else:
                h_prev = states[0]
            prior = tf.matmul(h_prev, self.prior_kernel)
            p_mu = tf.matmul(prior, self.prior_mu_kernel)
            p_logvar = tf.matmul(prior, self.prior_logvar_kernel)
            
            input_state_concat = tf.concat([x_t, h_prev], axis=1)
            
            pos = tf.matmul(input_state_concat, self.pos_kernel)
            q_mu = tf.matmul(pos, self.encoder_mu_kernel)
            q_logvar = tf.matmul(pos, self.encoder_logvar_kernel)
            z_t = self.sample(p_mu, p_logvar)
            phi_z_t = tf.nn.relu(tf.matmul(z_t, self.z_kernel))
            
            inp = tf.concat([x_t, phi_z_t], axis=1)
            _, h_next = super().call(inp, h_prev)
            
            output_mean = tf.matmul(tf.concat([h_prev, phi_z_t], axis=1), self.output_mean_kernel)
            output_log_var = tf.matmul(tf.concat([h_prev, phi_z_t], axis=1), self.output_logvar_kernel)

            output = self.sample(output_mean, output_log_var)
            
            all_output = (output, z_t, q_mu, p_mu, q_logvar, p_logvar, output_mean, output_log_var)
            return all_output, h_next
    
   
    def get_config(self):
        return {"units":self.units}

class VRNNGRU(tf.keras.Model):
    def __init__(self, feature_space, latent_dim, timesteps, **kwargs):
        super(VRNNGRU, self).__init__(**kwargs)
        vrnn_cell = VRNNCell(latent_dim, feature_space)
        self.latent_dim = latent_dim
        
        vrnn_input = keras.layers.Input(shape=(timesteps, feature_space))
        vrnn_output = keras.layers.RNN(vrnn_cell, return_sequences=True)(vrnn_input)
        self.vrnn = keras.Model(vrnn_input, vrnn_output)
        
        decoder_input = keras.layers.Input(shape=(timesteps, latent_dim))
        output =  keras.layers.TimeDistributed(keras.layers.Dense(feature_space, activation='softmax'))(decoder_input)
        self.decoder =keras.Model(decoder_input, output)
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            input_data = data[0]
            output_data = data[1]
            
            timesteps = input_data.shape[1]
            num_feats = input_data.shape[2]
        with tf.GradientTape() as tape:
            outputs = self.vrnn(input_data, training=True)
            preds = outputs[0]
#             preds = self.decoder(z)
            
            q_mu = outputs[2]
            p_mu = outputs[3]
            q_log_var = outputs[4]
            p_log_var = outputs[5]
            
            kl_loss = tf.reduce_mean(kl_gauss(q_mu, p_mu, q_log_var, p_log_var))
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.mean_squared_error(output_data, preds), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss 
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)       
        return {
            'total_loss': self.total_loss_tracker.result(),
            'loss': self.reconstruction_loss_tracker.result(),
            'kl': self.kl_loss_tracker.result()
        }

    def call(self, inputs):
        outputs = self.vrnn(inputs, training=True)
        return outputs
    
    def generate(self, inputs):
        outputs = self.vrnn(inputs, training=True)
        return outputs


class kCallback(tf.keras.callbacks.Callback):
    def __init__(self, count, limit):
        self.limit = limit
        self.count = count
    def on_train_batch_end(self, batch, logs={}):
        if self.count == self.limit:
            self.count = 0
        else:
            self.count += 1
        print('k is currently {}'.format(self.count))


class VRNNGRUGAN(tf.keras.Model):
    def __init__(self, feature_space, latent_dim, timesteps, **kwargs):
        super(VRNNGRUGAN, self).__init__(**kwargs)
        self.vrnn_cell = VRNNCell(latent_dim, feature_space)
        self.feature_space = feature_space
        self.latent_dim = latent_dim
        
        vrnn_input = keras.layers.Input(shape=(timesteps, feature_space))
        vrnn_output = keras.layers.RNN(self.vrnn_cell, return_sequences=True)(vrnn_input)
        self.vrnn = keras.Model(vrnn_input, vrnn_output)
                
        disc_input = keras.layers.Input(shape=(timesteps, feature_space))
        disc_rnn = keras.layers.GRU(16, recurrent_dropout=0.5, dropout=0.5, name='feature_ext')(disc_input)
        # disc_rnn = keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu')(disc_input)
        # disc_rnn = keras.layers.MaxPooling1D(pool_size=8, strides=1, padding='same')(disc_rnn)
        # disc_rnn = keras.layers.Flatten(name='feature_ext')(disc_rnn)
        disc_output = keras.layers.Dense(1, activation='sigmoid')(disc_rnn)
        disc_output = keras.layers.Dropout(0.4)(disc_output)
        disc_model = keras.Model(disc_input, disc_output)
        self.discrim = disc_model
        
        # self.f = keras.Model(disc_input, self.discrim.get_layer('feature_ext').output)
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.next_step_loss_tracker = keras.metrics.Mean(
            name="next_step_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.discrim_loss_tracker = keras.metrics.Mean(name="discrim_loss")
        self.discrim_fake_loss_tracker = keras.metrics.Mean(name="discrim_fake_loss")
        self.discrim_real_loss_tracker = keras.metrics.Mean(name="discrim_real_loss")
        self.misled_loss_tracker = keras.metrics.Mean(name="misled_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.discrim_loss_tracker,
            self.discrim_fake_loss_tracker,
            self.discrim_real_loss_tracker,
            self.misled_loss_tracker
        ]
    def compile(self, vae_optimizer, discrim_optimizer):
        super(VRNNGRUGAN, self).compile()
        self.vae_optimizer = vae_optimizer
        self.discrim_optimizer = discrim_optimizer

    def train_step(self, data):
        if isinstance(data, tuple):
            input_data = data[0]
            output_data = data[1]
            
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.vrnn(input_data)
            preds = outputs[0]
            
#             cut_preds = tf.slice(preds, [0,19,0], [-1,-1,-1])
#             cut_inputs = tf.slice(input_data, [0,1,0], [-1,-1,-1])
#             fake = tf.concat([cut_inputs,cut_preds], axis=1)

            discrim_fake_output = self.discrim(preds)
            discrim_real_output = self.discrim(output_data)
            
            discrim_output_loss_fake = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.zeros_like(discrim_fake_output), discrim_fake_output)
            )
        
            discrim_output_loss_real = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.ones_like(discrim_real_output), discrim_real_output)
            )
              
            discrim_loss = 0.5 * (discrim_output_loss_fake + discrim_output_loss_real)
        discrim_grads = tape.gradient(discrim_loss, self.discrim.trainable_weights)
        self.discrim_optimizer.apply_gradients(zip(discrim_grads, self.discrim.trainable_weights))
        self.discrim_loss_tracker.update_state(discrim_loss)
        self.discrim_fake_loss_tracker.update_state(discrim_output_loss_fake)
        self.discrim_real_loss_tracker.update_state(discrim_output_loss_real)
        del tape
        
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.vrnn(input_data)
            preds = outputs[0]
            discrim_fake_output = self.discrim(preds)
            
            # real_embed = self.f(output_data)
            # fake_embed = self.f(preds)

            q_mu = outputs[2]
            p_mu = outputs[3]
            q_log_var = outputs[4]
            p_log_var = outputs[5]


            kl_loss = tf.reduce_mean(kl_gauss(q_mu, p_mu, q_log_var, p_log_var))
            bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_mean(tf.keras.losses.mean_squared_error(output_data, preds), axis=1)
            )
            
            mislead_output_discrim_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.ones_like(discrim_fake_output), discrim_fake_output) 
            )
            
            # mislead_output_discrim_loss = tf.reduce_mean(
            #     tf.keras.losses.mean_squared_error(real_embed, fake_embed)
            # )
            
            
            discrim_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.zeros_like(discrim_fake_output), discrim_fake_output)
            )
            total_loss =  reconstruction_loss + 0.1 * kl_loss + mislead_output_discrim_loss


        encoder_grads = tape.gradient(total_loss, self.vrnn.trainable_weights)
        encoder_grads,_ = tf.clip_by_global_norm(encoder_grads, 3)
        
        self.vae_optimizer.apply_gradients(zip(encoder_grads, self.vrnn.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.misled_loss_tracker.update_state(mislead_output_discrim_loss)
        del tape
            
        return {
            'total_loss': self.total_loss_tracker.result(),
            'loss': self.reconstruction_loss_tracker.result(),
            'kl': self.kl_loss_tracker.result(),
            'discrim_loss':self.discrim_loss_tracker.result(),
            'discrim_loss_fake':self.discrim_fake_loss_tracker.result(),
            'discrim_loss_real':self.discrim_real_loss_tracker.result(),
            'misled_loss':self.misled_loss_tracker.result(),
        }

    def call(self, inputs):
        outputs = self.vrnn(inputs)
        return outputs
    

    def generate(self, inputs):
        outputs = self.vrnn(inputs, training=True)
        return outputs
    
    def discrminator_score(self, inputs):
        score = self.discrim(inputs)
        return score

    def rec_gen(self, data, length):
        state=None 
        generated = []
        # seed state
        for i in range(data.shape[0]):
            reshaped_data = np.asarray([[data[i]]])
            outputs, state = self.vrnn_cell(reshaped_data, states=state, training=False)
            state = [state]
        gen_data = outputs[0]
        for i in range(length):
            outputs, state = self.vrnn_cell(gen_data, states=state, training=False)
            state = [state]
            preds = outputs[0].numpy()
            generated.append(preds)
            gen_data = preds

        return generated


    
    def test_step(self, data):
        inputs = data[0]
        outputs = data[1]
        preds = self(inputs, training=False)
        recon_loss = tf.keras.losses.categorical_crossentropy(outputs, preds)
        return {
            "loss": recon_loss
        }


        