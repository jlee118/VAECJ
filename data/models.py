import tensorflow as tf
import numpy as np
# Implementing Variational RNN's and variations by subclassing Keras RNN-type Cells

class VRNNCell(tf.keras.layers.GRUCell):
    def __init__(self, units, **kwargs):
        super(VRNNCell, self).__init__(units, **kwargs)
    

    def build(self, input_shape):
        # Taking most of the standard weight initiaalizations from the base GRU class
        super().build((input_shape[0], input_shape[1] + self.units))
        
        self.input_kernel = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='uniform')
        
        self.state_kernel = self.add_weight(shape=(self.units, self.units), initializer='uniform')
        
        self.encoder_mu_kernel = self.add_weight(shape=(input_shape[-1] + self.units, self.units), initializer='uniform')
        
        self.encoder_logvar_kernel = self.add_weight(shape=(input_shape[-1] + self.units, self.units), initializer='uniform')
        
        self.prior_mu_kernel = self.add_weight(shape=(self.units, self.units), initializer='uniform')
        
        self.prior_logvar_kernel = self.add_weight(shape=(self.units, self.units), initializer='uniform')  


    def sample(self, mu, log_var):
        # Sample from unit Normal
        epsilon = tf.random.normal([1, self.units])
        half_constant = tf.convert_to_tensor(np.full((1, self.units), 0.5).astype('float32'))
        # All element-wise computations
        z = tf.math.multiply(half_constant, tf.math.exp(log_var)) + mu
        return z
    
    def call(self, inputs, states, training=False):
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
            x_t = tf.matmul(inputs, self.input_kernel)
            h_prev = tf.matmul(states[0], self.state_kernel)

            p_mu = tf.matmul(h_prev, self.prior_mu_kernel)
            p_logvar = tf.matmul(h_prev, self.prior_logvar_kernel)
            
            input_state_concat = tf.concat([x_t, h_prev], axis=1)
            
            q_mu = tf.matmul(input_state_concat, self.encoder_mu_kernel)
            q_logvar = tf.matmul(input_state_concat, self.encoder_logvar_kernel)
            z_t = self.sample(q_mu, q_logvar)
            
            inp = tf.concat([x_t, z_t], axis=1)
            _, h_next = super().call(inp, h_prev)
            
            output = (z_t, q_mu, p_mu, q_logvar, p_logvar)
            # self.add_loss(self.kl_gauss(q_mu, p_mu, q_logvar, p_logvar))
            return output, h_next
        
        else:
            # Return prior and posterior parameters
            x_t = inputs
            h_prev = states[0]

            p_mu = tf.matmul(h_prev, self.prior_mu_kernel)
            p_logvar = tf.matmul(h_prev, self.prior_logvar_kernel)
            z_t = self.sample(p_mu, p_logvar)
            
            input_state_concat = tf.concat([x_t, h_prev], axis=1)
            
            q_mu = tf.matmul(input_state_concat, self.encoder_mu_kernel)
            q_logvar = tf.matmul(input_state_concat, self.encoder_logvar_kernel)
            
            
            i = tf.concat([x_t, z_t], axis=1)
            _, h_next = super().call(i, h_prev)
            
            output = (z_t, q_mu, p_mu, q_logvar, p_logvar)
            
            return z_t, h_next
    
   
    def get_config(self):
        return {"units":self.units}


def kl_gauss(posterior_means, prior_means, posterior_log_var, prior_log_var):   
    kl = prior_log_var - posterior_log_var + (tf.exp(posterior_log_var) + 
                                       tf.square(posterior_means - prior_means)) / tf.exp(prior_log_var) - 1
    kl = 0.5 * tf.reduce_sum(kl)
    return kl


class VRNNGRU(tf.keras.Model):
    def __init__(self, feature_space, latent_dim, **kwargs):
        super(VRNNGRU, self).__init__(**kwargs)
        vrnn_cell = VRNNCell(3)
        self.latent_dim = latent_dim
        self.vrnn = keras.layers.RNN(vrnn_cell, return_sequences=True)
        self.decoder = keras.layers.TimeDistributed(keras.layers.Dense(feature_space, activation='softmax'))
        
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
            z = outputs[0]
            preds = self.decoder(z)
            print(preds)
            
            q_mu = tf.squeeze(tf.squeeze(outputs[1]))
            p_mu = tf.squeeze(tf.squeeze(outputs[2]))
            q_log_var = tf.squeeze(tf.squeeze(outputs[3]))
            p_log_var = tf.squeeze(tf.squeeze(outputs[4]))
            
            kl_loss = tf.reduce_mean(kl_gauss(q_mu, p_mu, q_log_var, p_log_var))
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.categorical_crossentropy(output_data, preds), axis=1)
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

    def call(self, inputs, training=False):
        outputs = self.vrnn(inputs, training)
        return outputs
    
    def generate(self, inputs):
        outputs = self.vrnn(inputs, training=True)
        return outputs
    