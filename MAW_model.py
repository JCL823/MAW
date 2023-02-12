'''load packages'''
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from tqdm import tqdm
ds = tfp.distributions
tf.keras.backend.set_floatx('float32') # sets dtype as tf.float32
print(tf.__version__, tfp.__version__)
#%%%
'''Define the network as tf.keras.model object'''

class GMM(tf.keras.Model):
    """a GMM class for tensorflow
    
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(GMM, self).__init__()
        self.__dict__.update(kwargs)
        
        if len(self.ipt_shape) > 1:
            if self.ipt_shape[0] % 4 == 0:
                self.pad3 = 'same'
            else:
                self.pad3 = 'valid'        
    
            self.enc = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=self.ipt_shape),        
                tf.keras.layers.Conv2D(filters=self.hidden_layer_sizes[0], kernel_size=5, strides=(2, 2), padding='same', activation=self.activation, name="conv1"),                 
                tf.keras.layers.BatchNormalization(),            
                tf.keras.layers.Conv2D(filters=self.hidden_layer_sizes[1], kernel_size=5, strides=(2, 2), padding='same', activation=self.activation, name="conv2"),                                       
                tf.keras.layers.BatchNormalization(),                   
                tf.keras.layers.Conv2D(self.hidden_layer_sizes[2], kernel_size=3, strides=(2, 2), padding='same', activation=self.activation, name="conv3"),
                tf.keras.layers.BatchNormalization(),                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=4*self.hidden_layer_sizes[2], name="lastlayer") #no activation function                
                ]) 
            if self.NormalizationLayer is True:
                self.dec = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=self.hidden_layer_sizes[2]*int(self.ipt_shape[0]/4)*int(self.ipt_shape[0]/4), activation=self.activation, name="revealed"),                
                    tf.keras.layers.Reshape(target_shape=(int(self.ipt_shape[0]/4), int(self.ipt_shape[0]/4), self.hidden_layer_sizes[2])),                                     
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2DTranspose(filters=self.hidden_layer_sizes[1], kernel_size=3, strides=(2, 2), padding='same', activation=self.activation, name="deconv3"),                                       
                    tf.keras.layers.BatchNormalization(),                      
                    tf.keras.layers.Conv2DTranspose(filters=self.hidden_layer_sizes[0], kernel_size=5, strides=(2, 2), padding='same', activation=self.activation, name="deconv2"),
                    tf.keras.layers.BatchNormalization(),      
                    tf.keras.layers.Conv2DTranspose(filters=self.ipt_shape[2], kernel_size=5, strides=(1, 1), padding='same', activation=self.activation, name="deconv1"), #used to be sigmoid          
                    tf.keras.layers.LayerNormalization()
                    ])
            else:
                self.dec = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=self.hidden_layer_sizes[2]*int(self.ipt_shape[0]/4)*int(self.ipt_shape[0]/4), activation=self.activation, name="revealed"),                
                    tf.keras.layers.Reshape(target_shape=(int(self.ipt_shape[0]/4), int(self.ipt_shape[0]/4), self.hidden_layer_sizes[2])),                                     
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2DTranspose(filters=self.hidden_layer_sizes[1], kernel_size=3, strides=(2, 2), padding='same', activation=self.activation, name="deconv3"),                                       
                    tf.keras.layers.BatchNormalization(),                      
                    tf.keras.layers.Conv2DTranspose(filters=self.hidden_layer_sizes[0], kernel_size=5, strides=(2, 2), padding='same', activation=self.activation, name="deconv2"),
                    tf.keras.layers.BatchNormalization(),      
                    tf.keras.layers.Conv2DTranspose(filters=self.ipt_shape[2], kernel_size=5, strides=(1, 1), padding='same', activation=self.activation, name="deconv1"), #used to be sigmoid          
                    ])                

        else:            
            self.enc = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=self.ipt_shape),        
                tf.keras.layers.Dense(units=self.hidden_layer_sizes[0], activation=self.activation, name="conv1"),                 
                tf.keras.layers.BatchNormalization(),            
                tf.keras.layers.Dense(units=self.hidden_layer_sizes[1], activation=self.activation, name="conv2"),                                                  
                tf.keras.layers.BatchNormalization(),                   
                tf.keras.layers.Dense(units=self.hidden_layer_sizes[2], activation=self.activation, name="conv3"),            
                tf.keras.layers.BatchNormalization(),                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=4*self.hidden_layer_sizes[2], name="lastlayer") #no activation function                
                ]) 
            if self.NormalizationLayer is True:           
                self.dec = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=self.hidden_layer_sizes[2], activation=self.activation, name="revealed"),                
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(units=self.hidden_layer_sizes[1], activation=self.activation, name="revealed"),                                       
                    tf.keras.layers.BatchNormalization(),                      
                    tf.keras.layers.Dense(units=self.hidden_layer_sizes[0], activation=self.activation, name="revealed"),
                    tf.keras.layers.BatchNormalization(),      
                    tf.keras.layers.Dense(units=self.ipt_shape[0], activation=self.activation, name="revealed"),
                    tf.keras.layers.LayerNormalization()
                    ])
            else:
                self.dec = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=self.hidden_layer_sizes[2], activation=self.activation, name="revealed"),                
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(units=self.hidden_layer_sizes[1], activation=self.activation, name="revealed"),                                       
                    tf.keras.layers.BatchNormalization(),                      
                    tf.keras.layers.Dense(units=self.hidden_layer_sizes[0], activation=self.activation, name="revealed"),
                    tf.keras.layers.BatchNormalization(),      
                    tf.keras.layers.Dense(units=self.ipt_shape[0], activation=self.activation, name="revealed")
                    ])               

        self.disc = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.intrinsic_size),
            tf.keras.layers.Dense(units=32, activation=self.activation, name="disc1"),
            tf.keras.layers.Dense(units=64, activation=self.activation, name="disc2"),
            tf.keras.layers.Dense(units=128, activation=self.activation, name="disc3"),
            tf.keras.layers.Dense(units=1, activation=None)
            ])

        self.RSR = tf.keras.Sequential([tf.keras.layers.Flatten(), 
                                        tf.keras.layers.Dense(self.intrinsic_size, use_bias=False, name="rsr")])
                  
        self.vae_optimizer = tf.keras.optimizers.Adam(self.lr_rsr)
        self.disc_optimizer = tf.keras.optimizers.Adam(self.lr_rsr)
        self.enc_optimizer = tf.keras.optimizers.Adam(self.lr_rsr)
                         

    def dist_rsr(self, x, TEST): 
        mu1, sigma1, mu2, sigma2 = tf.split(self.enc(x), 
                                            num_or_size_splits=[self.hidden_layer_sizes[2], self.hidden_layer_sizes[2], self.hidden_layer_sizes[2], self.hidden_layer_sizes[2]], axis=1)
        self.mu1, self.sigma1, self.mu2, self.sigma2 = mu1, sigma1, mu2, sigma2

        mu_rsr1 = self.RSR(mu1) 
        A = self.RSR.weights[-1]
        self.A = tf.keras.layers.Flatten()(A)     
        self.Sigma1 = Sigma1 = tf.linalg.diag(sigma1)
        self.M1 = M1 = tf.matmul(tf.transpose(self.A), tf.matmul(Sigma1, self.A)) 
        
        d, U = tf.linalg.eigh(M1)
        sort = tf.sort(d, axis=1, direction="DESCENDING")
        s1 = tf.concat([sort[:,:int(self.intrinsic_size/2)], tf.zeros((x.shape[0],int(self.intrinsic_size/2)))], axis=1)
        S1 = tf.linalg.diag(s1)
        B = tf.matmul(tf.transpose(U, (0,2,1)), tf.matmul(S1, U))
        Sigma_rsr1 = tf.matmul(B, tf.transpose(B, (0,2,1))) + tf.eye(self.intrinsic_size) #produce semi-positive covariance
        self.Sigma_rsr1 = Sigma_rsr1
        LowG = ds.MultivariateNormalFullCovariance(loc=mu_rsr1, covariance_matrix=Sigma_rsr1)

        mu_rsr2 = self.RSR(mu2)            
        Sigma2 = tf.linalg.diag(sigma2)
        self.M2 = M2 = tf.matmul(tf.transpose(self.A), tf.matmul(Sigma2, self.A))       
        Sigma_rsr2 = tf.matmul(M2, tf.transpose(M2, (0,2,1))) + tf.eye(self.intrinsic_size)
        self.Sigma_rsr2 = Sigma_rsr2
        HighG = ds.MultivariateNormalFullCovariance(loc=mu_rsr2, covariance_matrix=Sigma_rsr2)

        if TEST is True:
            return LowG, HighG, "dummy~"
        else:   
            ratio = 1./(1. + self.prior_c)  
            l = np.random.uniform(0,1) 
            if l <= ratio:
                return LowG, mu1, mu_rsr1
            if l > ratio:
                return HighG, mu2, mu_rsr2
        
    def discriminate(self, x):
        return self.disc(x) 

    def decode(self, z):
        return self.dec(z)

    def renormalization(self, y):
        z = tf.math.l2_normalize(y, axis=-1, name="renormalization")
        return z

    def reconstruction_loss(self, x, x_tilde):    
        loss_norm_type = self.loss_norm_type
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        x_tilde = tf.reshape(x_tilde, (tf.shape(x_tilde)[0], -1))
        if loss_norm_type in ['MSE', 'mse', 'Frob', 'F']:
            return tf.reduce_mean(tf.square(tf.norm(x-x_tilde, ord=2, axis=1)))
        elif loss_norm_type in ['L1', 'l1']:
            return tf.reduce_mean(tf.norm(x-x_tilde, ord=1, axis=1))
        elif loss_norm_type in ['LAD', 'lad', 'L21', 'l21', 'L2', 'l2']:
            return tf.reduce_mean(tf.norm(x-x_tilde, ord=2, axis=1))
        else:
            raise Exception("Norm type error!")            


    def force_proj_loss(self):
        return tf.reduce_mean( tf.square (tf.matmul(tf.transpose(self.A), self.A) - \
                                          tf.eye(self.intrinsic_size) ) )   
 
    def prior_dist(self, z): # the only usage of input z is providing z's size      
        N = ds.MultivariateNormalDiag(loc=np.zeros((z.shape[0], self.intrinsic_size)), 
                                          scale_diag=[1.] * self.intrinsic_size)
        z_gen = tf.cast(N.sample(), dtype=tf.float32)
        return z_gen          

    def gan_discriminator_loss(self, real_output, fake_output):  
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def gan_generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)                
    
    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0], 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        self.x_hat = x_hat
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminate(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    def reconstruct(self, x):
        mean, _ = self.encode(x)
        return self.decode(mean)
    
    def get_reconstruction(self, X):        
        X_xgs = [] # after decoded
        X_zs = [] # latent 
        for _ in range(int(self.num_sampling)):            
            dist, _, _ = self.dist_rsr(X, TEST=self.pure_dists)
            X_z = dist.sample()                                    
            self.X_z = X_z
            self.X_xg = X_xg = self.decode(X_z)            
            X_xgs.append(X_xg)        
            X_zs.append(X_z)
        self.X_xgs = X_xgs = tf.reduce_mean(X_xgs, axis=0)
        self.X_zs = X_zs = tf.reduce_mean(X_zs, axis=0)
        return X_xgs

    def train(self, x):
        with tf.GradientTape() as vae_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as enc_tape:
            vae_loss = []                
            latent_loss = []
            enc_loss = []        
            for _ in range(int(self.num_sampling)):           
                
                self.q_z, self.mu, self.mu_rsr = q_z, mu, mu_rsr = self.dist_rsr(x, TEST=False)
                self.z = z = q_z.sample()                             
             
                self.x_recon = x_recon = self.decode(z)                
                self.z_gen = z_gen = self.prior_dist(z)
                            
                self.logits_z = logits_z = self.discriminate(z)
                self.logits_z_gen = logits_z_gen = self.discriminate(z_gen)            
                
                rec = self.reconstruction_loss(x, x_recon)   
                
                self.d_regularizer = d_regularizer = self.gradient_penalty(z, z_gen)
                wasserstein = tf.reduce_mean(logits_z_gen) - tf.reduce_mean(logits_z) + d_regularizer * self.gradient_penalty_weight
                enc = -tf.reduce_mean(logits_z)
                                       
                
                vae = rec
                         
                self.wasserstein = wasserstein
                
                vae_loss.append(vae)
                latent_loss.append(wasserstein) 
                enc_loss.append(enc)                     
            
            self.vae_loss = vae_loss = tf.reduce_mean(vae_loss, axis=0)        
            self.latent_loss = latent_loss = - tf.reduce_mean(latent_loss, axis=0)    
            self.enc_loss = enc_loss = tf.reduce_mean(enc_loss, axis=0)
        
        self.vae_gradients = vae_gradients = vae_tape.gradient( vae_loss, self.enc.trainable_variables + self.RSR.trainable_variables + self.dec.trainable_variables )            
        self.disc_gradients = disc_gradients = disc_tape.gradient( latent_loss, self.disc.trainable_variables )
        self.enc_gradients = enc_gradients = enc_tape.gradient(enc_loss, self.enc.trainable_variables + self.RSR.trainable_variables)

        self.vae_optimizer.apply_gradients(
            zip(vae_gradients, self.enc.trainable_variables + self.RSR.trainable_variables + self.dec.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )               
        self.enc_optimizer.apply_gradients(
            zip(enc_gradients, self.enc.trainable_variables + self.RSR.trainable_variables)
        )        
        
        return (vae_loss, latent_loss, enc_loss)

    
    def fit(self, x):

        losses = pd.DataFrame(columns=['vae_loss', 'disc_loss', 'enc_loss'])
        Vae_loss = Disc_loss = Enc_loss = []
                    
        if len(self.ipt_shape) > 1:
            n_samples, n_height, n_width, n_channel = x.shape
        else:
            n_samples, n_dim = x.shape

        n_batch = (n_samples - 1) // self.batch_size + 1
        self.n_batch = n_batch

        idx = np.arange(int(n_samples))
        np.random.shuffle(idx)       
                
        for epoch in range(self.epoch_size):
            print(f" Epoch: {epoch+1}/{self.epoch_size}")
     
            loss = []
            for batch in tqdm(range(self.n_batch), position=0):
                i_start = int(batch * self.batch_size)
                i_end = int((batch + 1) * self.batch_size)
                self.i_start = i_start
                self.i_end = i_end
                self.x_batch = x_batch = x[ idx[ i_start : i_end ] ]                  
                loss.append(self.train(x_batch))
                self.loss = loss
            losses.loc[len(losses.T)] = np.mean(loss, axis=0)
                          
            self.Vae_loss = Vae_loss.append(np.mean(loss, axis=0)[0])
            self.Disc_loss = Disc_loss.append(np.mean(loss, axis=0)[1])
            self.Enc_loss = Enc_loss.append(np.mean(loss, axis=0)[2])
            print(" epoch: {}/{} : &\n {}".format(epoch+1, self.epoch_size, losses))
   
    def get_generation(self, noise):
        x_gen = self.decode(noise)
        return x_gen