import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import register_keras_serializable
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
from keras.utils import serialize_keras_object, deserialize_keras_object

# Sampling layer
@register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

# VAE model
@register_keras_serializable()
class VAE(Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.beta = beta 

    def compile(self, optimizer):
        super(VAE, self).compile()
        self.optimizer = optimizer

    def reconstruction_loss(self, x, x_recon):
        return tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=[1, 2, 3]))

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            mu, log_var = self.encoder(data)
            z = self.sampling((mu, log_var))
            reconstruction = self.decoder(z)
            recon_loss = self.reconstruction_loss(data, reconstruction)
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
            total_loss = recon_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        mu, log_var = self.encoder(data)
        z = self.sampling((mu, log_var))
        reconstruction = self.decoder(z)
        recon_loss = self.reconstruction_loss(data, reconstruction)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
        total_loss = recon_loss + self.beta * kl_loss
        return {"loss": total_loss}

    def call(self, inputs):
        mu, log_var = self.encoder(inputs)
        z = self.sampling((mu, log_var))
        return self.decoder(z)
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "encoder": serialize_keras_object(self.encoder),
            "decoder": serialize_keras_object(self.decoder),
            "beta": self.beta,
        }

    @classmethod
    def from_config(cls, config):
        encoder = deserialize_keras_object(config.pop("encoder"))
        decoder = deserialize_keras_object(config.pop("decoder"))
        beta = config.pop("beta")
        return cls(encoder=encoder, decoder=decoder, beta=beta, **config)

def load_vae_model():
    loaded_vae = load_model('residual_vae (1).keras', custom_objects={"VAE": VAE, "Sampling": Sampling})
    return loaded_vae

def preprocess_image(img, target_size=(256, 256)):
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    return img_tensor

def compute_error_maps_and_scores(model, originals):
    reconstructed = model.predict(originals)
    error_maps = np.mean((originals - reconstructed) ** 2, axis=-1)
    scores = np.mean(error_maps, axis=(1, 2))
    return reconstructed, error_maps, scores

model = load_vae_model()

# --- Streamlit app ---
st.set_page_config(page_title="Detector de Anomalías", layout="centered")
st.title("Detección de Anomalías con Autoencoder")

uploaded_file = st.file_uploader("Sube una imagen PNG o JPG", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = preprocess_image(image)
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    _, error_maps, scores = compute_error_maps_and_scores(model, img_tensor)

    # Mostrar en horizontal original y mapa de error
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Mapa de error")
        fig, ax = plt.subplots()
        ax.imshow(error_maps[0], cmap="hot")
        ax.axis("off")
        st.pyplot(fig)

    # Mostrar resultado final
    threshold = 0.01
    error_score = scores[0]
    st.subheader("Resultado")
    if error_score > threshold:
        st.error(f"⚠️ Anomalía detectada (score: {error_score:.4f})")
    else:
        st.success(f"✅ Imagen normal (score: {error_score:.4f})")
