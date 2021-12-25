"""
DOCSTRING
"""
import glob
import imageio
import matplotlib.pyplot as pyplot
import PIL
import tensorflow
import time

class DCGAN:
    """
    DOCSTRING
    """
    def __init__(self):
        """
        DOCSTRING
        """
        self.cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.discriminator = self.make_discriminator_model()
        self.generator = self.make_generator_model()
        self.noise_dim = 100

    def create_gif(self):
        """
        DOCSTRING
        """
        anim_file = 'dcgan.gif'
        with imageio.get_writer(anim_file, mode='I') as writer:
          filenames = glob.glob('image*.png')
          filenames = sorted(filenames)
          for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
          image = imageio.imread(filename)
          writer.append_data(image)

    def discriminator_loss(self, real_output, fake_output):
        """
        DOCSTRING
        """
        real_loss = self.cross_entropy(tensorflow.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tensorflow.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def display_image(self, epoch_no):
        """
        This function displays a single image using the epoch number.
        """
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

    def generate_and_save_images(self, model, epoch, test_input):
        """
        DOCSTRING
        """
        predictions = model(test_input, training=False)
        fig = pyplot.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            pyplot.subplot(4, 4, i+1)
            pyplot.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            pyplot.axis('off')
        pyplot.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        pyplot.show()

    def generator_loss(self, fake_output):
        """
        DOCSTRING
        """
        return self.cross_entropy(tensorflow.ones_like(fake_output), fake_output)
    
    def make_discriminator_model(self):
        """
        DOCSTRING
        """
        model = tensorflow.keras.Sequential()
        model.add(tensorflow.keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        model.add(tensorflow.keras.layers.LeakyReLU())
        model.add(tensorflow.keras.layers.Dropout(0.3))
        model.add(tensorflow.keras.layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tensorflow.keras.layers.LeakyReLU())
        model.add(tensorflow.keras.layers.Dropout(0.3))
        model.add(tensorflow.keras.layers.Flatten())
        model.add(tensorflow.keras.layers.Dense(1))
        return model

    def make_generator_model(self):
        """
        DOCSTRING
        """
        model = tensorflow.keras.Sequential()
        model.add(tensorflow.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.LeakyReLU())
        model.add(tensorflow.keras.layers.Reshape((7, 7, 256)))
        model.add(tensorflow.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.LeakyReLU())
        model.add(tensorflow.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.LeakyReLU())
        model.add(tensorflow.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        return model

    def train(self, dataset, epochs, batch_size):
        """
        DOCSTRING
        """
        num_examples_to_generate = 16
        seed = tensorflow.random.normal([num_examples_to_generate, self.noise_dim])
        checkpoint = tensorflow.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator, discriminator=self.discriminator)
        for epoch in range(epochs):
            start = time.time()
            for image_batch in dataset:
              self.train_step(image_batch, batch_size)
            self.generate_and_save_images(self.generator, epoch + 1, seed)
            if (epoch + 1) % 15 == 0:
              checkpoint.save(file_prefix='.\\training_checkpoints\\ckpt')
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        self.generate_and_save_images(self.generator, epochs, seed)

    @tensorflow.function
    def train_step(self, images, batch_size):
        """
        DOCSTRING
        """
        print('step')
        noise = tensorflow.random.normal([batch_size, self.noise_dim])
        with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
          generated_images = self.generator(noise, training=True)
          real_output = self.discriminator(images, training=True)
          fake_output = self.discriminator(generated_images, training=True)
          gen_loss = self.generator_loss(fake_output)
          disc_loss = self.discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(
            gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(
            gradients_of_discriminator, self.discriminator.trainable_variables))

if __name__ == '__main__':
    (train_images, train_labels), (_, _) = tensorflow.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    batch_size = 256
    train_dataset = tensorflow.data.Dataset.from_tensor_slices(
        train_images).shuffle(60000).batch(batch_size)
    dcgan = DCGAN()
    dcgan.train(train_dataset, 50, batch_size)
