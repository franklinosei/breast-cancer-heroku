
# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
    im = image
    # image = tf.image.resize(image, [target_dim, target_dim])
    # image = np.array(image)
    return im