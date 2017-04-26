import numpy as np
# import cts.model as model
# from cts.model.CTS import model
from .TreeDensity import TreeDensity as model

def L_shaped_context(image, y, x):
    """This grabs the L-shaped context around a given pixel.
    Out-of-bounds values are set to 0xFFFFFFFF."""
    context = [0] * 4
    if x > 0:
        context[3] = image[y][x - 1]
    if y > 0:
        context[2] = image[y - 1][x]
        context[1] = image[y - 1][x - 1] if x > 0 else 0
        context[0] = image[y - 1][x + 1] if x < image.shape[1] - 1 else 0

    # The most important context symbol, 'left', comes last.
    return context


def dilations_context(image, y, x):
    """Generates a dilations-based context.
    We successively dilate first to the left, then up, then diagonally, with strides 1, 2, 4, 8, 16.
    """
    SPAN = 5
    # Default to -1 context.
    context = [0] * (SPAN * 3)

    min_x, index = 1, (SPAN * 3) - 1
    for i in range(SPAN):
        if x >= min_x:
            context[index] = image[y][x - min_x]
        index -= 3
        min_x = min_x << 1

    min_y, index = 1, (SPAN * 3) - 2
    for i in range(SPAN):
        if y >= min_y:
            context[index] = image[y - min_y][x]
        index -= 3
        min_y = min_y << 1

    min_p, index = 1, (SPAN * 3) - 3
    for i in range(SPAN):
        if x >= min_p and y >= min_p:
            context[index] = image[y - min_p][x - min_p]
        index -= 3
        min_p = min_p << 1

    return context


def gray_to_symbol(channel):
    return int(channel * 15)


def symbol_to_gray(colour):
    return colour / 15


def gray_to_symbols(frame, output):
    """Preprocesses the given frame into a CTS-compatible representation.
    """
    assert(frame.shape[:2] == output.shape)
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            output[y, x] = gray_to_symbol(frame[y, x])
            # print(frame[y, x])

    return output


def symbols_to_gray(frame, output):
    """Inverse of gray_to_symbols.
    """
    assert(frame.shape == output.shape[:2])
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            output[y, x] = symbol_to_gray(frame[y, x])

    return output


def rgb_to_symbol(channels):
    """Converts an RGB triple into an atomic colour (24-bit integer).
    """
    return (channels[0] << 16) | (channels[1] << 8) | (channels[2] << 0)


def symbol_to_rgb(colour):
    """Inverse operation of rgb_to_symbol.
    Returns: a (r, g, b) tuple.
    """
    return (colour >> 16, (colour >> 8) & 0xFF, (colour >> 0) & 0xFF)


def rgb_to_symbols(frame, output):
    """Preprocesses the given frame into a CTS-compatible representation.
    """
    assert(frame.shape[:2] == output.shape)
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            output[y, x] = rgb_to_symbol(frame[y, x])

    return output


def symbols_to_rgb(frame, output):
    """Inverse of rgb_to_symbols.
    """
    assert(frame.shape == output.shape[:2])
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            output[y, x] = symbol_to_rgb(frame[y, x])

    return output


class TreeDensity:

    def __init__(self, frame_shape):
        self.symbol_frame = np.zeros((frame_shape[0:2]), dtype=np.uint32)
        self.context_functor = L_shaped_context
        self.model = model(16, 4, frame_shape[0], frame_shape[1])
        self.contexts_vector = np.zeros(shape=(self.symbol_frame.shape[0], self.symbol_frame.shape[1], 5))

    def new_old(self, frame, keep=True):
        gray_to_symbols(frame, self.symbol_frame)
        total_log_probability = 0.0
        log_probs = np.zeros(shape=self.symbol_frame.shape)

        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                context = self.context_functor(self.symbol_frame, y, x)
                colour = self.symbol_frame[y, x]
                self.contexts_vector[y, x, :] = context + [colour]

        log_probs = self.model.new_old(self.contexts_vector, keep=keep)
        total_log_probability = np.sum(log_probs)

        return total_log_probability, log_probs


class DensityModel(object):
    """A density model for Freeway frames.
    This is exactly the same as the ConvolutionalDensityModel, except that we use one model for each
    pixel location.
    """

    def __init__(self, frame_shape, context_functor, conv=False, alphabet=None):
        """Constructor.
        Args:
            init_frame: A sample frame (numpy array) from which we determine the shape and type of our data.
            context_functor: Function mapping image x position to a context.
        """
        # For efficiency, we'll pre-process the frame into our internal representation.
        self.symbol_frame = np.zeros((frame_shape[0:2]), dtype=np.uint32)

        context_length = len(context_functor(self.symbol_frame, -1, -1))
        self.models = np.zeros(frame_shape[0:2], dtype=object)
        if conv:
            self.convolutional_model = model.CTS(context_length=context_length, alphabet=alphabet)
        for y in range(frame_shape[0]):
            for x in range(frame_shape[1]):
                if conv:
                    self.models[y, x] = self.convolutional_model
                else:
                    self.models[y, x] = model.CTS(context_length=context_length, alphabet=alphabet)

        self.context_functor = context_functor

    def log_prob(self, frame):
        gray_to_symbols(frame, self.symbol_frame)
        total_log_probability = 0.0
        log_probs = np.zeros(shape=self.symbol_frame.shape)
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                context = self.context_functor(self.symbol_frame, y, x)
                colour = self.symbol_frame[y, x]
                log_val = self.models[y, x].log_prob(context=context, symbol=colour)
                total_log_probability += log_val
                log_probs[y, x] = log_val

        return total_log_probability, log_probs

    def update(self, frame):
        gray_to_symbols(frame, self.symbol_frame)
        total_log_probability = 0.0
        log_probs = np.zeros(shape=self.symbol_frame.shape)
        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                context = self.context_functor(self.symbol_frame, y, x)
                colour = self.symbol_frame[y, x]
                log_val = self.models[y, x].update(context=context, symbol=colour)
                total_log_probability += log_val
                log_probs[y, x] = log_val

        return total_log_probability, log_probs

    def sample(self):
        output_frame = np.zeros((*self.symbol_frame.shape, 1), dtype=np.float32)
        print(output_frame.shape)

        for y in range(self.symbol_frame.shape[0]):
            for x in range(self.symbol_frame.shape[1]):
                # From a programmer's perspective, this is why we must respect the chain rule: otherwise
                # we condition on garbage.
                context = self.context_functor(self.symbol_frame, y, x)
                self.symbol_frame[y, x] = self.models[y, x].sample(context=context, rejection_sampling=True)

        return symbols_to_gray(self.symbol_frame, output_frame)