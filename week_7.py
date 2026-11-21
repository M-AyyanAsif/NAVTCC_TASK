import math
import random
import re
from collections import Counter

# ==============================================================================
# 0. BOOSTING (AdaBoost Algorithm from Scratch)
# ==============================================================================
class DecisionStump:
    """A 1-level decision tree (weak classifier) for boosting."""
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = len(X)
        X_column = [x[self.feature_idx] for x in X]
        predictions = [1] * n_samples
        if self.polarity == 1:
            for i in range(n_samples):
                if X_column[i] < self.threshold:
                    predictions[i] = -1
        else:
            for i in range(n_samples):
                if X_column[i] > self.threshold:
                    predictions[i] = -1
        return predictions


class AdaBoostClassifier:
    """AdaBoost ensemble classifier."""
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        w = [1.0 / n_samples] * n_samples

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                X_column = [x[feature_i] for x in X]
                thresholds = sorted(list(set(X_column)))
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        # Predict
                        preds = [1] * n_samples
                        if polarity == 1:
                            for idx in range(n_samples):
                                if X_column[idx] < threshold:
                                    preds[idx] = -1
                        else:
                            for idx in range(n_samples):
                                if X_column[idx] > threshold:
                                    preds[idx] = -1

                        # Calculate weighted error
                        error = sum(w[i] for i in range(n_samples) if y[i] != preds[i])

                        if error > 0.5:
                            error = 1.0 - error
                            polarity = -polarity

                        if error < min_error:
                            min_error = error
                            stump.polarity = polarity
                            stump.threshold = threshold
                            stump.feature_idx = feature_i

            EPS = 1e-10
            min_error = max(min_error, EPS)
            stump.alpha = 0.5 * math.log((1.0 - min_error) / min_error)
            predictions = stump.predict(X)

            # Update sample weights
            for i in range(n_samples):
                w[i] = w[i] * math.exp(-stump.alpha * y[i] * predictions[i])
            w_sum = sum(w)
            w = [val / w_sum for val in w]
            self.estimators.append(stump)

    def predict(self, X):
        stump_preds = [stump.alpha * math.copysign(1, p) for stump in self.estimators for p in stump.predict(X)]
        n_samples = len(X)
        final_preds = []
        for i in range(n_samples):
            score = sum(stump.alpha * stump.predict([X[i]])[0] for stump in self.estimators)
            final_preds.append(1 if score >= 0 else -1)
        return final_preds


# ==============================================================================
# TASK 56: DEMONSTRATION OF NEURAL NETWORKS (From Scratch)
# ==============================================================================
class TwoLayerNeuralNetwork:
    """
    A foundational 2-layer Neural Network with 1 hidden layer.
    Demonstrates forward pass, MSE loss, and analytical backpropagation.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr
        # Initialize weights with small random values
        random.seed(42)
        self.W1 = [[(random.random() - 0.5) * 0.5 for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[(random.random() - 0.5) * 0.5 for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * output_dim

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

    @staticmethod
    def sigmoid_deriv(x):
        s = TwoLayerNeuralNetwork.sigmoid(x)
        return s * (1.0 - s)

    def forward(self, x):
        # Layer 1
        self.z1 = [sum(x[i] * self.W1[i][j] for i in range(len(x))) + self.b1[j] for j in range(len(self.b1))]
        self.a1 = [self.sigmoid(val) for val in self.z1]

        # Layer 2
        self.z2 = [sum(self.a1[i] * self.W2[i][j] for i in range(len(self.a1))) + self.b2[j] for j in range(len(self.b2))]
        self.a2 = [self.sigmoid(val) for val in self.z2]
        return self.a2

    def train_step(self, x, target):
        # Forward pass
        output = self.forward(x)

        # Output error: delta2 = (output - target) * sigmoid'(z2)
        delta2 = [(output[k] - target[k]) * output[k] * (1.0 - output[k]) for k in range(len(target))]

        # Hidden layer error: delta1 = (delta2 * W2^T) * sigmoid'(z1)
        delta1 = [0.0] * len(self.a1)
        for j in range(len(self.a1)):
            error = sum(delta2[k] * self.W2[j][k] for k in range(len(delta2)))
            delta1[j] = error * self.a1[j] * (1.0 - self.a1[j])

        # Update weights and biases for W2, b2
        for j in range(len(self.a1)):
            for k in range(len(delta2)):
                self.W2[j][k] -= self.lr * delta2[k] * self.a1[j]
        for k in range(len(delta2)):
            self.b2[k] -= self.lr * delta2[k]

        # Update weights and biases for W1, b1
        for i in range(len(x)):
            for j in range(len(delta1)):
                self.W1[i][j] -= self.lr * delta1[j] * x[i]
        for j in range(len(delta1)):
            self.b1[j] -= self.lr * delta1[j]

        # Loss (MSE)
        return sum(0.5 * (output[k] - target[k]) ** 2 for k in range(len(target)))


# ==============================================================================
# TASK 57 & TASK 58: MULTI-LAYER PERCEPTRON (MLP) & FEED FORWARD NN
# Supports multiple activations, dropout, and backpropagation
# ==============================================================================
class Activation:
    @staticmethod
    def relu(x):
        return max(0.0, x)

    @staticmethod
    def relu_deriv(x):
        return 1.0 if x > 0 else 0.0

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

    @staticmethod
    def sigmoid_deriv(x):
        s = Activation.sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def tanh_deriv(x):
        t = math.tanh(x)
        return 1.0 - t ** 2

    @staticmethod
    def softmax(vector):
        max_v = max(vector)
        exp_vals = [math.exp(v - max_v) for v in vector]
        total = sum(exp_vals)
        return [v / total for v in exp_vals]


class MultiLayerPerceptron:
    """
    Modular Multi-Layer Perceptron with customizable architecture,
    activation functions, dropout regularization, and cross-entropy loss.
    """
    def __init__(self, layer_sizes, activation="relu", dropout_rate=0.0):
        self.layer_sizes = layer_sizes
        self.activation_type = activation
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []

        # He/Xavier weight initialization
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = math.sqrt(2.0 / fan_in) if activation == "relu" else math.sqrt(1.0 / fan_in)
            W = [[random.gauss(0, scale) for _ in range(fan_out)] for _ in range(fan_in)]
            b = [0.0 for _ in range(fan_out)]
            self.weights.append(W)
            self.biases.append(b)

    def _activate(self, x):
        if self.activation_type == "relu":
            return Activation.relu(x)
        elif self.activation_type == "tanh":
            return Activation.tanh(x)
        return Activation.sigmoid(x)

    def _activate_deriv(self, x):
        if self.activation_type == "relu":
            return Activation.relu_deriv(x)
        elif self.activation_type == "tanh":
            return Activation.tanh_deriv(x)
        return Activation.sigmoid_deriv(x)

    def forward(self, x, training=False):
        self.activations = [x]
        self.pre_activations = []
        self.dropout_masks = []

        current_a = x
        num_layers = len(self.weights)

        for l in range(num_layers):
            W = self.weights[l]
            b = self.biases[l]
            fan_in = len(W)
            fan_out = len(W[0])

            z = [sum(current_a[i] * W[i][j] for i in range(fan_in)) + b[j] for j in range(fan_out)]
            self.pre_activations.append(z)

            if l == num_layers - 1:
                # Output layer: Softmax for multi-class or Sigmoid
                a = Activation.softmax(z)
            else:
                a = [self._activate(val) for val in z]
                # Dropout regularization
                if training and self.dropout_rate > 0.0:
                    keep_prob = 1.0 - self.dropout_rate
                    mask = [1.0 if random.random() < keep_prob else 0.0 for _ in range(len(a))]
                    a = [(val * m) / keep_prob for val, m in zip(a, mask)]
                    self.dropout_masks.append(mask)
                else:
                    self.dropout_masks.append([1.0] * len(a))

            self.activations.append(a)
            current_a = a

        return current_a

    def backward(self, target, lr=0.01):
        num_layers = len(self.weights)
        output = self.activations[-1]

        # For Softmax + Cross-Entropy: delta = output - target
        delta = [output[k] - target[k] for k in range(len(target))]

        # Backpropagate through layers
        for l in reversed(range(num_layers)):
            a_prev = self.activations[l]
            W = self.weights[l]
            b = self.biases[l]

            # Compute gradients for weights & biases
            dW = [[a_prev[i] * delta[j] for j in range(len(delta))] for i in range(len(a_prev))]
            db = delta[:]

            # Compute delta for earlier layer if not input
            if l > 0:
                z_prev = self.pre_activations[l - 1]
                mask = self.dropout_masks[l - 1]
                new_delta = [0.0] * len(a_prev)
                for i in range(len(a_prev)):
                    error = sum(delta[j] * W[i][j] for j in range(len(delta)))
                    deriv = self._activate_deriv(z_prev[i])
                    new_delta[i] = error * deriv * mask[i]
                delta = new_delta

            # Gradient update
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W[i][j] -= lr * dW[i][j]
            for j in range(len(b)):
                b[j] -= lr * db[j]


# ==============================================================================
# TASK 59: EXERCISE ON NEURAL NETWORK LOSS FUNCTIONS (Detailed 7 Functions)
# ==============================================================================
class LossFunctions:
    """
    Comprehensive implementation of 7 primary loss functions in ML/DL.
    """
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """1. MSE: Mean Squared Error (Regression)"""
        n = len(y_true)
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """2. MAE: Mean Absolute Error (L1 Loss)"""
        n = len(y_true)
        return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

    @staticmethod
    def binary_cross_entropy(y_true, y_pred, eps=1e-15):
        """3. BCE: Binary Cross-Entropy (Binary Classification)"""
        loss = 0.0
        n = len(y_true)
        for yt, yp in zip(y_true, y_pred):
            p = max(eps, min(1.0 - eps, yp))
            loss += -(yt * math.log(p) + (1.0 - yt) * math.log(1.0 - p))
        return loss / n

    @staticmethod
    def categorical_cross_entropy(y_true_onehot, y_pred_prob, eps=1e-15):
        """4. CCE: Categorical Cross-Entropy (Multi-class Classification)"""
        total_loss = 0.0
        for true_vec, pred_vec in zip(y_true_onehot, y_pred_prob):
            for yt, yp in zip(true_vec, pred_vec):
                p = max(eps, min(1.0 - eps, yp))
                total_loss += -yt * math.log(p)
        return total_loss / len(y_true_onehot)

    @staticmethod
    def hinge_loss(y_true, y_pred):
        """5. Hinge Loss (SVM / Maximum Margin) - expects y_true in {-1, 1}"""
        return sum(max(0.0, 1.0 - yt * yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

    @staticmethod
    def huber_loss(y_true, y_pred, delta=1.0):
        """6. Huber Loss (Robust regression combining MSE & MAE)"""
        total = 0.0
        for yt, yp in zip(y_true, y_pred):
            error = abs(yt - yp)
            if error <= delta:
                total += 0.5 * (error ** 2)
            else:
                total += delta * (error - 0.5 * delta)
        return total / len(y_true)

    @staticmethod
    def kl_divergence(p_dist, q_dist, eps=1e-15):
        """7. Kullback-Leibler (KL) Divergence (Relative Entropy)"""
        kl = 0.0
        for p, q in zip(p_dist, q_dist):
            p_safe = max(eps, p)
            q_safe = max(eps, q)
            kl += p_safe * math.log(p_safe / q_safe)
        return kl


# ==============================================================================
# TASK 60, 61, 62: NLP, TEXT PROCESSING & ANALYSIS
# ==============================================================================
class TextProcessor:
    """NLTK/SpaCy-like text processing and NLP feature engineering pipeline."""

    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they'
    }

    @staticmethod
    def tokenize_words(text):
        """Word tokenization using regex."""
        return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

    @staticmethod
    def tokenize_sentences(text):
        """Sentence tokenization."""
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    @staticmethod
    def remove_stopwords(tokens):
        """Filter out common stopwords."""
        return [t for t in tokens if t not in TextProcessor.STOPWORDS]

    @staticmethod
    def simple_stem(word):
        """Porter-style suffix removal stemmer."""
        suffixes = ['ing', 'ly', 'ed', 'ies', 'es', 's', 'ment', 'able', 'tion']
        for suffix in sorted(suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    @staticmethod
    def pos_tag(tokens):
        """Rule-based Part-Of-Speech (POS) tagger."""
        tags = []
        for t in tokens:
            if t.endswith('ly'):
                tags.append((t, 'RB'))  # Adverb
            elif t.endswith('ing') or t.endswith('ed'):
                tags.append((t, 'VBD')) # Verb
            elif t.endswith('tion') or t.endswith('ness') or t.endswith('er'):
                tags.append((t, 'NN'))  # Noun
            elif t.isdigit():
                tags.append((t, 'CD'))  # Cardinal Number
            else:
                tags.append((t, 'NN'))  # Default noun
        return tags


class TextAnalysis:
    """Bag-of-Words, TF-IDF calculation, Cosine Similarity, and Sentiment."""

    @staticmethod
    def compute_tf_idf(corpus):
        """Computes TF-IDF matrix for a collection of documents."""
        # 1. Build vocabulary
        tokenized_docs = [TextProcessor.remove_stopwords(TextProcessor.tokenize_words(doc)) for doc in corpus]
        vocab = sorted(list(set(w for doc in tokenized_docs for w in doc)))
        n_docs = len(corpus)

        # 2. IDF: log(N / (1 + doc_frequency))
        idf = {}
        for term in vocab:
            doc_freq = sum(1 for doc in tokenized_docs if term in doc)
            idf[term] = math.log((n_docs + 1) / (doc_freq + 1)) + 1.0

        # 3. TF-IDF vectors
        tfidf_matrix = []
        for doc in tokenized_docs:
            term_counts = Counter(doc)
            total_terms = len(doc) if len(doc) > 0 else 1
            vector = []
            for term in vocab:
                tf = term_counts[term] / total_terms
                vector.append(tf * idf[term])
            # L2 normalize
            norm = math.sqrt(sum(v ** 2 for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            tfidf_matrix.append(vector)

        return vocab, tfidf_matrix

    @staticmethod
    def cosine_similarity(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def rule_based_sentiment(text):
        """Basic lexicon sentiment scoring."""
        positive_words = {'good', 'great', 'excellent', 'happy', 'best', 'super', 'positive', 'amazing', 'love', 'fantastic'}
        negative_words = {'bad', 'terrible', 'worst', 'horrible', 'poor', 'negative', 'sad', 'hate', 'awful', 'fail'}
        tokens = TextProcessor.tokenize_words(text)
        pos_count = sum(1 for t in tokens if t in positive_words)
        neg_count = sum(1 for t in tokens if t in negative_words)
        score = pos_count - neg_count
        label = "Positive" if score > 0 else ("Negative" if score < 0 else "Neutral")
        return {"score": score, "label": label, "positive_hits": pos_count, "negative_hits": neg_count}


# ==============================================================================
# TASK 63 & TASK 64: CONVOLUTIONAL NEURAL NETWORKS (CNN) & IMAGE CLASSIFICATION
# ==============================================================================
class Conv2DLayer:
    """2D Convolution layer with padding and stride from scratch."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize kernels with small weights
        scale = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.kernels = [
            [
                [[random.gauss(0, scale) for _ in range(kernel_size)] for _ in range(kernel_size)]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.biases = [0.0] * out_channels

    def _pad(self, channel_data):
        h = len(channel_data)
        w = len(channel_data[0])
        p = self.padding
        padded = [[0.0] * (w + 2 * p) for _ in range(h + 2 * p)]
        for r in range(h):
            for c in range(w):
                padded[r + p][c + p] = channel_data[r][c]
        return padded

    def forward(self, input_feature_map):
        """
        input_feature_map: [in_channels, height, width]
        Returns: [out_channels, out_h, out_w]
        """
        h = len(input_feature_map[0])
        w = len(input_feature_map[0][0])
        k = self.kernel_size
        s = self.stride
        p = self.padding

        out_h = (h + 2 * p - k) // s + 1
        out_w = (w + 2 * p - k) // s + 1

        padded_inputs = [self._pad(input_feature_map[c]) for c in range(self.in_channels)]
        output_maps = []

        for out_c in range(self.out_channels):
            feature_map = [[self.biases[out_c]] * out_w for _ in range(out_h)]
            for r in range(out_h):
                for c in range(out_w):
                    val = self.biases[out_c]
                    r_start = r * s
                    c_start = c * s
                    for in_c in range(self.in_channels):
                        kernel = self.kernels[out_c][in_c]
                        for kr in range(k):
                            for kc in range(k):
                                val += padded_inputs[in_c][r_start + kr][c_start + kc] * kernel[kr][kc]
                    feature_map[r][c] = val
            output_maps.append(feature_map)
        return output_maps


class MaxPooling2D:
    """2D Max Pooling layer."""
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_feature_map):
        channels = len(input_feature_map)
        h = len(input_feature_map[0])
        w = len(input_feature_map[0][0])
        out_h = h // self.stride
        out_w = w // self.stride

        pooled_maps = []
        for ch in range(channels):
            pooled = [[0.0] * out_w for _ in range(out_h)]
            for r in range(out_h):
                for c in range(out_w):
                    r_start = r * self.stride
                    c_start = c * self.stride
                    vals = [
                        input_feature_map[ch][r_start + kr][c_start + kc]
                        for kr in range(self.pool_size)
                        for kc in range(self.pool_size)
                        if r_start + kr < h and c_start + kc < w
                    ]
                    pooled[r][c] = max(vals) if vals else 0.0
            pooled_maps.append(pooled)
        return pooled_maps


class SimpleCNNClassifier:
    """
    Complete Convolutional Neural Network combining:
    Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense / Softmax
    """
    def __init__(self, in_channels=1, img_size=(8, 8), num_classes=2):
        self.conv = Conv2DLayer(in_channels=in_channels, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.pool = MaxPooling2D(pool_size=2, stride=2)
        # Flattened dimensions: 2 channels * (8//2) * (8//2) = 2 * 4 * 4 = 32
        self.flatten_dim = 2 * (img_size[0] // 2) * (img_size[1] // 2)
        self.mlp = MultiLayerPerceptron([self.flatten_dim, 16, num_classes], activation="relu")

    def forward(self, img_channels):
        # 1. Conv2D
        conv_out = self.conv.forward(img_channels)
        # 2. ReLU
        relu_out = [[[max(0.0, v) for v in row] for row in channel] for channel in conv_out]
        # 3. MaxPool2D
        pooled_out = self.pool.forward(relu_out)
        # 4. Flatten
        flattened = [val for ch in pooled_out for row in ch for val in row]
        # 5. Dense MLP
        return self.mlp.forward(flattened)


# ==============================================================================
# MAIN TEST & DEMONSTRATION SUITE
# ==============================================================================
def run_all_week7_tasks():
    print("=" * 80)
    print("RUNNING WEEK 7 TASKS: DEEP LEARNING I & NLP FUNDAMENTALS")
    print("=" * 80)

    # 1. Boosting Demonstration
    print("\n[Week 7 - Syllabus] AdaBoost Ensemble Demonstration:")
    X_boost = [[1.0, 2.0], [2.0, 1.0], [1.5, 1.8], [6.0, 5.0], [7.0, 8.0], [6.5, 7.2]]
    y_boost = [-1, -1, -1, 1, 1, 1]
    boost = AdaBoostClassifier(n_estimators=3)
    boost.fit(X_boost, y_boost)
    preds = boost.predict(X_boost)
    print(f"  True Labels:       {y_boost}")
    print(f"  AdaBoost Predicted: {preds}")

    # 2. Task 56: Neural Network from Scratch
    print("\n[Task 56] Demonstration of Neural Networks (XOR Problem):")
    nn = TwoLayerNeuralNetwork(input_dim=2, hidden_dim=4, output_dim=1, lr=0.5)
    X_xor = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y_xor = [[0.0], [1.0], [1.0], [0.0]]
    for epoch in range(1000):
        for x_samp, y_samp in zip(X_xor, y_xor):
            nn.train_step(x_samp, y_samp)
    print("  Testing Trained NN on XOR:")
    for x_samp, y_samp in zip(X_xor, y_xor):
        out = nn.forward(x_samp)
        print(f"    Input: {x_samp} -> Target: {y_samp[0]} | Predicted: {out[0]:.4f}")

    # 3. Task 57 & 58: Multi-Layer Perceptron (MLP) & Feed Forward NN
    print("\n[Task 57 & 58] Multi-Layer Perceptron (3 Layers with Softmax & Dropout):")
    mlp = MultiLayerPerceptron([4, 8, 3], activation="relu", dropout_rate=0.1)
    sample_x = [1.2, 0.4, -0.5, 2.1]
    prob_dist = mlp.forward(sample_x, training=False)
    print(f"  Input: {sample_x}")
    print(f"  Softmax Output Probabilities: {[round(p, 4) for p in prob_dist]}")
    print(f"  Sum of probabilities: {sum(prob_dist):.4f}")

    # 4. Task 59: Loss Functions
    print("\n[Task 59] Exercise on 7 Neural Network Loss Functions:")
    y_t = [1.0, 0.0, 1.0, 1.0]
    y_p = [0.9, 0.1, 0.8, 0.95]
    print(f"  1. MSE:  {LossFunctions.mean_squared_error(y_t, y_p):.6f}")
    print(f"  2. MAE:  {LossFunctions.mean_absolute_error(y_t, y_p):.6f}")
    print(f"  3. BCE:  {LossFunctions.binary_cross_entropy(y_t, y_p):.6f}")
    y_t_onehot = [[1, 0], [0, 1]]
    y_p_probs = [[0.95, 0.05], [0.2, 0.8]]
    print(f"  4. CCE:  {LossFunctions.categorical_cross_entropy(y_t_onehot, y_p_probs):.6f}")
    print(f"  5. Hinge Loss: {LossFunctions.hinge_loss([1, -1, 1], [0.8, -0.7, 0.5]):.6f}")
    print(f"  6. Huber Loss: {LossFunctions.huber_loss(y_t, y_p, delta=0.5):.6f}")
    print(f"  7. KL Divergence: {LossFunctions.kl_divergence([0.7, 0.3], [0.6, 0.4]):.6f}")

    # 5. Task 60, 61, 62: Text Processing & Analysis
    print("\n[Task 60, 61, 62] Text Processing, Linguistics, and Document Similarity:")
    corpus = [
        "Artificial Intelligence and Machine Learning are transforming modern industry.",
        "Deep Learning models with Neural Networks process images and text.",
        "Natural Language Processing helps machines understand human speech and documents."
    ]
    vocab, tfidf = TextAnalysis.compute_tf_idf(corpus)
    print(f"  Vocabulary size: {len(vocab)} words")
    sim_0_1 = TextAnalysis.cosine_similarity(tfidf[0], tfidf[1])
    sim_0_2 = TextAnalysis.cosine_similarity(tfidf[0], tfidf[2])
    print(f"  Similarity between Doc 1 & Doc 2: {sim_0_1:.4f}")
    print(f"  Similarity between Doc 1 & Doc 3: {sim_0_2:.4f}")

    sample_sentence = "The student is actively learning deep learning and training artificial models!"
    tokens = TextProcessor.tokenize_words(sample_sentence)
    filtered = TextProcessor.remove_stopwords(tokens)
    stemmed = [TextProcessor.simple_stem(t) for t in filtered]
    pos = TextProcessor.pos_tag(filtered)
    sentiment = TextAnalysis.rule_based_sentiment("This artificial intelligence course is amazing and great!")
    print(f"  Original Text: {sample_sentence}")
    print(f"  Filtered Tokens: {filtered}")
    print(f"  Stemmed Tokens:  {stemmed}")
    print(f"  POS Tags:        {pos[:5]}")
    print(f"  Sentiment Analysis Result: {sentiment}")

    # 6. Task 63 & 64: CNN Mechanics & Image Classification
    print("\n[Task 63 & 64] Convolutional Neural Network (Conv2D + MaxPool + Dense):")
    # Synthetic 8x8 single-channel image
    dummy_image = [[[random.random() for _ in range(8)] for _ in range(8)]]
    cnn = SimpleCNNClassifier(in_channels=1, img_size=(8, 8), num_classes=2)
    output_probs = cnn.forward(dummy_image)
    print(f"  Input Image Dimensions: 1 channel x 8 x 8")
    print(f"  CNN Forward Pass Output Class Probabilities: {[round(p, 4) for p in output_probs]}")
    print("=" * 80)
    print("WEEK 7 ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_week7_tasks()
