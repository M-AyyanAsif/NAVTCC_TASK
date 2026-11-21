import math
import random
import json
import re
from collections import Counter

# ==============================================================================
# UTILITY MATH & VECTOR OPERATIONS
# ==============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

def softmax(vector):
    max_v = max(vector)
    exps = [math.exp(v - max_v) for v in vector]
    total = sum(exps)
    return [e / total for e in exps]

def rand_matrix(rows, cols, scale=0.1):
    return [[(random.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]

def matvec(M, v):
    return [sum(M[r][c] * v[c] for c in range(len(v))) for r in range(len(M))]

def vecadd(*vectors):
    return [sum(vals) for vals in zip(*vectors)]

def dot(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


# ==============================================================================
# DAY 1: BI-DIRECTIONAL LSTM / RNN IN SEQUENCE MODELS
# ==============================================================================
class BiDirectionalRNN:
    """
    Bidirectional Recurrent Neural Network.
    Computes forward hidden sequence (t=1..T) and backward hidden sequence (t=T..1).
    Concatenates [h_forward_t; h_backward_t] to capture past and future context.
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forward parameters
        self.W_xh_fwd = rand_matrix(hidden_size, input_size)
        self.W_hh_fwd = rand_matrix(hidden_size, hidden_size)
        self.b_fwd = [0.0] * hidden_size

        # Backward parameters
        self.W_xh_bwd = rand_matrix(hidden_size, input_size)
        self.W_hh_bwd = rand_matrix(hidden_size, hidden_size)
        self.b_bwd = [0.0] * hidden_size

    def forward(self, sequence):
        """
        sequence: list of vectors of length T
        Returns: list of concatenated hidden states [h_t] of dimension 2 * hidden_size
        """
        T = len(sequence)

        # 1. Forward Pass (t = 0 ... T-1)
        h_fwd = [0.0] * self.hidden_size
        fwd_states = []
        for t in range(T):
            xt = sequence[t]
            h_in = vecadd(matvec(self.W_xh_fwd, xt), matvec(self.W_hh_fwd, h_fwd), self.b_fwd)
            h_fwd = [math.tanh(val) for val in h_in]
            fwd_states.append(h_fwd)

        # 2. Backward Pass (t = T-1 ... 0)
        h_bwd = [0.0] * self.hidden_size
        bwd_states = [None] * T
        for t in reversed(range(T)):
            xt = sequence[t]
            h_in = vecadd(matvec(self.W_xh_bwd, xt), matvec(self.W_hh_bwd, h_bwd), self.b_bwd)
            h_bwd = [math.tanh(val) for val in h_in]
            bwd_states[t] = h_bwd

        # 3. Concatenate forward and backward representations
        bi_states = [fwd + bwd for fwd, bwd in zip(fwd_states, bwd_states)]
        return bi_states


# ==============================================================================
# DAY 2 & DAY 3: ATTENTION MECHANISMS (Bahdanau & Scaled Dot-Product)
# ==============================================================================
class BahdanauAdditiveAttention:
    """
    Bahdanau (Additive) Attention Mechanism:
    score(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j)
    alpha_{i,j} = softmax(score)
    context_i = sum_j (alpha_{i,j} * h_j)
    """
    def __init__(self, hidden_dim, query_dim):
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        att_dim = hidden_dim

        self.W_a = rand_matrix(att_dim, query_dim)
        self.U_a = rand_matrix(att_dim, hidden_dim)
        self.v_a = [(random.random() * 2 - 1) * 0.1 for _ in range(att_dim)]

    def compute_attention(self, query_state, encoder_hidden_states):
        """
        query_state: vector s_{i-1}
        encoder_hidden_states: list of vectors [h_1, ..., h_T]
        """
        W_query = matvec(self.W_a, query_state)
        scores = []

        for h_j in encoder_hidden_states:
            U_h = matvec(self.U_a, h_j)
            combined = [math.tanh(q + u) for q, u in zip(W_query, U_h)]
            score = dot(self.v_a, combined)
            scores.append(score)

        # Softmax weights (alpha)
        attention_weights = softmax(scores)

        # Context vector c = sum(alpha_j * h_j)
        context_vector = [0.0] * self.hidden_dim
        for alpha, h_j in zip(attention_weights, encoder_hidden_states):
            for k in range(self.hidden_dim):
                context_vector[k] += alpha * h_j[k]

        return context_vector, attention_weights


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention (Vaswani et al. - Attention Is All You Need):
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    @staticmethod
    def calculate(queries, keys, values):
        """
        queries: list of Q vectors [num_q, d_k]
        keys:    list of K vectors [num_k, d_k]
        values:  list of V vectors [num_k, d_v]
        """
        d_k = len(keys[0])
        scale = math.sqrt(d_k)
        num_q = len(queries)
        num_k = len(keys)

        # Compute Q @ K^T / sqrt(d_k)
        attention_matrices = []
        context_vectors = []

        for q in queries:
            raw_scores = [dot(q, k) / scale for k in keys]
            weights = softmax(raw_scores)
            attention_matrices.append(weights)

            # Compute weighted sum of values
            d_v = len(values[0])
            context = [0.0] * d_v
            for w, v in zip(weights, values):
                for col in range(d_v):
                    context[col] += w * v[col]
            context_vectors.append(context)

        return context_vectors, attention_matrices


# ==============================================================================
# DAY 4 & DAY 5: EMPLOYABLE INDUSTRY PROJECT (CAPSTONE ASSIGNMENT)
# "Deep Learning Intent & Sentiment Intelligence Engine with Bi-LSTM + Attention"
# ==============================================================================
class IntentSentimentPipeline:
    """
    An End-to-End Deep Learning Natural Language Understanding (NLU) System:
      1. Tokenization & Vocabulary mapping
      2. Embedding Layer
      3. Bidirectional Sequence Encoder (Bi-RNN/LSTM)
      4. Self-Attention pooling layer
      5. Classification Heads for Intent & Sentiment
      6. Production-Ready Inference API
    """
    def __init__(self, embedding_dim=8, hidden_dim=8):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = {}
        self.inv_vocab = {}
        self.embedding_table = {}

        # Intent labels & Sentiment labels
        self.intent_labels = ["Billing_Query", "Technical_Support", "Account_Update", "Product_Inquiry"]
        self.sentiment_labels = ["Negative", "Neutral", "Positive"]

        # Deep Learning Components
        self.bi_rnn = BiDirectionalRNN(input_size=embedding_dim, hidden_size=hidden_dim)
        # Attends over bidirectional hidden dimension (2 * hidden_dim)
        self.att_query = rand_matrix(1, 2 * hidden_dim)[0]

        # Classification Heads (Dense projections)
        self.W_intent = rand_matrix(len(self.intent_labels), 2 * hidden_dim)
        self.b_intent = [0.0] * len(self.intent_labels)

        self.W_sent = rand_matrix(len(self.sentiment_labels), 2 * hidden_dim)
        self.b_sent = [0.0] * len(self.sentiment_labels)

    def build_vocab(self, corpus):
        tokens = [w for doc in corpus for w in self._tokenize(doc)]
        counts = Counter(tokens)
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for idx, (word, _) in enumerate(counts.most_common(), start=2):
            self.vocab[word] = idx
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Initialize embeddings
        for idx in range(len(self.vocab)):
            self.embedding_table[idx] = [(random.random() * 2 - 1) * 0.2 for _ in range(self.embedding_dim)]

    def _tokenize(self, text):
        return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())

    def _encode_sequence(self, text, max_len=8):
        tokens = self._tokenize(text)
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens][:max_len]
        while len(ids) < max_len:
            ids.append(self.vocab["<PAD>"])
        return [self.embedding_table[id_val] for id_val in ids], tokens

    def forward(self, text):
        """Complete forward pass: Text -> Embed -> Bi-RNN -> Attention -> Prediction."""
        embedded_seq, tokens = self._encode_sequence(text)

        # 1. Bidirectional encoding
        bi_states = self.bi_rnn.forward(embedded_seq)

        # 2. Self-Attention Pooling
        # score = h_t dot att_query
        scores = [dot(h_t, self.att_query) for h_t in bi_states]
        att_weights = softmax(scores)

        pooled_context = [0.0] * (2 * self.hidden_dim)
        for w, h_t in zip(att_weights, bi_states):
            for i in range(len(pooled_context)):
                pooled_context[i] += w * h_t[i]

        # 3. Predict Intent & Sentiment
        intent_logits = vecadd(matvec(self.W_intent, pooled_context), self.b_intent)
        intent_probs = softmax(intent_logits)

        sent_logits = vecadd(matvec(self.W_sent, pooled_context), self.b_sent)
        sent_probs = softmax(sent_logits)

        intent_idx = intent_probs.index(max(intent_probs))
        sent_idx = sent_probs.index(max(sent_probs))

        return {
            "text": text,
            "tokens": tokens,
            "predicted_intent": self.intent_labels[intent_idx],
            "intent_confidence": round(intent_probs[intent_idx], 4),
            "predicted_sentiment": self.sentiment_labels[sent_idx],
            "sentiment_confidence": round(sent_probs[sent_idx], 4),
            "attention_weights": [round(w, 4) for w in att_weights[:len(tokens)]]
        }

    def train_mock(self, training_data, epochs=30):
        """Simulate optimization and alignment for demonstration."""
        for _ in range(epochs):
            for item in training_data:
                # Forward pass
                _ = self.forward(item["text"])

    def predict_api(self, json_request_payload):
        """REST/Microservice endpoint simulation taking JSON string and returning JSON."""
        data = json.loads(json_request_payload)
        text = data.get("query", "")
        result = self.forward(text)
        return json.dumps({
            "status": "success",
            "code": 200,
            "analysis": result
        }, indent=2)


# ==============================================================================
# MAIN TEST & DEMONSTRATION SUITE
# ==============================================================================
def run_all_week10_tasks():
    print("=" * 80)
    print("RUNNING WEEK 10 TASKS: BI-DIRECTIONAL MODELS, ATTENTION & EMPLOYABLE CAPSTONE")
    print("=" * 80)

    # 1. Day 1: Bi-Directional RNN
    print("\n[Day 1] Bi-Directional Recurrent Neural Network (Forward + Backward):")
    bi_rnn = BiDirectionalRNN(input_size=4, hidden_size=3)
    sample_seq = [[0.1, 0.5, -0.2, 0.3], [-0.4, 0.2, 0.7, 0.1], [0.8, -0.1, 0.3, 0.5]]
    bi_hidden = bi_rnn.forward(sample_seq)
    print(f"  Input Sequence: {len(sample_seq)} tokens x 4 features")
    print(f"  Bi-Directional Hidden State Output Dimension: {len(bi_hidden[0])} (3 fwd + 3 bwd)")
    print(f"  Hidden State at t=0: {[round(v, 4) for v in bi_hidden[0]]}")
    print(f"  Hidden State at t=2: {[round(v, 4) for v in bi_hidden[-1]]}")

    # 2. Day 2 & 3: Attention Mechanisms
    print("\n[Day 2 & Day 3] Attention Mechanisms (Bahdanau Additive & Scaled Dot-Product):")
    # A. Bahdanau Additive Attention
    query_s = [0.2, -0.5, 0.4]
    encoder_states = [[0.1, 0.9, -0.3], [0.8, -0.2, 0.4], [-0.5, 0.1, 0.6]]
    bahdanau = BahdanauAdditiveAttention(hidden_dim=3, query_dim=3)
    context_vec, att_weights = bahdanau.compute_attention(query_s, encoder_states)
    print(f"  * Bahdanau Additive Attention Weights: {[round(w, 4) for w in att_weights]}")
    print(f"  * Generated Context Vector:            {[round(c, 4) for c in context_vec]}")

    # B. Scaled Dot-Product Attention (Transformer Style)
    queries = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.2]]
    keys = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0]]
    values = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    contexts, att_matrices = ScaledDotProductAttention.calculate(queries, keys, values)
    print(f"\n  * Scaled Dot-Product Attention:")
    print(f"      Query 1 Weights across 3 Keys: {[round(w, 4) for w in att_matrices[0]]}")
    print(f"      Query 1 Output Representation: {[round(c, 4) for c in contexts[0]]}")

    # 3. Day 4 & 5: Employable Project (Capstone)
    print("\n[Day 4 & Day 5] Employable Industry Project / Final Year Project (FYP):")
    print("  Title: 'Deep Learning Intent & Sentiment Intelligence Engine with Bi-LSTM + Attention'")

    training_samples = [
        {"text": "Why was my credit card charged twice this month?", "intent": "Billing_Query", "sent": "Negative"},
        {"text": "My application keeps crashing whenever I log in.", "intent": "Technical_Support", "sent": "Negative"},
        {"text": "I would like to update my primary email address.", "intent": "Account_Update", "sent": "Neutral"},
        {"text": "What are the new features of your premium subscription?", "intent": "Product_Inquiry", "sent": "Positive"},
        {"text": "Thank you for the prompt resolution of my invoice!", "intent": "Billing_Query", "sent": "Positive"}
    ]

    corpus = [item["text"] for item in training_samples]
    pipeline = IntentSentimentPipeline(embedding_dim=8, hidden_dim=8)
    pipeline.build_vocab(corpus)
    pipeline.train_mock(training_samples)

    # Test sample inference via simulated REST API endpoint
    api_request = json.dumps({"query": "Why was my card billed twice? Please refund me immediately!"})
    api_response = pipeline.predict_api(api_request)
    print("\n  Simulated Production Microservice API Request/Response:")
    print(api_response)

    print("=" * 80)
    print("WEEK 10 ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_week10_tasks()
