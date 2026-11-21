import math
import random

# ==============================================================================
# UTILITY MATH FUNCTIONS
# ==============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

def dsigmoid(y):
    """Derivative given sigmoid output y."""
    return y * (1.0 - y)

def dtanh(y):
    """Derivative given tanh output y."""
    return 1.0 - y ** 2

def dot(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def matvec(M, v):
    """Matrix-vector multiplication M @ v."""
    return [sum(M[r][c] * v[c] for c in range(len(v))) for r in range(len(M))]

def vecadd(*vectors):
    """Element-wise addition of vectors."""
    return [sum(vals) for vals in zip(*vectors)]

def vecmul(v1, v2):
    """Element-wise multiplication of vectors."""
    return [a * b for a, b in zip(v1, v2)]

def rand_matrix(rows, cols, scale=0.1):
    return [[(random.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]


# ==============================================================================
# DAY 1: RECURRENT NEURAL NETWORK (Vanilla RNN from Scratch)
# ==============================================================================
class VanillaRNN:
    """
    Standard Elman Recurrent Neural Network.
    h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
    y_t = W_hy @ h_t + b_y
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weight matrices
        self.W_xh = rand_matrix(hidden_size, input_size)
        self.W_hh = rand_matrix(hidden_size, hidden_size)
        self.b_h = [0.0] * hidden_size

        self.W_hy = rand_matrix(output_size, hidden_size)
        self.b_y = [0.0] * output_size

    def forward(self, sequence_inputs, initial_hidden=None):
        """
        sequence_inputs: list of vectors [x_1, x_2, ..., x_T]
        Returns list of outputs [y_1, ..., y_T] and hidden states [h_1, ..., h_T]
        """
        T = len(sequence_inputs)
        h = initial_hidden if initial_hidden else [0.0] * self.hidden_size
        hidden_states = []
        outputs = []

        for t in range(T):
            xt = sequence_inputs[t]
            # h_t = tanh(W_xh @ xt + W_hh @ h_{t-1} + b_h)
            h_in = vecadd(matvec(self.W_xh, xt), matvec(self.W_hh, h), self.b_h)
            h = [math.tanh(val) for val in h_in]
            hidden_states.append(h)

            # Output projection
            yt = vecadd(matvec(self.W_hy, h), self.b_y)
            outputs.append(yt)

        return outputs, hidden_states

    @staticmethod
    def explain_vanishing_gradients():
        return (
            "RNN Vanishing/Exploding Gradient Problem:\n"
            "During Backpropagation Through Time (BPTT), the gradient dh_0/dh_T involves repeated "
            "matrix multiplications by W_hh^T and derivatives of tanh (which are in (0, 1]).\n"
            "If the largest eigenvalue of W_hh is < 1, gradients vanish exponentially as sequence length T increases,\n"
            "making it impossible for standard RNNs to learn long-term dependencies.\n"
            "LSTMs and GRUs solve this by using additive memory cell updates and gating mechanisms."
        )


# ==============================================================================
# DAY 2 & DAY 3: LONG SHORT-TERM MEMORY (LSTM) & CODE PRACTICE
# ==============================================================================
class LSTMCell:
    """
    Single LSTM cell implementing forget, input, cell, and output gates.
    f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
    i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
    c_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c)
    C_t = f_t * C_{t-1} + i_t * c_tilde
    o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
    h_t = o_t * tanh(C_t)
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        # Gate weights
        self.W_f = rand_matrix(hidden_size, concat_size)
        self.b_f = [1.0] * hidden_size  # Forget gate bias initialized to 1 for long-term retention

        self.W_i = rand_matrix(hidden_size, concat_size)
        self.b_i = [0.0] * hidden_size

        self.W_c = rand_matrix(hidden_size, concat_size)
        self.b_c = [0.0] * hidden_size

        self.W_o = rand_matrix(hidden_size, concat_size)
        self.b_o = [0.0] * hidden_size

    def step(self, xt, prev_h, prev_c):
        # Concatenate [h_{t-1}, x_t]
        concat_input = prev_h + xt

        # Forget gate
        f_t = [sigmoid(val) for val in vecadd(matvec(self.W_f, concat_input), self.b_f)]
        # Input gate
        i_t = [sigmoid(val) for val in vecadd(matvec(self.W_i, concat_input), self.b_i)]
        # Candidate memory
        c_tilde = [math.tanh(val) for val in vecadd(matvec(self.W_c, concat_input), self.b_c)]

        # Cell state update
        curr_c = vecadd(vecmul(f_t, prev_c), vecmul(i_t, c_tilde))

        # Output gate
        o_t = [sigmoid(val) for val in vecadd(matvec(self.W_o, concat_input), self.b_o)]

        # Hidden state
        curr_h = vecmul(o_t, [math.tanh(c_val) for c_val in curr_c])

        cache = {
            "f_t": f_t, "i_t": i_t, "c_tilde": c_tilde,
            "o_t": o_t, "curr_c": curr_c, "curr_h": curr_h
        }
        return curr_h, curr_c, cache


class LSTMNetwork:
    """Multi-step LSTM model with dense readout layer."""
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W_out = rand_matrix(output_size, hidden_size)
        self.b_out = [0.0] * output_size

    def forward(self, sequence):
        """Processes sequence through LSTM and outputs prediction for each step."""
        h = [0.0] * self.hidden_size
        c = [0.0] * self.hidden_size
        hidden_history = []
        cell_history = []
        outputs = []

        for xt in sequence:
            h, c, _ = self.cell.step(xt, h, c)
            hidden_history.append(h)
            cell_history.append(c)
            # Final linear projection
            yt = vecadd(matvec(self.W_out, h), self.b_out)
            outputs.append(yt)

        return outputs, hidden_history, cell_history


# ==============================================================================
# DAY 4 & DAY 5: GATED RECURRENT UNIT (GRU) & BENCHMARKING
# ==============================================================================
class GRUCell:
    """
    Gated Recurrent Unit (GRU).
    r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)  -- Reset gate
    z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)  -- Update gate
    h_tilde = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)  -- Candidate hidden
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde  -- Final hidden state
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        self.W_r = rand_matrix(hidden_size, concat_size)
        self.b_r = [0.0] * hidden_size

        self.W_z = rand_matrix(hidden_size, concat_size)
        self.b_z = [0.0] * hidden_size

        self.W_h = rand_matrix(hidden_size, concat_size)
        self.b_h = [0.0] * hidden_size

    def step(self, xt, prev_h):
        concat_input = prev_h + xt

        # Reset and Update gates
        r_t = [sigmoid(val) for val in vecadd(matvec(self.W_r, concat_input), self.b_r)]
        z_t = [sigmoid(val) for val in vecadd(matvec(self.W_z, concat_input), self.b_z)]

        # Reset applied to previous hidden state
        reset_h = vecmul(r_t, prev_h)
        cand_input = reset_h + xt
        h_tilde = [math.tanh(val) for val in vecadd(matvec(self.W_h, cand_input), self.b_h)]

        # Linear interpolation between previous and candidate state
        curr_h = [
            (1.0 - z) * h_prev + z * h_cand
            for z, h_prev, h_cand in zip(z_t, prev_h, h_tilde)
        ]

        return curr_h, {"r_t": r_t, "z_t": z_t, "h_tilde": h_tilde}


class GRUNetwork:
    """Full sequence GRU with output readout."""
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W_out = rand_matrix(output_size, hidden_size)
        self.b_out = [0.0] * output_size

    def forward(self, sequence):
        h = [0.0] * self.hidden_size
        outputs = []
        hidden_history = []

        for xt in sequence:
            h, _ = self.cell.step(xt, h)
            hidden_history.append(h)
            yt = vecadd(matvec(self.W_out, h), self.b_out)
            outputs.append(yt)

        return outputs, hidden_history


# ==============================================================================
# SEQUENCE PREDICTION / SENTIMENT CODE PRACTICE
# ==============================================================================
def sequence_prediction_practice():
    """
    Demonstrates practical sequence forecasting:
    Given a sine wave sequence [x_1, ..., x_t], predict the next step x_{t+1}.
    """
    # Generate 15-step synthetic temporal sequence
    seq_len = 10
    sequence = [[math.sin(0.3 * t) + random.uniform(-0.05, 0.05)] for t in range(seq_len)]

    # 1. Evaluate with Vanilla RNN
    rnn = VanillaRNN(input_size=1, hidden_size=6, output_size=1)
    rnn_outputs, _ = rnn.forward(sequence)

    # 2. Evaluate with LSTM
    lstm = LSTMNetwork(input_size=1, hidden_size=6, output_size=1)
    lstm_outputs, _, _ = lstm.forward(sequence)

    # 3. Evaluate with GRU
    gru = GRUNetwork(input_size=1, hidden_size=6, output_size=1)
    gru_outputs, _ = gru.forward(sequence)

    return {
        "inputs": [round(s[0], 3) for s in sequence],
        "rnn_last_pred": round(rnn_outputs[-1][0], 3),
        "lstm_last_pred": round(lstm_outputs[-1][0], 3),
        "gru_last_pred": round(gru_outputs[-1][0], 3),
    }


def architectural_comparison():
    """Returns comparative analysis of RNN vs LSTM vs GRU."""
    return {
        "Vanilla RNN": {
            "Gates": 0,
            "Parameters": "O(H^2 + H*D)",
            "Memory Cell": "No (Hidden state only)",
            "Pros": "Fastest training, fewest parameters",
            "Cons": "Severe vanishing/exploding gradients on long sequences"
        },
        "LSTM": {
            "Gates": 3, # Forget, Input, Output
            "Parameters": "O(4 * (H^2 + H*D))",
            "Memory Cell": "Yes (Explicit cell state C_t)",
            "Pros": "Excellent long-term dependency capture, stable gradients via additive cell",
            "Cons": "Higher computational cost, more parameters"
        },
        "GRU": {
            "Gates": 2, # Reset, Update
            "Parameters": "O(3 * (H^2 + H*D))",
            "Memory Cell": "No (Modulates hidden state directly)",
            "Pros": "Fewer parameters than LSTM (~25% less), faster training, comparable accuracy",
            "Cons": "Slightly less representational capacity on extremely complex sequences"
        }
    }


# ==============================================================================
# MAIN TEST & DEMONSTRATION SUITE
# ==============================================================================
def run_all_week8_tasks():
    print("=" * 80)
    print("RUNNING WEEK 8 TASKS: DEEP LEARNING II - RECURRENT NETWORKS (RNN, LSTM, GRU)")
    print("=" * 80)

    # Day 1: Vanilla RNN
    print("\n[Day 1] Vanilla Recurrent Neural Network (RNN):")
    rnn = VanillaRNN(input_size=3, hidden_size=4, output_size=2)
    sample_seq = [[0.5, 0.2, -0.1], [0.1, 0.9, 0.4], [-0.3, 0.4, 0.8]]
    outputs, hidden_states = rnn.forward(sample_seq)
    print(f"  Processed {len(sample_seq)} timesteps.")
    print(f"  Final Output Vector (t={len(sample_seq)}): {[round(v, 4) for v in outputs[-1]]}")
    print(f"  Final Hidden State: {[round(h, 4) for h in hidden_states[-1]]}")
    print("\n  Theoretical Insight:")
    print("  " + VanillaRNN.explain_vanishing_gradients().replace("\n", "\n  "))

    # Day 2 & Day 3: LSTM Cell & Network
    print("\n[Day 2 & Day 3] Long Short-Term Memory Network (LSTM):")
    lstm = LSTMNetwork(input_size=3, hidden_size=4, output_size=1)
    lstm_out, lstm_h, lstm_c = lstm.forward(sample_seq)
    print(f"  Processed sequence of length {len(sample_seq)} through LSTM.")
    print(f"  Final Cell State C_T:   {[round(c, 4) for c in lstm_c[-1]]}")
    print(f"  Final Hidden State h_T: {[round(h, 4) for h in lstm_h[-1]]}")
    print(f"  Final Prediction y_T:   {round(lstm_out[-1][0], 4)}")

    # Day 4: Gated Recurrent Unit (GRU)
    print("\n[Day 4] Gated Recurrent Unit (GRU):")
    gru = GRUNetwork(input_size=3, hidden_size=4, output_size=1)
    gru_out, gru_h = gru.forward(sample_seq)
    print(f"  Final GRU Hidden State h_T: {[round(h, 4) for h in gru_h[-1]]}")
    print(f"  Final GRU Prediction y_T:   {round(gru_out[-1][0], 4)}")

    # Day 5: Code Practice & Comparison
    print("\n[Day 5] Code Practice: Sequence Prediction & Architectural Benchmark:")
    res = sequence_prediction_practice()
    print(f"  Input Sequence Samples: {res['inputs'][:5]} ...")
    print(f"  RNN Prediction:  {res['rnn_last_pred']}")
    print(f"  LSTM Prediction: {res['lstm_last_pred']}")
    print(f"  GRU Prediction:  {res['gru_last_pred']}")

    print("\n  Architectural Comparison Matrix:")
    comp = architectural_comparison()
    for model_name, info in comp.items():
        print(f"\n  * Model: {model_name}")
        for k, v in info.items():
            print(f"      - {k}: {v}")

    print("=" * 80)
    print("WEEK 8 ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_week8_tasks()
