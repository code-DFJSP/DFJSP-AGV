import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import torch

class TF_DuelingNet(Model):
    def __init__(self, state_dim, action_dim):
        super(TF_DuelingNet, self).__init__()
        self.h1 = layers.Dense(32, name='l1', activation='relu')
        self.h2 = layers.Dense(32, name='l2', activation='relu')
        self.h3 = layers.Dense(32, name='l3', activation='relu')
        self.h4 = layers.Dense(32, name='l4', activation='relu')
        self.h5 = layers.Dense(32, name='l5', activation='relu')
        self.h6 = layers.Dense(32, name='l6', activation='relu')
        self.v = layers.Dense(1)
        self.a = layers.Dense(action_dim)

    def call(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        x = self.h5(x)
        x = self.h6(x)
        v = self.v(x)
        a = self.a(x)
        q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        return q
class DummyTorchCompat:
    def __init__(self, tf_model):
        self.tf_model = tf_model
    def parameters(self):
        return []
    def state_dict(self):
        return {}
    def load_state_dict(self, d):
        return
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x, dtype=np.float32)
        x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)
        out = self.tf_model(x_tf).numpy()
        return torch.from_numpy(out).float().to(x.device).requires_grad_(True)

class D3QN:
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.lr = float(cfg.get('lr', 1e-4))
        self.gamma = float(cfg.get('gamma', 0.99))
        self.batch_size = int(cfg.get('batch_size', 32))
        self.update_target_steps = int(cfg.get('update_target_steps', 200))
        self.buffer = []
        self.buffer_size = int(cfg.get('buffer_size', 50000))
        self.global_step = 0

        self.epsilon_init = float(cfg.get('epsilon', 1.0))
        self.epsilon_min = float(cfg.get('epsilon_min', 0.05))
        self.epsilon = self.epsilon_init

        self.online_tf = TF_DuelingNet(self.state_dim, self.action_dim)
        self.target_tf = TF_DuelingNet(self.state_dim, self.action_dim)
        _ = self.online_tf(tf.zeros((1, self.state_dim), dtype=tf.float32))
        _ = self.target_tf(tf.zeros((1, self.state_dim), dtype=tf.float32))
        self.target_tf.set_weights(self.online_tf.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        self.online = DummyTorchCompat(self.online_tf)
        self.target = DummyTorchCompat(self.target_tf)

    def select_action(self, state, n_step, N_iter):
        eps = max(0.0, self.epsilon_init - (self.epsilon_init / float(max(1, N_iter))) * float(n_step))
        self.epsilon = eps
        if n_step >= N_iter:
            s = np.array([state], dtype=np.float32)
            q = self.online_tf(s).numpy()
            return int(np.argmax(q[0]))
        if np.random.rand() < eps:
            return int(np.random.randint(self.action_dim))
        s = np.array([state], dtype=np.float32)
        q = self.online_tf(s).numpy()
        return int(np.argmax(q[0]))

    def store(self, s, a, r, s2, d):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((np.array(s, dtype=np.float32),
                            int(a),
                            float(r),
                            np.array(s2, dtype=np.float32),
                            float(d)))

    def replace_target(self):
        self.target_tf.get_layer(name='l1').set_weights(self.online_tf.get_layer(name='l1').get_weights())
        self.target_tf.get_layer(name='l2').set_weights(self.online_tf.get_layer(name='l2').get_weights())
        self.target_tf.get_layer(name='l3').set_weights(self.online_tf.get_layer(name='l3').get_weights())
        self.target_tf.get_layer(name='l4').set_weights(self.online_tf.get_layer(name='l4').get_weights())
        self.target_tf.get_layer(name='l5').set_weights(self.online_tf.get_layer(name='l5').get_weights())
        self.target_tf.get_layer(name='l6').set_weights(self.online_tf.get_layer(name='l6').get_weights())
        self.target_tf.set_weights(self.online_tf.get_weights())

    def replay(self):
        if len(self.buffer) < self.batch_size:
            return None
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()

        idxs = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32).reshape(-1,1)
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32).reshape(-1,1)
        with tf.GradientTape() as tape:
            q_vals = self.online_tf(states)  # (B, A)
            q_pred = tf.reduce_sum(q_vals * tf.one_hot(actions, self.action_dim), axis=1, keepdims=True)  # (B,1)
            q_next_online = self.online_tf(next_states)  # (B,A)
            next_actions = tf.argmax(q_next_online, axis=1)  # (B,)
            q_next_target = self.target_tf(next_states)  # (B,A)
            q_target_next = tf.reduce_sum(q_next_target * tf.one_hot(next_actions, self.action_dim), axis=1, keepdims=True)  # (B,1)
            q_target = rewards + (1.0 - dones) * self.gamma * q_target_next
            loss = tf.reduce_mean(tf.square(q_pred - q_target))
        grads = tape.gradient(loss, self.online_tf.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.online_tf.trainable_variables))
        self.global_step += 1
        return float(loss.numpy())
