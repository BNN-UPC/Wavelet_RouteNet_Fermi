"""
Copyright 2024 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf


class RouteNet_Fermi(tf.keras.Model):
    """
    RouteNet_Fermi adapted to this repository's datasets. Takes distribution parameters
    for poisson and on-off
    """

    z_scores_fields = {
        "flow_traffic",
        "flow_packets",
        "flow_lambda",
        "flow_ON_bits_rate",
        "flow_ON_time",
        "flow_OFF_time",
    }
    name = "RouteNet_Fermi"

    def __init__(self, z_scores=None, log=False, max_num_dist=3, **kwargs):
        super().__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file

        self.max_buffer_types = 2
        self.max_num_dist = max_num_dist

        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

        self.z_scores = z_scores
        assert (
            type(z_scores) == dict
            and all(kk in self.z_scores for kk in self.z_scores_fields)
            and all(len(val) == 2 for val in self.z_scores.values())
        ), "overriden z_scores dict is not valid!"
        self.log = log

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.link_update = tf.keras.layers.GRUCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.GRUCell(self.queue_state_dim)

        self.path_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=6 + max_num_dist),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.queue_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.max_buffer_types),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=1),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    int(self.link_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    int(self.path_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(1),
            ],
            name="PathReadout",
        )

    @tf.function
    def call(self, inputs):
        traffic = inputs["flow_traffic"]
        packets = inputs["flow_packets"]
        length = tf.squeeze(inputs["flow_length"], 1)

        flow_dist = inputs["flow_time_dist"]
        flow_lambda = inputs["flow_lambda"]
        flow_ON_bits_rate = inputs["flow_ON_bits_rate"]
        flow_ON_time = inputs["flow_ON_time"]
        flow_OFF_time = inputs["flow_OFF_time"]

        capacity = inputs["link_capacity"]

        buffer_type = inputs["buffer_type"]

        queue_to_path = link_to_path = inputs["link_to_path"]
        path_to_link = path_to_queue = inputs["path_to_link"]
        queue_to_link = inputs["queue_to_link"]

        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity

        pkt_size = traffic / packets

        # Initialize the initial hidden state for links
        path_state = self.path_embedding(
            tf.concat(
                [
                    (traffic - self.z_scores["flow_traffic"][0])
                    / self.z_scores["flow_traffic"][1],
                    (packets - self.z_scores["flow_packets"][0])
                    / self.z_scores["flow_packets"][1],
                    tf.one_hot(tf.squeeze(flow_dist, 1), self.max_num_dist),
                    (flow_lambda - self.z_scores["flow_lambda"][0])
                    / self.z_scores["flow_lambda"][1],
                    (flow_ON_bits_rate - self.z_scores["flow_ON_bits_rate"][0])
                    / self.z_scores["flow_ON_bits_rate"][1],
                    (flow_ON_time - self.z_scores["flow_ON_time"][0])
                    / self.z_scores["flow_ON_time"][1],
                    (flow_OFF_time - self.z_scores["flow_OFF_time"][0])
                    / self.z_scores["flow_OFF_time"][1],
                ],
                axis=1,
            )
        )

        # Initialize the initial hidden state for paths
        link_state = self.link_embedding(tf.concat([load], axis=1))

        # Initialize the initial hidden state for paths

        queue_state = self.queue_embedding(
            tf.squeeze(tf.one_hot(buffer_type, self.max_buffer_types), 1)
        )

        # Iterate t times doing the message passing
        for it in range(self.iterations):
            ###################
            #  LINK AND QUEUE #
            #     TO PATH     #
            ###################
            queue_gather = tf.gather(queue_state, queue_to_path)
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            path_update_rnn = tf.keras.layers.RNN(
                self.path_update, return_sequences=True, return_state=True
            )
            previous_path_state = path_state

            path_state_sequence, path_state = path_update_rnn(
                tf.concat([queue_gather, link_gather], axis=2), initial_state=path_state
            )

            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            ###################
            #  PATH TO QUEUE  #
            ###################
            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            ###################
            #  QUEUE TO LINK  #
            ###################
            queue_gather = tf.gather(queue_state, queue_to_link)

            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(queue_gather, initial_state=link_state)

        capacity_gather = tf.gather(capacity * 1e9, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)
        trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

        if self.log:
            return tf.math.log(queue_delay + trans_delay)
        return queue_delay + trans_delay


class RouteNet_Fermi_wavelet_single_level(tf.keras.Model):
    """
    RouteNet_Fermi adapted to work with wavelet data. This is done by codifying the
    wavelet sequence using a RNN. Accepts only one single level of wavelet
    decomposition.
    """

    name = "RouteNet_Fermi_wavelet_single_level"

    def __init__(self, wt_field, log=False, **kwargs):
        super().__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file

        self.max_buffer_types = 2
        self.wt_field = wt_field

        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

        self.log = log

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.link_update = tf.keras.layers.GRUCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.GRUCell(self.queue_state_dim)

        self.path_embedding = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(self.path_state_dim)
        )

        self.queue_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.max_buffer_types),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=1),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    int(self.link_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    int(self.path_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(1),
            ],
            name="PathReadout",
        )

    @tf.function
    def call(self, inputs):
        traffic = inputs["flow_traffic"]
        packets = inputs["flow_packets"]
        length = tf.squeeze(inputs["flow_length"], 1)

        capacity = inputs["link_capacity"]

        buffer_type = inputs["buffer_type"]

        queue_to_path = link_to_path = inputs["link_to_path"]
        path_to_link = path_to_queue = inputs["path_to_link"]
        queue_to_link = inputs["queue_to_link"]

        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity

        pkt_size = traffic / packets

        # Initialize the initial hidden state for paths
        path_state = self.path_embedding(
            tf.concat(
                [
                    inputs[f"flow_packet_size_wt_{self.wt_field}"],
                    inputs[f"flow_ipg_wt_{self.wt_field}"],
                ],
                axis=2,
            )
        )

        # Initialize the initial hidden state for paths
        link_state = self.link_embedding(tf.concat([load], axis=1))

        # Initialize the initial hidden state for queues
        queue_state = self.queue_embedding(
            tf.squeeze(tf.one_hot(buffer_type, self.max_buffer_types), 1)
        )

        # Iterate t times doing the message passing
        for it in range(self.iterations):
            ###################
            #  LINK AND QUEUE #
            #     TO PATH     #
            ###################
            queue_gather = tf.gather(queue_state, queue_to_path)
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            path_update_rnn = tf.keras.layers.RNN(
                self.path_update, return_sequences=True, return_state=True
            )
            previous_path_state = path_state

            path_state_sequence, path_state = path_update_rnn(
                tf.concat([queue_gather, link_gather], axis=2), initial_state=path_state
            )

            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            ###################
            #  PATH TO QUEUE  #
            ###################
            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            ###################
            #  QUEUE TO LINK  #
            ###################
            queue_gather = tf.gather(queue_state, queue_to_link)

            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(queue_gather, initial_state=link_state)

        capacity_gather = tf.gather(capacity * 1e9, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)
        trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

        if self.log:
            return tf.math.log(queue_delay + trans_delay)
        return queue_delay + trans_delay


class RouteNet_Fermi_wavelet_multiple_level(tf.keras.Model):
    """
    RouteNet_Fermi adapted to work with wavelet data. This is done by codifying the
    wavelet sequence using a RNN. Accepts multiple levels of wavelet decomposition. Each
    is processed by its own RNN, then results are concatenated and passed through an
    MLP to extract is correct size.
    """

    name = "RouteNet_Fermi_wavelet_multiple_level"

    def __init__(self, wt_fields, flow_rnn_inner_dim=[], log=False, **kwargs):
        super().__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file

        self.max_buffer_types = 2

        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

        self.wt_fields = [
            (f"flow_ipg_wt_{field}", f"flow_packet_size_wt_{field}")
            for field in wt_fields
        ]

        self.log = log

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.link_update = tf.keras.layers.GRUCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.GRUCell(self.queue_state_dim)

        self.path_embedding_rnns = [
            tf.keras.layers.RNN(
                tf.keras.layers.StackedRNNCells(
                    [
                        tf.keras.layers.GRUCell(dim)
                        for dim in flow_rnn_inner_dim + [self.path_state_dim]
                    ]
                )
            )
            for _ in wt_fields
        ]
        self.path_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(len(wt_fields) * self.path_state_dim)),
                tf.keras.layers.Dense(
                    self.path_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.queue_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.max_buffer_types),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=1),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(
                    int(self.link_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    int(self.path_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(1),
            ],
            name="PathReadout",
        )

    @tf.function
    def call(self, inputs):
        traffic = inputs["flow_traffic"]
        packets = inputs["flow_packets"]
        length = tf.squeeze(inputs["flow_length"], 1)

        capacity = inputs["link_capacity"]

        buffer_type = inputs["buffer_type"]

        queue_to_path = link_to_path = inputs["link_to_path"]
        path_to_link = path_to_queue = inputs["path_to_link"]
        queue_to_link = inputs["queue_to_link"]

        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity

        pkt_size = traffic / packets

        # Initialize the initial hidden state for paths
        encoded_pkt_states = [
            field_rnn(tf.concat([inputs[ff] for ff in fields], axis=2))
            for fields, field_rnn in zip(self.wt_fields, self.path_embedding_rnns)
        ]
        path_state = self.path_embedding(tf.concat(encoded_pkt_states, axis=1))

        # Initialize the initial hidden state for paths
        link_state = self.link_embedding(tf.concat([load], axis=1))

        # Initialize the initial hidden state for paths

        queue_state = self.queue_embedding(
            tf.squeeze(tf.one_hot(buffer_type, self.max_buffer_types), 1)
        )

        # Iterate t times doing the message passing
        for it in range(self.iterations):
            ###################
            #  LINK AND QUEUE #
            #     TO PATH     #
            ###################
            queue_gather = tf.gather(queue_state, queue_to_path)
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            path_update_rnn = tf.keras.layers.RNN(
                self.path_update, return_sequences=True, return_state=True
            )
            previous_path_state = path_state

            path_state_sequence, path_state = path_update_rnn(
                tf.concat([queue_gather, link_gather], axis=2), initial_state=path_state
            )

            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            ###################
            #  PATH TO QUEUE  #
            ###################
            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            ###################
            #  QUEUE TO LINK  #
            ###################
            queue_gather = tf.gather(queue_state, queue_to_link)

            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(queue_gather, initial_state=link_state)

        capacity_gather = tf.gather(capacity * 1e9, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)
        trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

        if self.log:
            return tf.math.log(queue_delay + trans_delay)
        return queue_delay + trans_delay
