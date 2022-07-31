import itertools
from typing import Tuple

import tensorflow as tf
import tensorflow.keras as K
from keras.layers import concatenate
from keras.models import Model


def build_dense_layer(inputs, activation: str, units: int, name: str, seed: int):
    initializer = K.initializers.GlorotUniform(seed)

    x = K.layers.Dense(units, kernel_initializer=initializer, name=f"{name}_dense")(inputs)
    x = K.layers.Activation(activation, name=f"{name}_activation")(x)

    return x


def build_mlp_block(inputs, shape: Tuple[int], activation: str, activation_head: str, name: str, seed: int):
    for layer_no, units in enumerate(shape):
        x = build_dense_layer(
            inputs=inputs if layer_no == 0 else x,
            activation=activation if layer_no < len(shape) - 1 else activation_head,
            units=units,
            name=f"{name}_{layer_no}",
            seed=seed,
        )

    return x


def build_core(inputs, shape: Tuple[int], activation: str, seed: int):

    core_model = build_mlp_block(
        inputs=inputs,
        shape=shape,
        activation=activation,
        activation_head=activation,
        seed=seed,
        name="core_model",
    )

    return core_model


def build_expert(inputs, shape: Tuple[int], activation: str, name: str, seed: int):

    expert_model = build_mlp_block(
        inputs=inputs,
        shape=shape,
        activation=activation,
        activation_head=None,
        seed=seed,
        name=name,
    )

    return expert_model


def build_gate(inputs, experts_num: int, name: str, seed: int):
    gate = build_mlp_block(
        inputs, shape=(experts_num,), activation=None, activation_head="softmax", name=name, seed=seed
    )

    return gate


def compile_model(model: Model, learning_rate: float, loss: str):
    model.compile(optimizer=K.optimizers.Adam(learning_rate=learning_rate), loss=loss)


def buld_mmoe_model(
    n_inputs: int,
    core_shape: Tuple[int],
    core_activation: str,
    experts_num: int,
    experts_shape: Tuple[int],
    experts_activation: str,
    tasks_num: int,
):
    seed_generator = itertools.count()

    inputs = K.Input(shape=n_inputs, name="inputs")
    core_model = build_core(
        inputs=inputs, shape=core_shape, activation=core_activation, seed=next(seed_generator)
    )
    experts = [
        build_expert(
            inputs=core_model,
            shape=experts_shape,
            activation=experts_activation,
            name=f"expert_{i}",
            seed=next(seed_generator),
        )
        for i in range(experts_num)
    ]

    gates = [
        build_gate(inputs=core_model, name=f"gate_{i}", experts_num=experts_num, seed=next(seed_generator))
        for i in range(tasks_num)
    ]

    # prepare input for task blocks
    experts = tf.stack(experts, axis=1)  # -> (batch_size, experts_num, experts_out)
    gates = tf.stack(gates, axis=1)  # -> (batch_size, tasks_num, experts_num)

    # for each task average experts_out with relevant gate
    # -> (batch_size, tasks_num, experts_out)
    weighted_experts_out = tf.einsum("bij, bki -> bkj", experts, gates)
    weighted_experts_out = [weighted_experts_out[:, task_num, :] for task_num in range(tasks_num)]

    mmoe_model = Model(inputs, weighted_experts_out, name="mmoe")

    return mmoe_model
