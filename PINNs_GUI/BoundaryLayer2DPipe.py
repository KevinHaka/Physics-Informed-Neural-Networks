import numpy as np
import tensorflow as tf

def BL2DPipe_PINNs(epochs, r, mu, G, initial_learning_rate, final_learning_rate):
    """
    Trains Physics-Informed Neural Networks (PINNs) to solve the boundary layer problem in a 2D pipe.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    r : float
        Radius of the pipe.
    mu : float
        Dynamic viscosity of the fluid.
    G : float
        Pressure gradient.
    initial_learning_rate : float
        Initial learning rate for the optimizer.
    final_learning_rate : float
        Final learning rate for the optimizer.

    Yields
    ------
    itr : int
        Current iteration number.
    y_plot : numpy.ndarray
        Array of y values for plotting.
    speed : numpy.ndarray
        Array of predicted speed values.
    u_analytical : numpy.ndarray
        Array of analytical solution values.
    train_loss_record : list
        List of training loss values.
    """

    # Define the analytical solution for the boundary layer problem
    u = lambda y, r, G, mu: (G/(2*mu))*(r**2 - y**2)

    def pde_system(y, model, G, mu):
        """
        Defines the PDE system and computes the loss.

        Parameters
        ----------
        y : numpy.ndarray
            Array of y values.
        model : tf.keras.Model
            The neural network model.
        G : float
            Pressure gradient.
        mu : float
            Dynamic viscosity of the fluid.

        Returns
        -------
        total_loss : tf.Tensor
            The total loss combining PDE loss and boundary condition loss.
        """
        # Boundary conditions
        BC = tf.zeros(2)

        y = tf.constant(y, dtype=tf.float32)

        # Compute the gradients of the model output with respect to y
        with tf.GradientTape() as tape:
            tape.watch(y)

            with tf.GradientTape() as tape2:
                tape2.watch(y)

                u = model(y)

                grads = tape2.gradient(u, y)
            grads2 = tape.gradient(grads, y)

        d2u_dy2 = grads2

        # Compute the PDE loss
        pde_loss = mu * d2u_dy2 + G
        # Compute the boundary condition loss
        bc_loss = [u[0], u[-1]] - BC

        # Combine the PDE loss and boundary condition loss
        total_loss = 1e-1 * tf.reduce_mean(tf.square(pde_loss)) + 1e+0 * tf.reduce_mean(tf.square(bc_loss))
        return total_loss

    # Define the neural network model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(1)], name='NN')

    # Number of points in the y direction
    leny = 32

    # Define the learning rate schedule
    decay_steps = int(epochs / 10)
    decay_rate = (final_learning_rate / initial_learning_rate) ** (decay_steps / epochs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
    optm = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Generate training points and analytical solution
    y_train = np.linspace(-r, r, leny)
    y_plot = y_train
    u_analytical = u(y_plot, r, G, mu)

    # Initialize the list to record training loss
    train_loss_record = []

    # Training loop
    for itr in range(1, epochs + 1):
        with tf.GradientTape() as tape:
            # Compute the training loss
            train_loss = pde_system(y_train, model, G, mu)
            train_loss_record.append(train_loss)

            # Compute the gradients and apply them to the model
            grad_w = tape.gradient(train_loss, model.trainable_variables)
            optm.apply_gradients(zip(grad_w, model.trainable_variables))

        # Yield the results every 2 iterations
        if itr % 2 == 0:
            speed = model(y_plot).numpy().reshape(-1)
            yield itr, y_plot, speed, u_analytical, train_loss_record