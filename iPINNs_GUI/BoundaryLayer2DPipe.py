import numpy as np
import tensorflow as tf

def BL2DPipe_iPINNs(epochs, evaluation_points, learning_rate, epochs_per_frame, y_data, u_data):
    """
    Solves a boundary layer problem in a 2D pipe using inverse Physics-Informed Neural Networks (iPINNs).
    Specifically, it trains a neural network with the provided data and the differential equation d2u_dy2 + C = 0.
    The goal is to determine the value of (C) that best fits the given data, by solving the differential equation.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    evaluation_points : int
        Number of points for evaluation.
    learning_rate : float
        Learning rate for the optimizer.
    epochs_per_frame : int
        Number of epochs per frame for yielding results.
    y_data : numpy.ndarray
        Points along the y-axis representing the spatial variable.
    u_data : numpy.ndarray
        Corresponding velocity values at each y point given.

    Yields
    ------
    tuple
        Contains the current epoch, training points, model predictions, 
        record of the parameter C, and training loss record.
    """

    # Generate training points
    y_train = np.linspace(min(y_data), max(y_data), evaluation_points)

    # Convert data to tensors
    y_train_tensor = tf.convert_to_tensor(y_train[:, None], dtype=tf.float32)
    y_data_tensor = tf.convert_to_tensor(y_data[:, None], dtype=tf.float32)
    u_data_tensor = tf.convert_to_tensor(u_data[:, None], dtype=tf.float32)

    # Initialize records for training loss and parameter C
    train_loss_record = []
    C_record = []

    class iPINNs(tf.keras.Model):
        """
        Inverse Physics-Informed Neural Networks (iPINNs) model class.
        """

        def __init__(self):
            """
            Initializes the iPINNs model.
            """
            super(iPINNs, self).__init__()

            # Define hidden layers and their activation functions
            self.hidden_layers = [
                tf.keras.layers.Dense(32, activation='tanh') for _ in range(3)
            ]

            # Define output layer
            self.output_layer = tf.keras.layers.Dense(1)
            
            # Define trainable parameter C
            self.C = self.add_weight(name="C", initializer=tf.constant_initializer(0), trainable=True)

        def call(self, y):
            """
            Forward pass through the network.

            Parameters
            ----------
            y : tf.Tensor
                Input tensor.

            Returns
            -------
            tf.Tensor
                Output tensor.
            """
            x = y
            
            # Pass through layers
            for layer in self.hidden_layers:
                x = layer(x)
            return self.output_layer(x)
    
    @tf.function
    def compute_loss(model, y, y_data, u_data):
        """
        Computes the combined physics and data loss.

        Parameters
        ----------
        model : iPINNs
            The iPINNs model.
        y : tf.Tensor
            Training points tensor.
        y_data : tf.Tensor
            Tensor of input data points for the spatial variable.
        u_data : tf.Tensor
            Tensor of corresponding velocity values at each y point.

        Returns
        -------
        tf.Tensor
            Computed loss.
        """
        C = model.C

        with tf.GradientTape() as tape1:
            tape1.watch(y)

            with tf.GradientTape() as tape2:
                tape2.watch(y)

                u_pred = model(y)

                # Compute first derivative
                du_dy = tape2.gradient(u_pred, y)  

            # Compute second derivative
            d2u_dy2 = tape1.gradient(du_dy, y)  

        # Physics loss
        physics_loss = tf.reduce_mean(tf.square(d2u_dy2 + C))

        # Data loss
        data_loss = tf.reduce_mean(tf.square(u_data - model(y_data)))

        return physics_loss + data_loss

    @tf.function
    def train_step(model, y, y_data, u_data, optimizer):
        """
        Performs a single training step.

        Parameters
        ----------
        model : iPINNs
            The iPINNs model.
        y : tf.Tensor
            Training points tensor.
        y_data : tf.Tensor
            Tensor of input data points for the spatial variable.
        u_data : tf.Tensor
            Tensor of corresponding velocity values at each y point.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer for training.

        Returns
        -------
        tf.Tensor
            Computed loss.
        """
        with tf.GradientTape() as tape:
            loss = compute_loss(model, y, y_data, u_data)
            gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Instantiate the model and optimizer
    model = iPINNs()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):

        # Perform a training step
        loss = train_step(model, y_train_tensor, y_data_tensor, u_data_tensor, optimizer)

        # Record training loss and parameter C
        train_loss_record.append(loss)
        C_record.append(model.C.numpy())

        # Yield results every 'epochs_per_frame' epochs
        if epoch % epochs_per_frame == 0:
            yield (
                epoch, 
                y_train, 
                tf.squeeze(model(y_train_tensor), axis=1),
                C_record,
                train_loss_record
            )