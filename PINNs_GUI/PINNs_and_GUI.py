import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns

from BoundaryLayer2DPipe import BL2DPipe_PINNs
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sns.set_theme()

class App:
    """
    A GUI application for visualizing the results of Physics-Informed Neural Networks (PINNs) 
    applied to the boundary layer in a 2D pipe.

    Parameters
    ----------
    root : tk.Tk
        The root window of the Tkinter application.
    """
    def __init__(self, root):
        # Define color scheme
        self.primary_bg_color = "#808e91"
        self.secondary_bg_color = "#596263"
        self.tertiary_bg_color = "#b2e2eb"

        # Initialize the root window
        self.root = root
        self.root.title("App")
        self.root.geometry("1680x720")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.configure(bg=self.primary_bg_color)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Load the equation image
        self.equation = ImageTk.PhotoImage(Image.open("equation.png"))
        
        # Create the data frame and graphs frame
        self._dataFrame()
        self._graphsFrame()

    def _dataFrame(self):
        """
        Creates the data frame section of the GUI, which includes the equation display and input fields.
        """
        # Create the labeled data frame
        dataFrame = tk.LabelFrame(self.root, text='Data', bg=self.primary_bg_color)
        dataFrame.grid(column=0, row=0, padx=5, pady=5, sticky="enws", rowspan=2)
        dataFrame.rowconfigure(1, weight=1)

        # Add equation and input fields to the data frame
        self._equation(dataFrame)
        self._inputs(dataFrame)
    
    def _equation(self, dataFrame):
        """
        Displays the equation image and title in the data frame.

        Parameters
        ----------
        dataFrame : tk.LabelFrame
            The frame in which the equation is displayed.
        """
        # Create a label for the equation name
        equationName = tk.Label(dataFrame, bg=self.secondary_bg_color)
        equationName.grid(row=0, column=0,  padx=5, pady=5)

        # Set the equation title text
        txt = "Boundary Layer in a 2D Pipe"

        # Create a label for the equation title
        name = tk.Label(master=equationName, text=txt, font="Calibri 15 bold", bg=self.secondary_bg_color)
        name.grid(row=0, column=0, padx=10, pady=10)

        # Display the equation image
        equation_image = self.equation
        equation = tk.Label(master=equationName, image=equation_image, bg=self.secondary_bg_color)
        equation.grid(row=2, column=0, pady=10)

    def _inputs(self, dataFrame):
        """
        Creates input fields for the parameters required by the PINNs.

        Parameters
        ----------
        dataFrame : tk.LabelFrame
            The frame in which the input fields are displayed.
        """
        # Define the input fields with their default values and types
        inputs = [
            ["epochs", 1000, "integer"],
            ["r", 1., "float"],
            ["μ", 1., "float"],
            ["G", 1., "float"],
            ["initial learning rate", 1e-2, "float"],
            ["final learning rate", 1e-4, "float"]
        ]

        # Create a frame for the input fields
        inputsFrame = tk.Label(master=dataFrame, bg=self.secondary_bg_color)
        inputsFrame.grid(row=1, column=0, padx=5, pady=5, sticky="enws")
        inputsFrame.rowconfigure(len(inputs), weight=1)
        inputsFrame.columnconfigure([0,1], weight=1, uniform="input_columns")

        # Create input fields and store their variables
        Entries = []
        for idx in range(len(inputs)):
            name, value, varType = inputs[idx]
            Entries.append(self._input(inputsFrame, idx, name, value, varType))

        # Define the function to get input values and run the PINNs
        inputs_ = lambda Entries: [entry.get() for entry in Entries]
        runPINNs = lambda : self._new_run(inputs_(Entries))
        
        # Create a submit button to start the PINNs run
        button = tk.Button(inputsFrame, text="Submit", command=runPINNs)
        button.grid(row=len(inputs), columnspan=2)

    def _input(self, frame, idx, name, value, varType):
        """
        Creates a single input field.

        Parameters
        ----------
        frame : tk.Frame
            The frame in which the input field is displayed.
        idx : int
            The index of the input field.
        name : str
            The name of the parameter.
        value : int or float
            The default value of the parameter.
        varType : str
            The type of the parameter ("integer" or "float").

        Returns
        -------
        var : tk.Variable
            The Tkinter variable associated with the input field.
        """
        # Create a variable based on the type
        if varType == "integer":
            var = tk.IntVar(value=value)
        elif varType == "float":
            var = tk.DoubleVar(value=value)

        # Create a label for the input field
        label = tk.Label(frame, text=name, font="Calibri 10 bold", bg=self.secondary_bg_color)
        label.grid(row=idx, column=0, padx=(5,0), pady=10, sticky="nsew")

        # Create an entry widget for the input field
        entry = tk.Entry(frame, textvariable=var, font="Calibri 10 bold", bg=self.tertiary_bg_color, width=10)
        entry.grid(row=idx, column=1, padx=(0,5), pady=10, sticky="ns")
    
        return var

    def _graphsFrame(self):
        """
        Creates the graphs frame section of the GUI, which includes the graphs and progress bar.
        """
        # Initialize the running state
        self.running = False

        # Create the graphs frame
        self.graphsFrame = tk.LabelFrame(self.root, text='Graphs', bg=self.primary_bg_color)
        self.graphsFrame.grid(row=0, column=1, padx=5, pady=5, sticky="enws")
        self.graphsFrame.rowconfigure(0, weight=1)
        self.graphsFrame.columnconfigure(0, weight=1)

        # Add graphs and progress bar to the graphs frame
        self._graphs()
        self._progressBar()

    def _graphs(self):
        """
        Initializes the graphs for displaying the training results.

        Returns
        -------
        canvas : FigureCanvasTkAgg
            The canvas on which the graphs are drawn.
        trainingAx : matplotlib.axes.Axes
            The axes for the training graph.
        lossAx : matplotlib.axes.Axes
            The axes for the loss graph.
        analytical : matplotlib.lines.Line2D
            The line representing the analytical solution.
        PINNs : matplotlib.lines.Line2D
            The line representing the PINNs solution.
        loss : matplotlib.lines.Line2D
            The line representing the training loss.
        """
        # Create a figure with two subplots
        figure, (trainingAx, lossAx) = plt.subplots(1, 2)
        canvas = FigureCanvasTkAgg(figure, master=self.graphsFrame)
        canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="enws")

        # Initialize the lines for the graphs
        analytical, = trainingAx.plot([], [], 'b-', label='Analytical')
        PINNs, = trainingAx.plot([], [], 'r*', label='PINNs')
        loss, = lossAx.plot([1], [1], 'g-', label='Loss')

        # Set the y-axis of the loss graph to logarithmic scale
        lossAx.set_yscale('log')

        # Set labels for the axes
        trainingAx.set_xlabel('Distance from the center')
        trainingAx.set_ylabel('Speed')
        lossAx.set_xlabel('Epochs')
        lossAx.set_ylabel('Loss')

        # Add legends to the graphs
        trainingAx.legend()
        lossAx.legend()

        return canvas, trainingAx, lossAx, analytical, PINNs, loss

    def _progressBar(self):
        """
        Initializes the progress bar for displaying the training progress.

        Returns
        -------
        progressBar : Progressbar
            The progress bar widget.
        progress : tk.IntVar
            The variable associated with the progress bar.
        progressToString : tk.StringVar
            The string variable for displaying the progress percentage.
        """
        # Create a variable for the progress and its string representation
        progress = tk.IntVar()
        progressToString = tk.StringVar(value=f"{progress.get()}%")

        # Create the progress bar and label
        progressBar = Progressbar(self.graphsFrame, variable=progress)
        progressVal = tk.Label(self.graphsFrame, textvariable=progressToString, font=("Helvetica", 20))
        
        # Place the progress bar and label in the grid
        progressBar.grid(row=1, column=0, padx=5, pady=5, sticky="enws", columnspan=2)
        progressVal.grid(row=1, column=0, padx=5, pady=10, sticky="ns")

        return progressBar, progress, progressToString

    def _runPINNs(self, inputs):
        """
        Runs the PINNs training and updates the graphs and progress bar.

        Parameters
        ----------
        inputs : list
            The list of input parameters for the PINNs.
        """
        # Set the running state to True
        self.running = True

        # Extract the number of epochs from the inputs
        epochs = inputs[0]
        
        # Initialize the graphs and progress bar
        canvas, trainingAx, lossAx, analytical, PINNs, loss = self._graphs()
        progressBar, progress, progressToString = self._progressBar()
        
        # Run the PINNs training loop
        for itr, y_plot, speed, u_analytical, train_loss_record in BL2DPipe_PINNs(*inputs):
            if not self.running: break
            
            # Update the progress bar
            progress.set(itr*100/epochs)
            progressToString.set(f"{progress.get()}%")

            # Update the data for the graphs
            analytical.set_data(y_plot, u_analytical)
            PINNs.set_data(y_plot, speed)
            loss.set_data(range(1, itr+1), train_loss_record)

            # Update the graphs and progress bar
            self._updateGraphsAndProgressBar(trainingAx, lossAx, progressBar, canvas)
    
    def _new_run(self, inputs):
        """
        Starts a new run of the PINNs model.

        Parameters
        ----------
        inputs : list
            List of input parameters for the iPINNs model.
        """
        # Set the running state to False and start the PINNs run after a short delay
        self.running = False
        self.root.after(100, self._runPINNs, inputs)

    @staticmethod
    def _updateGraphsAndProgressBar(trainingAx, lossAx, progressBar, canvas):
        """
        Updates the graphs and progress bar.

        Parameters
        ----------
        trainingAx : matplotlib.axes.Axes
            The axes for the training graph.
        lossAx : matplotlib.axes.Axes
            The axes for the loss graph.
        progressBar : Progressbar
            The progress bar widget.
        canvas : FigureCanvasTkAgg
            The canvas on which the graphs are drawn.
        """
        # Rescale the axes and update the canvas
        trainingAx.relim()
        trainingAx.autoscale_view()
        lossAx.relim()
        lossAx.autoscale_view()
        progressBar.update()
        canvas.draw()

    def _on_closing(self):
        """
        Handles the closing event of the application.
        """
        # Set the running state to False
        self.running = False
        # Close all matplotlib figures
        plt.close('all')
        # Destroy the root window
        self.root.destroy()

if __name__=='__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()