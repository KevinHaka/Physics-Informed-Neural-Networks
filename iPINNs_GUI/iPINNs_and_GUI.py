import tkinter as tk
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from BoundaryLayer2DPipe import BL2DPipe_iPINNs
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk

sns.set_theme()


class App:
    """
    A class to create a GUI application for iPINNs.

    Attributes
    ----------
    root : tk.Tk
        The root window of the application.
    primary_bg_color : str
        Primary background color for the application.
    secondary_bg_color : str
        Secondary background color for the application.
    tertiary_bg_color : str
        Tertiary background color for the application.
    equation_image : ImageTk.PhotoImage
        Image of the equation to be displayed.
    drag_data : dict
        Data for handling drag events.
    layout : list
        Layout for the matplotlib subplots.
    """

    def __init__(self, root):
        """
        Constructs all the necessary attributes for the App object.

        Parameters
        ----------
        root : tk.Tk
            The root window of the application.
        """
        # Define the background colors for the application
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
        self.equation_image = ImageTk.PhotoImage(Image.open("equation.png"))
        # Initialize the drag data
        self.drag_data = {'press': None, 'xlim': None, 'ylim': None}
        # Define the layout for the matplotlib subplots
        self.layout = [
            ["A", "B"],
            ["A", "C"]
        ]
        
        # Create the data frame and graphs frame
        self._dataFrame()
        self._graphsFrame()

    def _dataFrame(self):
        """
        Creates the data frame containing the equation and input fields.
        """
        # Create a labeled frame for data
        self.dataFrame = tk.LabelFrame(self.root, text='Data', bg=self.primary_bg_color)
        self.dataFrame.grid(column=0, row=0, padx=5, pady=5, sticky="enws")
        self.dataFrame.rowconfigure(1, weight=1)

        # Add the equation and input fields to the data frame
        self._equation()
        self._inputs()
    
    def _equation(self):
        """
        Creates the frame for displaying the equation.
        """
        # Create a frame for the equation
        self.equationName = tk.Frame(self.dataFrame, bg=self.secondary_bg_color)
        self.equationName.grid(row=0, column=0,  padx=5, pady=5, sticky="nsew")
        self.dataFrame.rowconfigure(1, weight=1)

        # Set the text for the equation
        txt = "Boundary Layer in a 2D Pipe"

        # Create a label for the equation name
        self.name = tk.Label(self.equationName, text=txt, anchor="center", font="Calibri 15 bold", bg=self.secondary_bg_color)
        self.name.grid(row=0, column=0, padx=5, pady=5, sticky="we")

        # Create a label for the equation image
        self.equation = tk.Label(self.equationName, image=self.equation_image, bg=self.secondary_bg_color)
        self.equation.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

    def _inputs(self):
        """
        Creates the input fields for user parameters.
        """
        # Define the input fields with their default values and types
        inputs = [
            ["epochs", 5000, "integer"],
            ["evaluation points", 30, "integer"],
            ["learning rate", 1e-2, "float"],
            ["epochs per frame", 20, "integer"]
        ]

        # Create a frame for the input fields
        self.inputsFrame = tk.Frame(self.dataFrame, bg=self.secondary_bg_color)
        self.inputsFrame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.inputsFrame.rowconfigure(len(inputs), weight=1)
        self.inputsFrame.columnconfigure([0,1], weight=1, uniform="input_columns")

        # Create input fields and store their variables
        entries = []
        for idx in range(len(inputs)):
            name, value, varType = inputs[idx]
            entries.append(self._input(self.inputsFrame, idx, name, value, varType))

        # Define the actions for the submit and reset buttons
        input_values = lambda entries: [entry.get() for entry in entries]
        runPINNs = lambda : self._new_run(input_values(entries))

        # Create the submit button
        self.submit = tk.Button(self.inputsFrame, text="Submit", command=runPINNs)
        self.submit.grid(row=len(inputs), column=0, pady=(20,0), sticky="n")
        
        # Create the reset button
        self.reset = tk.Button(self.inputsFrame, text="Reset", command=self._graphsFrame)
        self.reset.grid(row=len(inputs), column=1, pady=(20,0), sticky="n")

    def _input(self, frame, idx, name, value, varType):
        """
        Creates an individual input field.

        Parameters
        ----------
        frame : tk.Frame
            The frame to place the input field in.
        idx : int
            The index of the input field.
        name : str
            The name of the input field.
        value : int or float
            The default value of the input field.
        varType : str
            The type of the input field ("integer" or "float").

        Returns
        -------
        tk.Variable
            The variable associated with the input field.
        """
        # Create a variable for the input field based on its type
        if varType == "integer":
            var = tk.IntVar(value=value)
        elif varType == "float":
            var = tk.DoubleVar(value=value)

        # Create a label for the input field
        label = tk.Label(frame, text=name, font="Calibri 10 bold", bg=self.secondary_bg_color)
        label.grid(row=idx, column=0, padx=(5,0), pady=20, sticky="nsew")

        # Create an entry widget for the input field
        entry = tk.Entry(frame, textvariable=var, font="Calibri 10 bold", bg=self.tertiary_bg_color, width=10)
        entry.grid(row=idx, column=1, padx=(0,5), pady=20, sticky="ns")
    
        return var

    def _graphsFrame(self):
        """
        Creates the frame for displaying graphs and progress bar.
        """
        # Initialize the running state
        self.running = False
        
        # Create a labeled frame for graphs
        self.graphsFrame = tk.LabelFrame(self.root, text='Graphs', bg=self.primary_bg_color)
        self.graphsFrame.grid(row=0, column=1, padx=5, pady=5, sticky="enws")
        self.graphsFrame.rowconfigure(0, weight=1)
        self.graphsFrame.columnconfigure(0, weight=1)

        # Add the graphs and progress bar to the frame
        self._graphs()
        self._progressBar()

    def _graphs(self):
        """
        Creates the matplotlib subplots for displaying data, iPINNs results, loss, and parameters.
        """
        # Create a mosaic layout for the subplots
        self.fig, axd = plt.subplot_mosaic(self.layout)
        self.trainingAx, self.lossAx, self.parameterAx = axd.values()

        # Create a canvas for the matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graphsFrame)
        self.canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="enws")

        # Initialize the plots for data, iPINNs results, loss, and parameters
        self.data, = self.trainingAx.plot([], [], 'b*', label='data')
        self.iPINNs, = self.trainingAx.plot([], [], 'r-', label='iPINNs')
        self.loss, = self.lossAx.plot([1], [1], 'g-')
        self.parameter, = self.parameterAx.plot([], [], 'y-')

        # Connect matplotlib events to their respective handlers
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)

        # Set labels for the axes
        self.lossAx.set_ylabel('Loss')
        self.parameterAx.set_xlabel('Epochs')
        self.parameterAx.set_ylabel('C (G/μ)')
        self.trainingAx.set_ylabel('$u_x$ (Speed)')
        self.trainingAx.set_xlabel('y (Boundary Coordinates)')

        # Set the y-axis scale to logarithmic for the loss plot
        self.lossAx.set_yscale('log')
        # Add legend to the training plot
        self.trainingAx.legend()

    def _progressBar(self):
        """
        Creates the progress bar for displaying training progress.
        """
        # Initialize the progress variable and its string representation
        self.progress = tk.IntVar()
        self.progressToString = tk.StringVar(value=f"{self.progress.get()}%")

        # Create the progress bar widget
        self.progressBar = Progressbar(self.graphsFrame, variable=self.progress)
        
        # Create a label to display the progress percentage
        self.progressVal = tk.Label(self.graphsFrame, textvariable=self.progressToString, font=("Helvetica", 20))

        # Place the progress bar and label in the frame
        self.progressBar.grid(row=1, column=0, padx=5, pady=5, sticky="enws")
        self.progressVal.grid(row=1, column=0, padx=5, pady=10, sticky="ns")

    def _run_iPINNs(self, inputs):
        """
        Runs the iPINNs model with the given inputs.

        Parameters
        ----------
        inputs : list
            List of input parameters for the iPINNs model.
        """
        # Set the running state to True
        self.running = True
        
        # Get the number of epochs from the inputs
        epochs = inputs[0]
        
        # Append current data points to inputs
        inputs.append(self.data.get_xdata())
        inputs.append(self.data.get_ydata())

        # Iterate through the iPINNs model training process
        for itr, y_plot, speed, C_record, train_loss_record in BL2DPipe_iPINNs(*inputs):
            if not self.running: break
            
            # Update the progress bar
            self.progress.set(itr*100/epochs)
            self.progressToString.set(f"{self.progress.get()}%")

            # Update the plots with the new data
            self.iPINNs.set_data(y_plot, speed)
            self.loss.set_data(range(1, itr+1), train_loss_record)
            self.parameter.set_data(range(1, itr+1), C_record)
            
            # Refresh the graphs and progress bar
            self._updateGraphsAndProgressBar()
    
    def _updateGraphsAndProgressBar(self):
        """
        Updates the graphs and progress bar.
        """
        # Adjust the limits and redraw the plots
        self.lossAx.relim()
        self.lossAx.autoscale_view()
        self.trainingAx.relim()
        self.trainingAx.autoscale_view()
        self.parameterAx.relim()
        self.parameterAx.autoscale_view()
        self.progressBar.update()
        self.canvas.draw()

    def _new_run(self, inputs):
        """
        Starts a new run of the iPINNs model.

        Parameters
        ----------
        inputs : list
            List of input parameters for the iPINNs model.
        """
        # Set the running state to False and start the iPINNs run after a short delay
        self.running = False
        self.root.after(100, self._run_iPINNs, inputs)

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

    def _on_scroll(self, event):
        """
        Handles the scroll event on the training axis.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The scroll event.
        """
        # Get the axis where the event occurred
        ax = event.inaxes

        # Only handle scroll events on the training axis
        if ax != self.trainingAx: return

        # Determine the scale factor based on the scroll direction
        scale_factor = 0.5 if event.button == 'up' else 2.0
        xdata, ydata = event.xdata, event.ydata

        # Get the current limits of the axis
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Calculate the new limits based on the scale factor
        ax.set_xlim([xdata - (xdata - xlim[0]) * scale_factor, xdata + (xlim[1] - xdata) * scale_factor])
        ax.set_ylim([ydata - (ydata - ylim[0]) * scale_factor, ydata + (ylim[1] - ydata) * scale_factor])
        
        # Auto-adjust the limits and redraw the plot
        self.trainingAx.set_xlim(auto=True)
        self.trainingAx.set_ylim(auto=True)
        ax.figure.canvas.draw()

    def _on_click(self, event):
        """
        Handles the click event on the training axis.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The click event.
        """
        # Only handle right-click events
        if event.button != 3: return

        # Get the current data points
        current_x = self.data.get_xdata()
        current_y = self.data.get_ydata()

        # Append the new data point to the current data
        updated_x = np.append(current_x, event.xdata)
        updated_y = np.append(current_y, event.ydata)

        # Update the data plot with the new data
        self.data.set_data(updated_x, updated_y)
        event.inaxes.figure.canvas.draw()
        
    def _on_press(self, event):
        """
        Handles the press event on the training axis.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The press event.
        """
        # Only handle left-click events
        if event.button != 1: return
        
        # Store the press position and current axis limits
        self.drag_data['press'] = (event.x, event.y)
        self.drag_data['xlim'] = self.trainingAx.get_xlim()
        self.drag_data['ylim'] = self.trainingAx.get_ylim()

    def _on_release(self, _):
        """
        Handles the release event on the training axis.
        """
        # Reset the drag data
        self.drag_data['press'] = None
        self.drag_data['xlim'] = None
        self.drag_data['ylim'] = None

    def _on_motion(self, event):
        """
        Handles the motion event on the training axis.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The motion event.
        """
        # Only handle motion events if a press event has occurred
        if self.drag_data['press'] is None: return
        if event.inaxes != self.trainingAx: return

        # Calculate the change in position
        dx = event.x - self.drag_data['press'][0]
        dy = event.y - self.drag_data['press'][1]

        # Calculate the scale factors for the x and y axes
        scale_x = (self.drag_data['xlim'][1] - self.drag_data['xlim'][0]) / self.trainingAx.figure.bbox.width
        scale_y = (self.drag_data['ylim'][1] - self.drag_data['ylim'][0]) / self.trainingAx.figure.bbox.height

        # Calculate the new limits based on the change in position
        new_xlim = [self.drag_data['xlim'][0] - dx * scale_x, self.drag_data['xlim'][1] - dx * scale_x]
        new_ylim = [self.drag_data['ylim'][0] - dy * scale_y, self.drag_data['ylim'][1] - dy * scale_y]

        # Set the new limits
        self.trainingAx.set_xlim(new_xlim)
        self.trainingAx.set_ylim(new_ylim)
        # Auto-adjust the limits
        self.trainingAx.set_xlim(auto=True)
        self.trainingAx.set_ylim(auto=True)
        # Redraw the plot
        self.trainingAx.figure.canvas.draw()

if __name__=='__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()