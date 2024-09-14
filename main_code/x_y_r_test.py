import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
def on_closing():
    root.quit()  # Stop the Tkinter mainloop
    root.destroy()  # Destroy the window
# Function to load the CSV file and filter data
def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        global data
        data = pd.read_csv(file_path, dtype={16: str, 19: str}, low_memory=False)
        lbl_file.config(text=f"File loaded: {file_path}")

        # Set the max values for ROI and Point
        max_roi = data['ROI number'].max()
        max_point = data['Point'].max()
        spinbox_roi.config(from_=1, to=max_roi)
        spinbox_point.config(from_=1, to=max_point)

        # Trigger the filter as soon as file is loaded
        filter_data()

# Function to filter data based on ROI number and Point
def filter_data(*args):
    try:
        roi = int(spinbox_roi.get())
        point = int(spinbox_point.get())
        filtered_data = data[(data['ROI number'] == roi) & (data['Point'] == point)]
        
        # Plot X, Y, and Radius
        plot_graphs(filtered_data)
        
    except ValueError:
        lbl_file.config(text="Please enter valid numbers for ROI and Point.")
    except KeyError:
        lbl_file.config(text="Selected ROI or Point does not exist in data.")

# Function to plot the graphs
def plot_graphs(filtered_data):
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    
    axs[0].plot(filtered_data['Frame'], filtered_data['X Coordinate'], label='X Coordinate')
    axs[0].set_title('X Coordinate')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('X Coordinate')
    
    axs[1].plot(filtered_data['Frame'], filtered_data['Y Coordinate'], label='Y Coordinate', color='orange')
    axs[1].set_title('Y Coordinate')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Y Coordinate')
    
    axs[2].plot(filtered_data['Frame'], filtered_data['Radius'], label='Radius', color='green')
    axs[2].set_title('Radius')
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('Radius')
    
    # Clear previous plot and display new one
    for widget in plot_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Setup the GUI window
root = tk.Tk()
root.title("CSV Data Plotter")
# Handle window close event
root.protocol("WM_DELETE_WINDOW", on_closing)
# Make the window resizable
root.rowconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

# Setup frame layout
input_frame = tk.Frame(root)
input_frame.grid(row=0, column=0, sticky="ns")

plot_frame = tk.Frame(root)
plot_frame.grid(row=0, column=1, sticky="nsew")

# Add file loading button
btn_load = tk.Button(input_frame, text="Load CSV File", command=load_csv)
btn_load.pack()

lbl_file = tk.Label(input_frame, text="No file loaded")
lbl_file.pack()

# Add spinboxes for ROI and Point
lbl_roi = tk.Label(input_frame, text="ROI number:")
lbl_roi.pack()
spinbox_roi = tk.Spinbox(input_frame, from_=1, to=10, command=filter_data)
spinbox_roi.pack()

lbl_point = tk.Label(input_frame, text="Point:")
lbl_point.pack()
spinbox_point = tk.Spinbox(input_frame, from_=1, to=10, command=filter_data)
spinbox_point.pack()

# Add a button to filter data and plot
btn_plot = tk.Button(input_frame, text="Filter and Plot", command=filter_data)
btn_plot.pack()

# Make plot_frame resizable with the window
plot_frame.rowconfigure(0, weight=1)
plot_frame.columnconfigure(0, weight=1)

root.mainloop()
