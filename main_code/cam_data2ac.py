import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt

# Load data from CSV
def load_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return None
    
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    return data

# Save data to .npy
def save_data(data):
    file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("Numpy files", "*.npy")])
    if not file_path:
        return
    
    np.save(file_path, data)
    messagebox.showinfo("Success", "Data saved successfully!")

# Calculate second derivative
def calculate_second_derivative(data):
    means = np.mean(data, axis=0)
    centered_data = data - means
    
    # Compute second derivative
    second_derivative = np.diff(centered_data, n=2, axis=0)
    return second_derivative

# Plot data
def plot_data(data):
    num_columns = data.shape[1]
    plt.figure(figsize=(12, 6))
    
    # Original data
    plt.subplot(2, 1, 1)
    for i in range(num_columns):
        plt.plot(data[:, i], label=f'Column {i+1}')
    plt.title('Original Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    
    # Second derivative
    if data.shape[0] > 2:
        second_derivative = calculate_second_derivative(data)
        plt.subplot(2, 1, 2)
        for i in range(second_derivative.shape[1]):
            plt.plot(second_derivative[:, i], label=f'Second Derivative Column {i+1}')
        plt.title('Second Derivative')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
    else:
        plt.subplot(2, 1, 2)
        plt.text(0.5, 0.5, 'Not enough data points for second derivative calculation',
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=12, color='red', transform=plt.gca().transAxes)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# GUI setup
def run_gui():
    root = tk.Tk()
    root.withdraw()
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Plot data
    plot_data(data)
    
    # Calculate second derivative and save data
    if data.shape[0] > 2:
        second_derivative = calculate_second_derivative(data)
        save_data(second_derivative)
    else:
        messagebox.showwarning("Warning", "Not enough data points to calculate second derivative.")

if __name__ == "__main__":
    run_gui()
