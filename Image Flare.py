import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Toplevel, ttk
from PIL import Image, ImageTk, ImageOps
import customtkinter as ctk
import customtkinter 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from skimage.util import random_noise
from tkinter import filedialog
from PIL import ImageEnhance, ImageFilter
from PIL import Image, ImageOps
import copy
from tkinter import simpledialog
from functools import wraps
from tkinter import messagebox
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle
from skimage import img_as_float
from skimage.util import random_noise
from skimage import exposure
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.restoration import denoise_bilateral
from skimage.restoration import denoise_wavelet
import tkinter.messagebox as messagebox 


customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("dark-blue")


# Initialize the main window
root = ctk.CTk()
root.title("Image Flare")
root.attributes('-fullscreen', True)



# Define colors
chatgpt_bg_color = '#f0f0f0'  # Replace with the exact color of ChatGPT's background
lighter_gray = 'black'

root.configure(bg=chatgpt_bg_color)

loaded_image = None
cropped_image = None
original_cropped_image = None
undo_stack = []
redo_stack = []
start_x = start_y = end_x = end_y = None
image_id = None
current_operation = None
zoom_scale = 1.0
description_shown = {}
histogram_open = False
compare_open = False  

# Global variable to track if the resize button has been clicked before
resize_button_clicked = False
open_file_clicked = False
save_image_clicked = False
crop_message_shown = False
convert_to_black_white_clicked = False
convert_to_color_clicked = False
undo_clicked = False
redo_clicked = False
reset_clicked = False
histogram_analysis_clicked = False
quantitative_analysis_clicked = False
brightness_message_shown = False
contrast_message_shown = False
sharpness_message_shown = False
blurriness_message_shown = False
noise_message_shown = False
filter_message_shown = False
denoise_message_shown = False

def open_image():
    global loaded_image, cropped_image, original_cropped_image, undo_stack, redo_stack, open_file_clicked
    try:
        file_path = filedialog.askopenfilename()

        print(f"Selected file path: {file_path}")  # Check the file path
        if file_path:
            loaded_image = Image.open(file_path)
            cropped_image = loaded_image.copy()
            original_cropped_image = cropped_image.copy()  # Store the original cropped image
            undo_stack = [cropped_image.copy()]
            redo_stack.clear()
            edited_canvas.delete("all")  # Clear the canvas
            display_image(cropped_image, edited_canvas, fit_to_canvas=True)  # Display the opened image
            # Show a pop-up message the first time an image is opened
            if open_file_clicked:
                messagebox.showinfo("Functionality Info", "You can now apply filters or resize the image.")
                open_file_clicked = True  # Set the flag to False after the first message
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {e}")
        

def save_image(): 
    global first_time_save  # Declare the global variable
    try: 
        if cropped_image:  # Ensure cropped_image is not None
            # Show a pop-up message the first time the user tries to save an image
            if save_image_clicked:
                messagebox.showinfo("Save Functionality", "You can now save the edited image.")
                save_image_clicked = True  # Set the flag to False after the first message
            
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", ".png"), ("JPEG files", ".jpg"), ("All files", ".*")]) 
            if file_path: 
                cropped_image.save(file_path) 
    except Exception as e: 
        messagebox.showerror("Error", f"Failed to save image: {e}")  # Optional: Show an error message if saving fails
    

def close_image():
    global loaded_image, cropped_image, original_cropped_image
    loaded_image = None
    cropped_image = None
    original_cropped_image = None
    display_background_image()  # Show the background image again

def exit_application():
    root.quit()  # Properly quit the main loop
    root.destroy()  # Destroy the window
root.protocol("WM_DELETE_WINDOW", exit_application)

def display_image(image, canvas, fit_to_canvas=False):
    canvas.delete("all")  # Clear the canvas
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    img = image.copy()
    
    # Fit the image to the canvas while maintaining aspect ratio
    if fit_to_canvas:
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            # Scale by width
            new_width = canvas_width
            new_height = int(new_width / img_ratio)
        else:
            # Scale by height
            new_height = canvas_height
            new_width = int(new_height * img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert the image for Tkinter display
    img_tk = ImageTk.PhotoImage(img)
    global image_id
    image_id = canvas.create_image(canvas_width / 2, canvas_height / 2, anchor='center', image=img_tk)
    canvas.image = img_tk  # Keep a reference to avoid garbage collection

def start_crop(event):
    global start_x, start_y
    start_x, start_y = event.x, event.y
    update_status_bar(event.x, event.y)
    draw_crop(event)
    

def draw_crop(event):
    global start_x, start_y, end_x, end_y
    end_x, end_y = event.x, event.y
    edited_canvas.delete('crop_rectangle')  # Delete the previous rectangle
    edited_canvas.create_rectangle(start_x, start_y, end_x, end_y, outline='red', tag='crop_rectangle')

def crop_image():
    global cropped_image, undo_stack, redo_stack, start_x, start_y, end_x, end_y, crop_message_shown

    try:
        if cropped_image and start_x is not None and start_y is not None and end_x is not None and end_y is not None:
            # Show a pop-up message the first time the user tries to crop an image
            if not crop_message_shown:
                messagebox.showinfo("Crop Functionality", "You can now crop the image using the selected area.")
                crop_message_shown = True  # Set the flag to True after the first message

            # Get the current image dimensions (latest updated image)
            current_width, current_height = cropped_image.size

            # Get the dimensions of the displayed image on the canvas
            canvas_width = edited_canvas.winfo_width()
            canvas_height = edited_canvas.winfo_height()

            # Calculate the aspect ratio of the image and the canvas
            image_ratio = current_width / current_height
            canvas_ratio = canvas_width / canvas_height

            # Determine the dimensions of the displayed image
            if image_ratio > canvas_ratio:
                displayed_width = canvas_width
                displayed_height = int(canvas_width / image_ratio)
            else:
                displayed_height = canvas_height
                displayed_width = int(canvas_height * image_ratio)

            # Calculate the scaling factors
            scale_x = current_width / displayed_width
            scale_y = current_height / displayed_height

            # Calculate the offset due to centering (if any)
            offset_x = (canvas_width - displayed_width) // 2
            offset_y = (canvas_height - displayed_height) // 2

            # Adjust the crop coordinates based on scaling and offset
            adj_start_x = int((start_x - offset_x) * scale_x)
            adj_start_y = int((start_y - offset_y) * scale_y)
            adj_end_x = int((end_x - offset_x) * scale_x)
            adj_end_y = int((end_y - offset_y) * scale_y)

            # Ensure the coordinates are within image boundaries
            adj_start_x = max(0, min(adj_start_x, current_width))
            adj_start_y = max(0, min(adj_start_y, current_height))
            adj_end_x = max(0, min(adj_end_x, current_width))
            adj_end_y = max(0, min(adj_end_y, current_height))

            # Ensure the crop rectangle is not zero-sized
            if adj_start_x != adj_end_x and adj_start_y != adj_end_y:
                # Crop the latest updated image
                cropped_image = cropped_image.crop((adj_start_x, adj_start_y, adj_end_x, adj_end_y))

                # Display the cropped image
                display_image(cropped_image, edited_canvas, fit_to_canvas=True)
                push_to_undo_stack(cropped_image)  # Save state to undo stack

        # Reset cropping coordinates
        start_x = start_y = end_x = end_y = None

    except Exception as e:
        messagebox.showwarning("Error in cropping the image.", str(e))


def apply_resize():
    global cropped_image, original_cropped_image, undo_stack, redo_stack, resize_button_clicked
    try:
        # Check if an image is loaded
        if cropped_image is None:
            messagebox.showwarning("No Image Loaded", "Please open an image first before resizing.")
            return
        
        # Show description message if this is the first click
        if not resize_button_clicked:
            messagebox.showinfo("Resize Function", "This function allows you to resize the current image. Input the desired width and height.")
            resize_button_clicked = True  # Set the flag to True after the first click
        
        # Check if width and height are entered
        width_str = resize_entry_width.get()
        height_str = resize_entry_height.get()
        
        if not width_str or not height_str:
            messagebox.showwarning("Input Required", "Please enter both width and height values before resizing.")
            return
        
        # Proceed with resizing if an image is loaded and values are provided
        width = int(width_str)
        height = int(height_str)
        cropped_image = cropped_image.resize((width, height), Image.Resampling.LANCZOS)
        original_cropped_image = cropped_image.copy()  # Update the original cropped image
        display_image(cropped_image, edited_canvas, fit_to_canvas=True)
        push_to_undo_stack(cropped_image) 
    except Exception as e:
        messagebox.showwarning("Error in resizing the image.")

def convert_to_black_white():
    global cropped_image, original_cropped_image, undo_stack, redo_stack, convert_to_black_white_clicked
    if cropped_image:
        # Show a pop-up message the first time the user tries to convert to black and white
        if convert_to_black_white_clicked:
            messagebox.showinfo("Grayscale Functionality", "The image will be converted to GrayScale.")
            convert_to_black_white_clicked = True  # Set the flag to False after the first message

        # Check if the image is in RGB mode
        if cropped_image.mode == "RGB":  # 'RGB' is the mode for color images
            # Convert the RGB image to grayscale
            cropped_image = ImageOps.grayscale(cropped_image)

            # Update the original cropped image and stacks
            original_cropped_image = cropped_image.copy()  # Store the converted image
        
            # Display the converted image on the right canvas
            display_image(cropped_image, edited_canvas, fit_to_canvas=True)  # Use fit_to_canvas=False for precise size
            push_to_undo_stack(cropped_image) 

def convert_to_color():
    global cropped_image, original_cropped_image, undo_stack, redo_stack, convert_to_color_clicked
    if cropped_image:
        # Show a pop-up message the first time the user tries to convert to color
        if convert_to_color_clicked:
            messagebox.showinfo("Color Functionality", "The image will be converted back to color.")
            convert_to_color_clicked = True  # Set the flag to False after the first message

        # Check if the image is in grayscale mode
        if cropped_image.mode == "L":  # 'L' is the mode for grayscale images
            # Convert the grayscale image back to RGB
            cropped_image = cropped_image.convert("RGB")
        else:
            # If already in RGB, do nothing or handle as needed
            messagebox.showinfo("Info", "Image is already in color mode.")
        
        # Update the original cropped image and stacks
        original_cropped_image = cropped_image.copy()
        undo_stack.append(cropped_image.copy())
        redo_stack.clear()
        
        # Display the converted image
        display_image(cropped_image, edited_canvas, fit_to_canvas=True)

def push_to_undo_stack(image):
    """Push the current image state to the undo stack."""
    global undo_stack, redo_stack
    if image:
        # Make a deep copy of the image to preserve its state
        undo_stack.append(copy.deepcopy(image))
        # Clear the redo stack when a new change is made
        redo_stack.clear()

def undo():
    """Undo the last image operation."""
    global undo_stack, redo_stack, cropped_image, undo_clicked
    if undo_stack:
        # Show a pop-up message the first time the user tries to undo
        if not undo_clicked:
            messagebox.showinfo("Undo Functionality", "You can now undo the last image operation.")
            undo_clicked = True  # Set the flag to True after the first message

        # Push current image state to redo stack
        redo_stack.append(copy.deepcopy(cropped_image))
        # Pop from undo stack and set as current image
        cropped_image = undo_stack.pop()
        display_image(cropped_image, edited_canvas, fit_to_canvas=True)
    else:
        messagebox.showwarning("No More Undos", "No more undos available.")

def redo():
    """Redo the last undone image operation."""
    global redo_stack, undo_stack, cropped_image, redo_clicked
    if redo_stack:
        # Show a pop-up message the first time the user tries to redo
        if not redo_clicked:
            messagebox.showinfo("Redo Functionality", "You can now redo the last undone image operation.")
            redo_clicked = True  # Set the flag to True after the first message

        # Push current image state to undo stack
        undo_stack.append(copy.deepcopy(cropped_image))
        # Pop from redo stack and set as current image
        cropped_image = redo_stack.pop()
        display_image(cropped_image, edited_canvas, fit_to_canvas=True)
    else:
        messagebox.showwarning("No More Redos", "No more redos available.")

def reset_process():
    """Reset the image to its original state."""
    global cropped_image, original_cropped_image, undo_stack, redo_stack, zoom_scale, start_x, start_y, end_x, end_y, reset_clicked

    # Show a pop-up message the first time the user tries to reset
    if not reset_clicked:
        messagebox.showinfo("Reset Functionality", "You can now reset the image to its original state.")
        reset_clicked = True  # Set the flag to True after the first message

    if original_cropped_image:
        # Reset cropped_image to the original image
        cropped_image = loaded_image.copy()
        
        # Reset undo and redo stacks
        undo_stack = [cropped_image.copy()]
        redo_stack.clear()
        
        # Reset zoom scale
        zoom_scale = 1.0
        
        # Reset cropping coordinates
        start_x = start_y = end_x = end_y = None
        
        # Display the reset image with fit_to_canvas=False to maintain size
        display_image(cropped_image, edited_canvas, fit_to_canvas=True)

        # Optionally reset the status bar or any additional UI elements
        update_status_bar(0, 0)
    else:
        messagebox.showwarning("No More Resets", "No more resets available.")

def move_image(event):
    global image_id
    x, y = event.x, event.y
    edited_canvas.coords(image_id, x, y)

def zoom(event):
    global cropped_image, zoom_scale, start_x, start_y, end_x, end_y

    if cropped_image:
        # Adjust zoom scale based on scroll direction
        if event.delta > 0:
            zoom_scale *= 1.1
        else:
            zoom_scale *= 0.9

        # Ensure zoom scale does not go below a certain threshold
        if zoom_scale < 0.1:
            zoom_scale = 0.1

        # Calculate new dimensions based on the latest edited image and the cumulative zoom scale
        new_width = int(cropped_image.width * zoom_scale)
        new_height = int(cropped_image.height * zoom_scale)

        # Resize the latest edited image to the new dimensions
        zoomed_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Display the resized image
        display_image(zoomed_image, edited_canvas)

        # Update the cropping coordinates
        if start_x is not None and start_y is not None and end_x is not None and end_y is not None:
            canvas_width = edited_canvas.winfo_width()
            canvas_height = edited_canvas.winfo_height()
            image_ratio = cropped_image.width / cropped_image.height
            canvas_ratio = canvas_width / canvas_height

            if image_ratio > canvas_ratio:
                displayed_width = canvas_width
                displayed_height = int(canvas_width / image_ratio)
            else:
                displayed_height = canvas_height
                displayed_width = int(canvas_height * image_ratio)

            scale_x = cropped_image.width / displayed_width
            scale_y = cropped_image.height / displayed_height

            offset_x = (canvas_width - displayed_width) // 2
            offset_y = (canvas_height - displayed_height) // 2

            start_x = int((start_x - offset_x) / scale_x)
            start_y = int((start_y - offset_y) / scale_y)
            end_x = int((end_x - offset_x) / scale_x)
            end_y = int((end_y - offset_y) / scale_y)

            # Ensure the coordinates are within image boundaries
            start_x = max(0, min(start_x, cropped_image.width))
            start_y = max(0, min(start_y, cropped_image.height))
            end_x = max(0, min(end_x, cropped_image.width))
            end_y = max(0, min(end_y, cropped_image.height))

            # Update the crop rectangle
            edited_canvas.delete('crop_rectangle')
            edited_canvas.create_rectangle(start_x, start_y, end_x, end_y, outline='red', tag='crop_rectangle')
def show_dual_analysis_window():
    """Create a window for image analysis with histograms and metrics"""
    if not loaded_image or not cropped_image:
        messagebox.showerror("Error", "Please load and modify an image first.")
        return

    dual_analysis_window = Toplevel(root)
    dual_analysis_window.title("Dual Image Analysis")
    dual_analysis_window.geometry("1200x900")

    # Create main container with scrollbar
    main_container = tk.Frame(dual_analysis_window)
    main_container.pack(fill=tk.BOTH, expand=True)

    # Create canvas with scrollbar
    canvas = tk.Canvas(main_container)
    scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create frames for different analyses
    image_frame = tk.Frame(scrollable_frame)
    image_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)

    dual_histogram_frame = tk.Frame(scrollable_frame)
    dual_histogram_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)

    stats_frame = tk.Frame(scrollable_frame)
    stats_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)

    # Pack scrollbar and canvas
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Display images
    original_image = ImageTk.PhotoImage(resize_image_aspect_ratio(loaded_image, (300, 300)))
    original_label = tk.Label(image_frame, image=original_image)
    original_label.image = original_image
    original_label.pack(side=tk.LEFT, padx=10)

    cropped_image = ImageTk.PhotoImage(resize_image_aspect_ratio(cropped_image, (300, 300)))
    cropped_label = tk.Label(image_frame, image=cropped_image)
    cropped_label.image = cropped_image
    cropped_label.pack(side=tk.LEFT, padx=10)

    # Call analysis functions
    compare_histograms(dual_histogram_frame)
    statistical_analysis(stats_frame)
def show_image_pair(parent_frame, title, first_image, second_image):
    section_frame = tk.Frame(parent_frame)
    section_frame.pack(fill=tk.X, pady=10)

    title_label = tk.Label(section_frame, text=title, font=("Arial", 14, "bold"))
    title_label.pack()

    image_container = tk.Frame(section_frame)
    image_container.pack(pady=10)

    max_size = (400, 300)
    first_image_resized = resize_image_aspect_ratio(first_image, max_size)
    
    first_photo = ImageTk.PhotoImage(first_image_resized)
    first_label = tk.Label(image_container, image=first_photo)
    first_label.image = first_photo
    first_label.pack(side=tk.LEFT, padx=10)
    tk.Label(image_container, text="Original Image", font=("Arial", 12)).pack(side=tk.LEFT)

    if second_image:
        second_image_resized = resize_image_aspect_ratio(second_image, max_size)
        second_photo = ImageTk.PhotoImage(second_image_resized)
        second_label = tk.Label(image_container, image=second_photo)
        second_label.image = second_photo
        second_label.pack(side=tk.LEFT, padx=10)
        tk.Label(image_container, text="Modified Image", font=("Arial", 12)).pack(side=tk.LEFT)

def resize_image_aspect_ratio(image, max_size):
    """Resize image maintaining aspect ratio"""
    width, height = image.size
    ratio = min(max_size[0]/width, max_size[1]/height)
    new_size = (int(width*ratio), int(height*ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def compare_histograms(parent_frame):
    """Create histogram comparison"""
    show_image_pair(parent_frame, "Image Flare", loaded_image, cropped_image)

    # Create histogram plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Original image histogram
    original_gray = loaded_image.convert("L")
    original_hist = np.array(original_gray.histogram())
    original_hist_norm = original_hist / original_hist.sum()

    ax1.bar(range(256), original_hist_norm, color='Navy Blue', alpha=0.7)
    ax1.set_title("Original Image Histogram")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Normalized Frequency")
    ax1.grid(True, alpha=0.3)

    # Modified image histogram
    modified_gray = cropped_image.convert("L")
    modified_hist = np.array(modified_gray.histogram())
    modified_hist_norm = modified_hist / modified_hist.sum()

    ax2.bar(range(256), modified_hist_norm, color='green', alpha=0.7)
    ax2.set_title("Modified Image Histogram")
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("Normalized Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Display histograms
    canvas = FigureCanvasTkAgg(fig, parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=10)
    
def show_image_comparison_dialog():
    global loaded_image, cropped_image

    if loaded_image is None or cropped_image is None:
        messagebox.showerror("Error", "Please load and modify an image first.")
        return

    # Create window
    comparison_window = Toplevel(root)
    comparison_window.title("Image Flare - Image Comparison")
    comparison_window.geometry("1200x600")
    comparison_window.configure(bg=chatgpt_bg_color)  # Set the background color

    # Create frames for original and modified images
    original_frame = tk.Frame(comparison_window, bg=chatgpt_bg_color)  # Set the background color
    original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    modified_frame = tk.Frame(comparison_window, bg=chatgpt_bg_color)  # Set the background color
    modified_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Display original image
    original_image = ImageTk.PhotoImage(resize_image_aspect_ratio(loaded_image, (500, 500)))
    original_label = tk.Label(original_frame, image=original_image, bg=chatgpt_bg_color)  # Set the background color
    original_label.image = original_image
    original_label.pack(pady=5)

    tk.Label(original_frame, text="Original Image", font=("Arial", 12, "bold"), bg=chatgpt_bg_color, fg=lighter_gray).pack(pady=5)  # Set the background and text color

    # Display modified image
    modified_image = ImageTk.PhotoImage(resize_image_aspect_ratio(cropped_image, (500, 500)))
    modified_label = tk.Label(modified_frame, image=modified_image, bg=chatgpt_bg_color)  # Set the background color
    modified_label.image = modified_image
    modified_label.pack(pady=5)

    tk.Label(modified_frame, text="Modified Image", font=("Arial", 12, "bold"), bg=chatgpt_bg_color, fg=lighter_gray).pack(pady=5)  # Set the background and text color
def statistical_analysis(parent_frame):
    """Create statistical analysis"""
    def calculate_metrics():
        original_array = np.array(loaded_image.convert('L'))
        modified_array = np.array(cropped_image.convert('L'))

        mse = np.mean((original_array - modified_array) ** 2)
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(max(mse, 1e-10)))

        signal_power = np.mean(original_array ** 2)
        noise = original_array - modified_array
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))

        return mse, psnr, snr, original_array, modified_array

    # Calculate metrics
    mse, psnr, snr, orig_arr, mod_arr = calculate_metrics()

    # Create metrics table
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    # Table data
    table_data = [
        ['Metric', 'Original', 'Modified', 'Quality'],
        ['Mean', f"{np.mean(orig_arr):.2f}", f"{np.mean(mod_arr):.2f}", "-"],
        ['Std Dev', f"{np.std(orig_arr):.2f}", f"{np.std(mod_arr):.2f}", "-"],
        ['MSE', "-", f"{mse:.2f}", "Good" if mse < 1000 else "Poor"],
        ['PSNR (dB)', "-", f"{psnr:.2f}", "Good" if psnr > 30 else "Poor"],
        ['SNR (dB)', "-", f"{snr:.2f}", "Good" if snr > 30 else "Poor"]
    ]

    # Create and style table
    table = ax.table(cellText=table_data[1:],
                    colLabels=table_data[0],
                    loc='center',
                    cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color code cells
    for i in range(len(table_data[0])):
        table[0, i].set_facecolor('#E6E6E6')

    # Color metrics results
    for idx, row in enumerate(table_data[1:], 1):
        if row[3] != "-":
            cell = table[idx, 3]
            cell.set_facecolor('#E6FFE6' if row[3] == "Good" else '#FFE6E6')

    plt.title("Image Quality Metrics", pad=20)
    fig.tight_layout()

    # Display table
    canvas = FigureCanvasTkAgg(fig, parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=10)

    # Add explanation
    explanation_text = """
    Image Quality Metrics Explanation:
    • MSE (Mean Squared Error): Lower values indicate more similarity
      - Good: < 1000
      - Poor: ≥ 1000
    
    • PSNR (Peak Signal-to-Noise Ratio): Higher values indicate better quality
      - Good: > 30 dB
      - Poor: ≤ 30 dB
    
    • SNR (Signal-to-Noise Ratio): Higher values indicate better signal quality
      - Good: > 30 dB
      - Poor: ≤ 30 dB
    
    Statistics:
    • Mean: Average pixel intensity
    • Std Dev: Variation in pixel intensities
    """
    
    explanation = tk.Label(parent_frame, text=explanation_text, 
                         justify=tk.LEFT, wraplength=800)
    explanation.pack(padx=10, pady=10)
def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images"""
    img1_array = np.array(img1.convert('L'), dtype=float)
    img2_array = np.array(img2.convert('L'), dtype=float)
    err = np.mean((img1_array - img2_array) ** 2)
    return err if err != 0 else 1e-10

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = calculate_mse(img1, img2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_snr(img1, img2):
    """Calculate Signal-to-Noise Ratio"""
    img1_array = np.array(img1.convert('L'), dtype=float)
    img2_array = np.array(img2.convert('L'), dtype=float)
    
    noise = img1_array - img2_array
    signal_power = np.mean(img1_array ** 2)
    noise_power = np.mean(noise ** 2)
    
    snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')
    return snr
def show_image_analysis_dialog():
    global loaded_image
    current_noisy_image = None  # Track current noisy image

    if loaded_image is None:
        messagebox.showerror("Error", "Please load an image first.")
        return

    # Create window
    analysis_window = Toplevel(root)
    analysis_window.title("Image Analysis")
    analysis_window.geometry("1200x800")

    # Main frame
    main_frame = tk.Frame(analysis_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Images frame
    images_frame = tk.Frame(main_frame)
    images_frame.pack(fill=tk.X, padx=5, pady=5)

    # Create three columns
    image_columns = []
    for i in range(3):
        column = tk.Frame(images_frame)
        column.pack(side=tk.LEFT, expand=True, padx=10)
        image_columns.append(column)
        
    def setup_histogram(parent, title):
        """Create consistent-sized histogram figures"""
        fig = plt.figure(figsize=(5, 3.5))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        return fig, ax, canvas

    def update_histogram(image, index):
        """Update histogram with consistent display"""
        if image is not None:
            fig, ax, canvas = histogram_plots[index]
            ax.clear()

            # Calculate histogram
            gray_image = image.convert("L")
            histogram = np.array(gray_image.histogram())
            pixel_counts = histogram
            normalized_hist = histogram / histogram.sum()
            
            # Calculate statistics
            total_pixels = np.sum(pixel_counts)
            mean_value = np.sum(np.arange(256) * pixel_counts) / total_pixels
            median_value = np.median(np.repeat(np.arange(256), pixel_counts))
            std_dev = np.sqrt(np.sum(((np.arange(256) - mean_value) ** 2) * pixel_counts) / total_pixels)

            # Plot histogram
            bars = ax.bar(range(256), normalized_hist, color='#03055B', alpha=0.7)

            # Set titles based on index
            titles = ["Original Image", "Noisy Image", "Filtered Image"]
            ax.set_title(f"{titles[index]} Histogram\n" +
                        f"Mean: {mean_value:.1f} | Median: {median_value:.1f}\n" +
                        f"Std Dev: {std_dev:.1f} | Total Pixels: {total_pixels:,}",
                        fontsize=9, pad=10)

            # Add statistical lines
            ax.axvline(mean_value, color='red', linestyle='--', alpha=0.8, label='Mean')
            ax.axvline(median_value, color='green', linestyle='--', alpha=0.8, label='Median')

            # Consistent axis labels and ticks
            ax.set_xlabel("Pixel Intensity", fontsize=8)
            ax.set_ylabel("Normalized Frequency", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=7, loc='upper right')

            # Ensure consistent y-axis limits
            ax.set_ylim(0, max(normalized_hist) * 1.1)

            # Update the canvas
            fig.tight_layout()
            canvas.draw()

    # Setup histograms
    histogram_frames = []
    histogram_plots = []
    for i, title in enumerate(["Original", "Noisy", "Filtered"]):
        hist_frame = tk.Frame(image_columns[i])
        hist_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        container = tk.Frame(hist_frame, width=400, height=300)
        container.pack(expand=True)
        container.pack_propagate(False)
        
        fig, ax, canvas = setup_histogram(container, title)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        histogram_frames.append(hist_frame)
        histogram_plots.append((fig, ax, canvas))

    # Image sections
    original_image = ImageTk.PhotoImage(resize_image_aspect_ratio(loaded_image, (300, 300)))
    original_label = tk.Label(image_columns[0], image=original_image)
    original_label.image = original_image
    original_label.pack(pady=5)

    tk.Label(image_columns[0], text="Original Image", font=("Arial", 12, "bold")).pack(pady=5)

    noise_var = tk.StringVar(value="None")
    noise_options = [
        "None",
        "Gaussian Noise",
        "Rayleigh Noise",
        "Gamma Noise",
        "Exponential Noise",
        "Uniform Noise",
        "Poisson Noise",
        "Quantization Noise",
        "Impulse Noise",
        "Salt and Pepper Noise"
    ]
    noise_dropdown = ttk.Combobox(image_columns[1], textvariable=noise_var, 
                                values=noise_options, state="readonly")
    noise_dropdown.pack(pady=5)

    noisy_image = ImageTk.PhotoImage(resize_image_aspect_ratio(loaded_image, (300, 300)))
    noisy_label = tk.Label(image_columns[1], image=noisy_image)
    noisy_label.image = noisy_image
    noisy_label.pack(pady=5)

    tk.Label(image_columns[1], text="Noisy Image", font=("Arial", 12, "bold")).pack(pady=5)

    filter_var = tk.StringVar(value="None")
    filter_options = [
        "None",
        "Arithmetic Mean Filter",
        "Geometric Mean Filter",
        "Harmonic Mean Filter",
        "Contraharmonic Mean Filter",
        "Median Filter",
        "Min Filter",
        "Max Filter",
        "Midpoint Filter",
        "Alpha Trimmed Mean Filter",
        "Adaptive Filter"
    ]
    filter_dropdown = ttk.Combobox(image_columns[2], textvariable=filter_var, 
                                 values=filter_options, state="readonly")
    filter_dropdown.pack(pady=5)

    filtered_image = ImageTk.PhotoImage(resize_image_aspect_ratio(loaded_image, (300, 300)))
    filtered_label = tk.Label(image_columns[2], image=filtered_image)
    filtered_label.image = filtered_image
    filtered_label.pack(pady=5)

    tk.Label(image_columns[2], text="Filtered Image", font=("Arial", 12, "bold")).pack(pady=5)

    # Metrics table
    metrics_frame = tk.Frame(main_frame)
    metrics_frame.pack(fill=tk.X, padx=10, pady=10)

    metrics_table = tk.Text(metrics_frame, height=10, width=80)
    metrics_table.pack()

    def update_metrics():
        """Update metrics based on current images"""
        orig_arr = np.array(loaded_image)
        mod_arr = np.array(current_noisy_image)

        mse = np.mean((orig_arr - mod_arr) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')
        snr = 10 * np.log10(np.var(orig_arr) / np.var(orig_arr - mod_arr))

        metrics_table.delete(1.0, tk.END)
        metrics_table.insert(tk.END, 
            f"{'Metric':<20}{'Original':<20}{'Modified':<20}{'Quality'}\n" +
            f"{'-'*80}\n" +
            f"{'Mean':<20}{f'{np.mean(orig_arr):.2f}':<20}{f'{np.mean(mod_arr):.2f}':<20}{'-'}\n" +
            f"{'Std Dev':<20}{f'{np.std(orig_arr):.2f}':<20}{f'{np.std(mod_arr):.2f}':<20}{'-'}\n" +
            f"{'MSE':<20}{'-':<20}{f'{mse:.2f}':<20}{'Good' if mse < 1000 else 'Poor'}\n" +
            f"{'PSNR (dB)':<20}{'-':<20}{f'{psnr:.2f}':<20}{'Good' if psnr > 30 else 'Poor'}\n" +
            f"{'SNR (dB)':<20}{'-':<20}{f'{snr:.2f}':<20}{'Good' if snr > 30 else 'Poor'}\n"
        )

    def display_image(image, label):
        """Display image on label"""
        if image is not None:
            resized = resize_image_aspect_ratio(image, (300, 300))
            photo = ImageTk.PhotoImage(resized)
            label.config(image=photo)
            label.image = photo

    def wrap_filter(func):
        """Wrapper for filter functions"""
        def wrapped(image):
            if hasattr(func, '__globals__') and 'cropped_image' in func.__globals__:
                func.__globals__['cropped_image'] = image
            return func()
        return wrapped

    def update_noisy_image(*args):
        """Update noisy image when dropdown selection changes"""
        nonlocal current_noisy_image
        method = noise_var.get()
        
        if method == "None":
            current_noisy_image = loaded_image
        else:
            # Map dropdown selection to noise function
            func_map = {
                "Gaussian Noise": add_gaussian_noise,
                "Rayleigh Noise": add_rayleigh_noise,
                "Gamma Noise": add_gamma_noise,
                "Exponential Noise": add_exponential_noise,
                "Uniform Noise": add_uniform_noise,
                "Poisson Noise": add_poisson_noise,
                "Quantization Noise": add_quantization_noise,
                "Salt and Pepper Noise": add_salt_and_pepper_noise
            }
            
            # Set the global cropped_image temporarily
            global cropped_image
            original_cropped = cropped_image
            cropped_image = loaded_image
            
            try:
                current_noisy_image = func_map[method]()
            finally:
                cropped_image = original_cropped

        display_image(current_noisy_image, noisy_label)
        update_histogram(current_noisy_image, 1)
        update_metrics()
        update_filtered_image()

    def update_filtered_image(*args):
        """Update filtered image when dropdown selection changes"""
        method = filter_var.get()
        if method == "None" or current_noisy_image is None:
            filtered = current_noisy_image
        else:
            func_map = {
                "Arithmetic Mean Filter": apply_arithmetic_filter,
                "Geometric Mean Filter": apply_geometric_filter,
                "Harmonic Mean Filter": apply_harmonic_filter,
                "Contraharmonic Mean Filter": apply_contraharmonic_filter,
                "Median Filter": apply_median_filter,
                "Min Filter": apply_min_filter,
                "Max Filter": apply_max_filter,
                "Midpoint Filter": apply_midpoint_filter,
                "Alpha Trimmed Mean Filter": lambda img: apply_alpha_trimmed_mean_filter(img, d=2),
                "Adaptive Filter": apply_adaptive_filter
            }
            filtered = func_map[method](current_noisy_image)

        display_image(filtered, filtered_label)
        update_histogram(filtered, 2)
        update_metrics()

    # Initial display
    display_image(loaded_image, original_label)
    update_histogram(loaded_image, 0)
    current_noisy_image = loaded_image
    display_image(current_noisy_image, noisy_label)
    update_histogram(current_noisy_image, 1)
    display_image(current_noisy_image, filtered_label)
    update_histogram(current_noisy_image, 2)
    update_metrics()

    # Bind dropdown events
    noise_var.trace_add("write", update_noisy_image)
    filter_var.trace_add("write", update_filtered_image)        
def update_brightness(value):
    try:
        if cropped_image:  # Use cropped_image instead of loaded_image
            # Adjust brightness using PIL's ImageEnhance
            enhancer = ImageEnhance.Brightness(cropped_image)
            enhanced_image = enhancer.enhance(1 + (float(value) / 100))
            display_image(enhanced_image, edited_canvas, fit_to_canvas=True)
    except Exception as e:
        print(f"Error adjusting brightness: {e}")

def live_brightness_update(value):
    global cropped_image, preview_image

    # Ensure cropped_image is loaded
    if cropped_image:
        enhancer = ImageEnhance.Brightness(cropped_image)
        
        # Adjust brightness by multiplying with the slider value (0-100)
        factor = 1 + (float(value) / 100.0)  # Scale from -100 to +100 to 0.0 to 2.0
        preview_image = enhancer.enhance(factor)
        
        # Display the modified image (preview) on the canvas
        display_image(preview_image, edited_canvas, fit_to_canvas=True)


def apply_brightness():
    global brightness_message_shown  # Declare the global variable
    try:
        global cropped_image, undo_stack
        if cropped_image:  # Apply brightness change to cropped_image
            
            # Show a pop-up message the first time the user applies brightness
            if not brightness_message_shown:
                messagebox.showinfo("Brightness Adjustment", "You can adjust the brightness of the image using the slider.")
                brightness_message_shown = True  # Set the flag to True after the first message
            
            enhancer = ImageEnhance.Brightness(cropped_image)
            cropped_image = enhancer.enhance(1 + (float(brightness_slider.get()) / 100))
            display_image(cropped_image, edited_canvas, fit_to_canvas=True)
            push_to_undo_stack(cropped_image)   # Add to undo stack
    except Exception as e:
        print(f"Error applying brightness: {e}")

def update_contrast(value):
    try:
        if cropped_image:  # Use cropped_image instead of loaded_image
            # Adjust contrast using PIL's ImageEnhance
            enhancer = ImageEnhance.Contrast(cropped_image)
            enhanced_image = enhancer.enhance(1 + (float(value) / 100))
            display_image(enhanced_image, edited_canvas, fit_to_canvas=True)
    except Exception as e:
        print(f"Error adjusting contrast: {e}")

def live_contrast_update(value):
    global cropped_image, preview_image

    # Ensure cropped_image is loaded
    if cropped_image:
        enhancer = ImageEnhance.Contrast(cropped_image)
        
        # Adjust contrast by multiplying with the slider value (0-200)
        factor = 1 + (float(value) / 100.0)  # Scale from -100 to +200 to 0.0 to 3.0
        preview_image = enhancer.enhance(factor)
        
        # Display the modified image (preview) on the canvas
        display_image(preview_image, edited_canvas, fit_to_canvas=True)

def apply_contrast():
    global contrast_message_shown  # Declare the global variable
    try:
        global cropped_image, undo_stack
        if cropped_image:  # Apply contrast change to cropped_image
            
            # Show a pop-up message the first time the user applies contrast
            if not contrast_message_shown:
                messagebox.showinfo("Contrast Adjustment", "You can adjust the contrast of the image using the slider.")
                contrast_message_shown = True  # Set the flag to True after the first message
            
            enhancer = ImageEnhance.Contrast(cropped_image)
            cropped_image = enhancer.enhance(1 + (float(contrast_slider.get()) / 100))
            display_image(cropped_image, edited_canvas, fit_to_canvas=True)
            push_to_undo_stack(cropped_image)   # Add to undo stack
    except Exception as e:
        print(f"Error applying contrast: {e}")


def update_sharpness(value):
    try:
        if cropped_image:  # Use cropped_image instead of loaded_image
            # Adjust sharpness using PIL's ImageEnhance
            enhancer = ImageEnhance.Sharpness(cropped_image)
            enhanced_image = enhancer.enhance(1 + (float(value) / 100))
            display_image(enhanced_image, edited_canvas, fit_to_canvas=True)
    except Exception as e:
        print(f"Error adjusting sharpness: {e}")

def live_sharpness_update(value):
    global cropped_image, preview_image

    # Ensure cropped_image is loaded
    if cropped_image:
        enhancer = ImageEnhance.Sharpness(cropped_image)
        
        # Adjust sharpness by multiplying with the slider value (0-100)
        factor = 1 + (float(value) / 100.0)  # Scale from -100 to +100 to 0.0 to 2.0
        preview_image = enhancer.enhance(factor)
        
        # Display the modified image (preview) on the canvas
        display_image(preview_image, edited_canvas, fit_to_canvas=True)

def apply_sharpness():
    global sharpness_message_shown, cropped_image, undo_stack
    if cropped_image:
        if not sharpness_message_shown:
            messagebox.showinfo("Sharpness Adjustment", "You can adjust the sharpness of the image using the slider.")
            sharpness_message_shown = True
        enhancer = ImageEnhance.Sharpness(cropped_image)
        factor = 1 + (float(sharpness_slider.get()) / 100)
        cropped_image = enhancer.enhance(factor)
        display_image(cropped_image, edited_canvas, fit_to_canvas=True)
        push_to_undo_stack(cropped_image)

def update_blurriness(value):
    try:
        if cropped_image:  # Use cropped_image instead of loaded_image
            # Adjust blurriness using PIL's ImageFilter
            blur_radius = float(value)  # The slider value determines the blur radius
            blurred_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            display_image(blurred_image, edited_canvas, fit_to_canvas=True)
    except Exception as e:
        print(f"Error adjusting blurriness: {e}")
def live_blurriness_update(value):
    global cropped_image, preview_image

    # Ensure cropped_image is loaded
    if cropped_image:
        blur_radius = float(value)  # The slider value determines the blur radius
        preview_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Display the modified image (preview) on the canvas
        display_image(preview_image, edited_canvas, fit_to_canvas=True)
    try:
        if cropped_image:  # Apply blurriness change to cropped_image
            blur_radius = float(blur_slider.get())  # Get the value from the blur slider
            cropped_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            display_image(cropped_image, edited_canvas, fit_to_canvas=True)
            push_to_undo_stack(cropped_image)   # Add to undo stack
    except Exception as e:
        print(f"Error applying blurriness: {e}")

def apply_blurriness():
    global blurriness_message_shown  # Declare the global variable
    try:
        global cropped_image, undo_stack
        if cropped_image:  # Apply blurriness change to cropped_image
            
            # Show a pop-up message the first time the user applies blurriness
            if not blurriness_message_shown:
                messagebox.showinfo("Blurriness Adjustment", "You can adjust the blurriness of the image using the slider.")
                blurriness_message_shown = True  # Set the flag to True after the first message
            
            blur_radius = float(blur_slider.get())  # Get the value from the blur slider
            cropped_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            display_image(cropped_image, edited_canvas, fit_to_canvas=True)
            push_to_undo_stack(cropped_image)   # Add to undo stack
    except Exception as e:
        print(f"Error applying blurriness: {e}")

# Function to convert PIL Image to NumPy array
def pil_to_numpy(image):
    return np.asarray(image)  # Using np.asarray to convert PIL Image to NumPy array

# Function to convert NumPy array to PIL Image
def numpy_to_pil(image_array):
    return Image.fromarray(image_array)  # Using Image.fromarray to convert NumPy array to PIL Image

# Apply noise function for the chosen noise type
def apply_noise(noise_function):
    global cropped_image, undo_stack, redo_stack
    
    if cropped_image is None:
        print("No image loaded to apply noise.")
        return
    
    try:
        noisy_image = noise_function()
        if noisy_image is None:
            print("Noise function returned None, unable to apply noise.")
            return
        # Push the current image to undo stack before changing
        push_to_undo_stack(cropped_image)
        cropped_image = noisy_image
        display_image(cropped_image, edited_canvas, fit_to_canvas=True)
    except Exception as e:
        print(f"Error applying noise: {e}")
    # Clear redo stack after applying noise
    redo_stack.clear()
def show_noise_message():
    global noise_message_shown
    if not noise_message_shown:
        messagebox.showinfo("Noise Functionality", "You can now apply the noise to the image.")
        noise_message_shown = True
# Add Gaussian noise
def add_gaussian_noise():
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)  # Convert PIL image to NumPy array
    mean = 0
    std_dev = 25
    noise = np.random.normal(mean, std_dev, img_arr.shape)
    noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return numpy_to_pil(noisy_img)  # Convert NumPy array back to PIL image

# Add Rayleigh noise
# @show_description("This function applies Rayleigh noise to the current image.")
def add_rayleigh_noise():
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)  # Convert PIL image to NumPy array
    scale = 25
    noise = np.random.rayleigh(scale, img_arr.shape)
    noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return numpy_to_pil(noisy_img)  # Convert NumPy array back to PIL image

# Add Gamma noise
# @show_description("This function applies Gamma noise to the current image.")
def add_gamma_noise():
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)  # Convert PIL image to NumPy array
    shape = 2
    scale = 2
    noise = np.random.gamma(shape, scale, img_arr.shape)
    noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return numpy_to_pil(noisy_img)  # Convert NumPy array back to PIL image

# Add Exponential noise
# @show_description("This function applies Exponential noise to the current image.")
def add_exponential_noise():
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)  # Convert PIL image to NumPy array
    scale = 25
    noise = np.random.exponential(scale, img_arr.shape)
    noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return numpy_to_pil(noisy_img)  # Convert NumPy array back to PIL image

# Add Uniform noise
# @show_description("This function applies Uniform noise to the current image.")
def add_uniform_noise():
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)  # Convert PIL image to NumPy array
    noise = np.random.uniform(-50, 50, img_arr.shape)
    noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return numpy_to_pil(noisy_img)  # Convert NumPy array back to PIL image

# @show_description("This function applies Poisson noise to the current image.")
def add_poisson_noise():
    global cropped_image
    if cropped_image is None:
        print("No image loaded to apply Poisson noise.")
        return None

    try:
        img_arr = pil_to_numpy(cropped_image)  # Convert PIL image to NumPy array
        
        # Generate Poisson noise
        noisy_img = np.random.poisson(img_arr).clip(0, 255).astype(np.uint8)
        
        return numpy_to_pil(noisy_img)  # Convert NumPy array back to PIL image
    except Exception as e:
        print(f"Error applying Poisson noise: {e}")
        return None
    
# @show_description("This function applies quantization noise to the current image.")
def add_quantization_noise():
    global cropped_image
    if cropped_image is None:
        print("No image loaded to apply quantization noise.")
        return None

    try:
        img_arr = pil_to_numpy(cropped_image)  # Convert PIL image to NumPy array
        
        # Define the number of quantization levels
        num_levels = 16  # You can adjust this value for more or less quantization
        quantized_img = (img_arr // (256 // num_levels)) * (256 // num_levels)
        
        return numpy_to_pil(quantized_img)  # Convert NumPy array back to PIL image
    except Exception as e:
        print(f"Error applying quantization noise: {e}")
        return None
        
# Add Salt and Pepper noise
# @show_description("This function applies Salt and Pepper noise to the current image.")
def add_salt_and_pepper_noise():
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)  # Convert PIL image to NumPy array
    prob = 0.05
    noisy_img = img_arr.copy()
    
    # Add salt noise
    salt = np.random.rand(*img_arr.shape[:2]) < prob
    noisy_img[salt] = 255
    
    # Add pepper noise
    pepper = np.random.rand(*img_arr.shape[:2]) < prob
    noisy_img[pepper] = 0
    
    return numpy_to_pil(noisy_img)  # Convert NumPy array back to PIL image

def show_denoise_message():
    global denoise_message_shown
    if not denoise_message_shown:
        messagebox.showinfo("Denoise Functionality", "You can now remove noise from the image.")
        denoise_message_shown = True

# Apply denoise function for the chosen denoise type
def apply_denoise(denoise_function,image):
    global cropped_image, undo_stack, redo_stack
    
    if cropped_image is None:
        print("No image loaded to apply denoise.")
        return
    
    try:
        denoised_image = denoise_function(image)
        if denoised_image is None:
            print("Denoise function returned None, unable to apply denoise.")
            return
        
        # Push the current image to undo stack before changing
        push_to_undo_stack(cropped_image)
        cropped_image = denoised_image
        display_image(cropped_image, edited_canvas, fit_to_canvas=True)
    except Exception as e:
        print(f"Error applying denoise: {e}")
    
    # Clear redo stack after applying denoise
    redo_stack.clear()

    
# Function to enhance clarity of the image
def enhance_clarity(image):
    # Convert the image to a numpy array
    img_arr = np.array(image)
    
    # Step 1: Apply Unsharp Masking for sharpening to remove blurriness
    blurred = cv2.GaussianBlur(img_arr, (5, 5), 1.5)  # Apply Gaussian Blur to create a blurred version
    sharpened = cv2.addWeighted(img_arr, 1.5, blurred, -0.5, 0)  # Combine the original and blurred image for sharpening
    
    # Step 2: Optionally, apply a more subtle sharpening filter (optional, you can skip if the result is too strong)
    sharpened = cv2.filter2D(sharpened, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))  # Basic sharpening kernel
    
    # Step 3: Convert the final sharpened result back to PIL format for display
    return Image.fromarray(np.uint8(sharpened))  # Convert the result back to PIL Image format

def remove_gaussian_noise(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)

    # Apply wavelet denoising for Gaussian noise
    try:
        wavelet_denoised = denoise_wavelet(img_arr, method='BayesShrink', mode='soft', channel_axis=-1)
        tv_denoised = denoise_tv_chambolle(wavelet_denoised, weight=0.1, channel_axis=-1)
        
        # Enhance clarity
        enhanced_img = enhance_clarity(numpy_to_pil((tv_denoised * 255).astype(np.uint8)))
        return enhanced_img
    except Exception as e:
        print(f"Error applying Gaussian denoise: {e}")
        return None
    
def remove_rayleigh_noise(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)

    # Apply wavelet denoising for Rayleigh noise
    try:
        wavelet_denoised = denoise_wavelet(img_arr, method='BayesShrink', mode='soft', channel_axis=-1)
        sharpened_img = cv2.addWeighted((wavelet_denoised * 255).astype(np.uint8), 1.2, img_arr, -0.2, 0)
        
        # Enhance clarity
        enhanced_img = enhance_clarity(numpy_to_pil(sharpened_img))
        return enhanced_img
    except Exception as e:
        print(f"Error applying Rayleigh denoise: {e}")
        return None

def remove_gamma_noise(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)

    # Log transformation to stabilize variance
    try:
        img_arr_log = np.log1p(img_arr / 255.0)  # Apply log transformation
        wavelet_denoised = denoise_wavelet(img_arr_log, method='BayesShrink', mode='soft', channel_axis=-1)
        denoised_img = np.expm1(wavelet_denoised) * 255.0  # Reverse log transformation
        denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
        sharpened_img = cv2.addWeighted(denoised_img, 1.3, img_arr, -0.3, 0)
        
        # Enhance clarity
        enhanced_img = enhance_clarity(numpy_to_pil(sharpened_img))
        return enhanced_img
    except Exception as e:
        print(f"Error applying Gamma denoise: {e}")
        return None

def remove_exponential_noise(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image).astype(np.float32) / 255.0

    # Log transformation to stabilize variance
    try:
        img_arr_log = np.log1p(img_arr)  # Apply log transformation
        wavelet_denoised = denoise_wavelet(img_arr_log, method='BayesShrink', mode='soft', channel_axis=-1)
        denoised_img = np.expm1(wavelet_denoised)  # Reverse log transformation
        denoised_img = np.clip(denoised_img, 0, 1) * 255.0
        denoised_img = denoised_img.astype(np.uint8)
        
        # Enhance clarity of the denoised image
        enhanced_img = enhance_clarity(denoised_img)
        return enhanced_img
    except Exception as e:
        print(f"Error applying Exponential denoise: {e}")
        return None    
    
def remove_uniform_noise(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)

    # Apply wavelet denoising for Uniform noise
    try:
        wavelet_denoised = denoise_wavelet(img_arr, method='BayesShrink', mode='soft', channel_axis=-1)
        sharpened_img = cv2.addWeighted((wavelet_denoised * 255).astype(np.uint8), 1.2, img_arr, -0.2, 0)
        
        # Enhance clarity
        enhanced_img = enhance_clarity(numpy_to_pil(sharpened_img))
        return enhanced_img
    except Exception as e:
        print(f"Error applying Uniform denoise: {e}")
        return None
    
def remove_poisson_noise(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)

    # Apply wavelet denoising for Poisson noise
    try:
        wavelet_denoised = denoise_wavelet(img_arr, method='BayesShrink', mode='soft', channel_axis=-1)
        tv_denoised = denoise_tv_chambolle(wavelet_denoised, weight=0.1, channel_axis=-1)
        
        # Enhance clarity
        enhanced_img = enhance_clarity(numpy_to_pil((tv_denoised * 255).astype(np.uint8)))
        return enhanced_img
    except Exception as e:
        print(f"Error removing Poisson noise: {e}")
        return None


def remove_quantization_noise(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)

    # Apply wavelet denoising for Quantization noise
    try:
        wavelet_denoised = denoise_wavelet(img_arr, method='BayesShrink', mode='soft', channel_axis=-1)
        sharpened_img = cv2.addWeighted((wavelet_denoised * 255).astype(np.uint8), 1.2, img_arr, -0.2, 0)
        
        # Enhance clarity
        enhanced_img = enhance_clarity(numpy_to_pil(sharpened_img))
        return enhanced_img
    except Exception as e:
        print(f"Error removing Quantization noise: {e}")
        return None

def remove_salt_and_pepper_noise(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)

    # Apply median filtering to remove salt and pepper noise
    try:
        median_filtered = cv2.medianBlur(img_arr, 5)
        wavelet_denoised = denoise_wavelet(median_filtered, method='BayesShrink', mode='soft', channel_axis=-1)
        
        # Enhance clarity
        enhanced_img = enhance_clarity(numpy_to_pil((wavelet_denoised * 255).astype(np.uint8)))
        return enhanced_img
    except Exception as e:
        print(f"Error applying Salt and Pepper denoise: {e}")
        return None


# Helper functions to convert between PIL and numpy arrays
def pil_to_numpy(image):
    return np.array(image)

def numpy_to_pil(array):
    return Image.fromarray(array.astype(np.uint8))

def show_filter_message():
    global filter_message_shown
    if not filter_message_shown:
        messagebox.showinfo("Filter Functionality", "You can now apply filters to the image.")
        filter_message_shown = True
# Apply filters functions
def apply_arithmetic_filter(image):
    global cropped_image
    img_arr = np.array(cropped_image)  # Convert image to numpy array
    img_arr = img_arr.astype(np.uint8)  # Ensure the input array is a numeric type
    filtered_img = cv2.blur(img_arr, (3, 3))  # 3x3 Arithmetic mean filter
    return Image.fromarray(filtered_img)  # Convert numpy array back to PIL image
    
def apply_geometric_filter(image):
    """Apply geometric mean filter with proper color channel handling"""
    global cropped_image
    img_array = np.array(cropped_image)
    filtered_img = np.zeros_like(img_array, dtype=np.float32)

    # Process each channel separately
    for c in range(img_array.shape[2] if len(img_array.shape) > 2 else 1):
        if len(img_array.shape) > 2:
            channel = img_array[..., c].astype(np.float32)
        else:
            channel = img_array.astype(np.float32)

        # Add small constant to avoid log(0)
        channel = channel + 1e-8
        
        # Apply geometric mean filter
        log_channel = np.log(channel)
        kernel = np.ones((3, 3), np.float32) / 9
        filtered_channel = cv2.filter2D(log_channel, -1, kernel)
        filtered_channel = np.exp(filtered_channel)

        if len(img_array.shape) > 2:
            filtered_img[..., c] = filtered_channel
        else:
            filtered_img = filtered_channel

    # Clip values to valid range
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
    return Image.fromarray(filtered_img)

def apply_harmonic_filter(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)  # Convert PIL to NumPy array
    harmonic_mean = 3 * 3 / cv2.boxFilter(1 / (img_arr + 1e-5), -1, (3, 3))  # Harmonic mean filter
    harmonic_mean = np.clip(harmonic_mean, 0, 255)  # Clip pixel values
    return numpy_to_pil(harmonic_mean)  # Convert back to PIL

def apply_contraharmonic_filter(image, order=1.5):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image).astype(np.float32)
    numerator = cv2.boxFilter(np.power(img_arr, order + 1), -1, (3, 3))
    denominator = cv2.boxFilter(np.power(img_arr, order), -1, (3, 3)) + 1e-5
    contraharmonic_mean = numerator / denominator  # Contraharmonic mean filter
    return numpy_to_pil(np.clip(contraharmonic_mean, 0, 255))


def apply_median_filter(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)
    filtered_img = cv2.medianBlur(img_arr, 3)  # Median filter with 3x3 kernel
    return numpy_to_pil(filtered_img)

def apply_min_filter(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)
    filtered_img = cv2.erode(img_arr, np.ones((3, 3), np.uint8))  # Min filter using erosion
    return numpy_to_pil(filtered_img)


# @show_description("This function applies a maximum filter using morphological dilation.")
def apply_max_filter(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)
    filtered_img = cv2.dilate(img_arr, np.ones((3, 3), np.uint8))  # Max filter using dilation
    return numpy_to_pil(filtered_img)

def apply_midpoint_filter(image):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image).astype(np.float32)
    min_img = cv2.erode(img_arr, np.ones((3, 3), np.uint8))
    max_img = cv2.dilate(img_arr, np.ones((3, 3), np.uint8))
    midpoint_img = (min_img + max_img) / 2  # Midpoint filter
    return numpy_to_pil(np.clip(midpoint_img, 0, 255).astype(np.uint8))

def apply_alpha_trimmed_mean_filter(image, d=2):
    global cropped_image
    img_arr = pil_to_numpy(cropped_image)  # Assuming cropped_image is already defined globally.
    
    # Check the shape of the img_arr array
    if len(img_arr.shape) == 2:
        height, width = img_arr.shape
        channels = 1
    else:
        height, width, channels = img_arr.shape
    
    total_pixels = height * width * channels  # Total number of pixels (elements in the flattened image)
    
    # Flatten the image
    flattened_img = img_arr.flatten()
    
    # Ensure the flattened image size is divisible by 9 (to perform 9-element chunk sorting)
    if len(flattened_img) % 9 != 0:
        # Calculate padding size for the flattened array to make it divisible by 9
        padding_size = 9 - (len(flattened_img) % 9)
        flattened_img = np.pad(flattened_img, (0, padding_size), mode='constant', constant_values=0)
        total_pixels = len(flattened_img)  # Update total_pixels after padding
    
    # Reshape the flattened image into 9-element chunks for sorting
    sorted_pixels = np.sort(flattened_img.reshape(-1, 9), axis=1)
    
    # Perform the alpha-trimmed mean filter (remove 'd' smallest and largest elements)
    trimmed_pixels = sorted_pixels[:, d:-d].mean(axis=1)
    
    # Remove padding before reshaping
    trimmed_pixels = trimmed_pixels[:total_pixels]
    
    # Calculate the new shape
    new_height = int(np.sqrt(len(trimmed_pixels) / channels))
    new_width = int(len(trimmed_pixels) / (new_height * channels))
    
    # Reshape the trimmed image back into the new dimensions (height, width, channels)
    trimmed_img = trimmed_pixels.reshape(new_height, new_width, channels)
    
    # Convert the result back to a PIL image and return
    return numpy_to_pil(trimmed_img.astype(np.uint8))
def apply_adaptive_filter(image):
    global crop_image
    img_arr = pil_to_numpy(cropped_image)
    
    # Check the input image dimensions
    if len(img_arr.shape) not in [2, 3]:
        print("Invalid image dimensions. Expected 2D grayscale or 3D multichannel image.")
        return None
    
    # Apply denoise_bilateral (adaptive bilateral filter)
    try:
        filtered_img = denoise_bilateral(img_arr, sigma_color=0.05, sigma_spatial=15, channel_axis=-1)
        return numpy_to_pil((filtered_img * 255).astype(np.uint8))
    except Exception as e:
        print(f"Error applying Adaptive Filter: {e}")
        return None

def apply_bandpass_filter(image):
    global crop_image
    img_arr = pil_to_numpy(cropped_image)
    
    # Check the input image dimensions
    if len(img_arr.shape) not in [2, 3]:
        print("Invalid image dimensions. Expected 2D grayscale or 3D multichannel image.")
        return None
    
    # Apply bandpass filter
    try:
        # Convert the image to frequency domain using FFT
        f_transform = np.fft.fft2(img_arr)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # Define a bandpass mask (example: allow frequencies within a certain range)
        rows, cols = img_arr.shape[:2]
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        r = 30  # Radius for the bandpass filter
        mask[crow - r:crow + r, ccol - r:ccol + r] = 1
        
        # Apply the mask to the frequency domain
        f_transform_shifted_filtered = f_transform_shifted * mask
        
        # Convert back to spatial domain
        f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
        img_filtered = np.fft.ifft2(f_transform_filtered)
        img_filtered = np.abs(img_filtered)  # Take the magnitude
        
        # Normalize and convert back to PIL image
        img_filtered = (img_filtered / img_filtered.max()) * 255
        return numpy_to_pil(img_filtered.astype(np.uint8))
    except Exception as e:
        print(f"Error applying Bandpass Filter: {e}")
        return None

def apply_bandreject_filter(image):
    global crop_image
    img_arr = pil_to_numpy(cropped_image)
    
    # Check the input image dimensions
    if len(img_arr.shape) not in [2, 3]:
        print("Invalid image dimensions. Expected 2D grayscale or 3D multichannel image.")
        return None
    
    # Apply bandreject filter
    try:
        # Convert the image to frequency domain using FFT
        f_transform = np.fft.fft2(img_arr)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # Define a bandreject mask (example: reject frequencies within a certain range)
        rows, cols = img_arr.shape[:2]
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        r = 30  # Radius for the bandreject filter
        mask[crow - r:crow + r, ccol - r:ccol + r] = 0
        
        # Apply the mask to the frequency domain
        f_transform_shifted_filtered = f_transform_shifted * mask
        
        # Convert back to spatial domain
        f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
        img_filtered = np.fft.ifft2(f_transform_filtered)
        img_filtered = np.abs(img_filtered)  # Take the magnitude
        
        # Normalize and convert back to PIL image
        img_filtered = (img_filtered / img_filtered.max()) * 255
        return numpy_to_pil(img_filtered.astype(np.uint8))
    except Exception as e:
        print(f"Error applying Bandreject Filter: {e}")
        return None

def apply_wiener_filter(image):
    global crop_image
    img_arr = pil_to_numpy(cropped_image)
    
    # Check the input image dimensions
    if len(img_arr.shape) not in [2, 3]:
        print("Invalid image dimensions. Expected 2D grayscale or 3D multichannel image.")
        return None
    
    # Apply Wiener filter
    try:
        # Estimate the noise power
        noise_power = np.var(img_arr)
        
        # Apply Wiener filter
        wiener_filtered_img = cv2.filter2D(img_arr, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) / 9)
        wiener_filtered_img = wiener_filtered_img + (1 - (noise_power / (noise_power + 1))) * (img_arr - wiener_filtered_img)
        
        # Clip values to valid range
        wiener_filtered_img = np.clip(wiener_filtered_img, 0, 255)
        
        # Convert back to PIL image
        return numpy_to_pil(wiener_filtered_img.astype(np.uint8))
    except Exception as e:
        print(f"Error applying Wiener Filter: {e}")
        return None
    
def apply_kalman_filter(image):
    global crop_image
    img_arr = pil_to_numpy(cropped_image)
    
    # Check the input image dimensions
    if len(img_arr.shape) not in [2, 3]:
        print("Invalid image dimensions. Expected 2D grayscale or 3D multichannel image.")
        return None
    
    # Apply Kalman filter
    try:
        # Initialize Kalman filter parameters
        n_iter = 5
        sz = img_arr.shape
        Q = 1e-5  # Process variance
        R = 0.1  # Measurement variance
        xhat = img_arr.copy()  # Initialize state estimate with the original image
        P = np.zeros(sz)  # Initialize error covariance
        K = np.zeros(sz)  # Initialize Kalman gain
        
        # Apply Kalman filter
        for i in range(n_iter):
            # Predict the state estimate and the error covariance
            xhat_pred = xhat
            P_pred = P + Q
            
            # Update the state estimate and the error covariance
            K = P_pred / (P_pred + R)
            xhat = xhat_pred + K * (img_arr - xhat_pred)
            P = (1 - K) * P_pred
        
        # Clip values to valid range
        kalman_filtered_img = np.clip(xhat, 0, 255)
        
        # Convert back to PIL image
        return numpy_to_pil(kalman_filtered_img.astype(np.uint8))
    except Exception as e:
        print(f"Error applying Kalman Filter: {e}")
        return None            
def update_status_bar(x, y):
    status_text.set(f"Coordinates: {x}, {y}")
    

# 1. Top Bar
# Create a frame for the top bar
top_bar = ctk.CTkFrame(root, height=50, bg_color=chatgpt_bg_color)
top_bar.pack(side=tk.TOP, fill=tk.X)

# # Company Name in the center
# company_name_label = ctk.CTkLabel(
#     top_bar, text="Image Flare", font=("Comic Sans MS", 25, "bold"),
# )
# company_name_label.pack(fill=tk.X, expand=True, padx=10)


# Function to show submenu options
def show_menu(menu_title, options):
    # Create a dropdown menu for the given options
    submenu = tk.Menu(root, tearoff=0)
    
    for label, command in options:
        if command in [compare_histograms, statistical_analysis]:
            # Wrap commands requiring a parent frame
            submenu.add_command(label=label, command=lambda cmd=command: create_analysis_frame(cmd))
        else:
            # Add commands normally for other options
            submenu.add_command(label=label, command=command)
    
    # Post the menu at the button's location
    submenu.post(menu_title.winfo_rootx(), menu_title.winfo_rooty() + menu_title.winfo_height())

# Function to dynamically create a parent frame and call analysis functions
def create_analysis_frame(analysis_function):
    # Create a new pop-up window for analysis
    analysis_window = Toplevel(root)
    analysis_window.title(f"{analysis_function.__name__.replace('_', ' ').capitalize()}")

    # Create a frame to pass as `parent_frame`
    analysis_frame = tk.Frame(analysis_window)
    analysis_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Call the analysis function with the created frame
    analysis_function(analysis_frame)

# Menu Container
menu_container = ctk.CTkFrame(top_bar,  bg_color=chatgpt_bg_color)
menu_container.pack(side=tk.LEFT, padx=28)

# Company Name in the center
company_name_label = ctk.CTkLabel(
    top_bar, text="Image Flare", font=("Comic Sans MS", 28, "bold"),
)
company_name_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
# File Menu
file_menu_options = [
    ("Open Image", open_image),
    ("Save Image", save_image),
    ("Close Image", close_image),
    ("Exit", root.quit),
]
file_menu = ctk.CTkButton(
    menu_container, 
    font=("Palanquin Dark", 13, "bold"), 
    text="File", 
    command=lambda: show_menu(file_menu, file_menu_options), 
    fg_color="Navy Blue",# Set button background color to black
    text_color="white"  # Set button text color to white
)
file_menu.pack(side=tk.LEFT, padx=8)


# Analysis Menu
analysis_menu_options = [
    # ("Histogram Analysis",compare_histograms),
    ("Statistical Analysis", statistical_analysis),
    ("Image Analysis", show_image_analysis_dialog),
    ("Compare Both Images", show_image_comparison_dialog)
]
analysis_menu = ctk.CTkButton(
    menu_container,
    font=("Palanquin Dark", 13, "bold"), 
    text="Analysis", 
    command=lambda: show_menu(analysis_menu, analysis_menu_options), 
    fg_color="Navy Blue",  # Set button background color to dark Navy Blue
    text_color="white"  # Set button text color to white
)
analysis_menu.pack(side=tk.LEFT, padx=8)
# # New Button for Analysis (3 Images)
# analysis_3_images_button = ctk.CTkButton(menu_container, text="Analysis (3 Images)", command=show_image_analysis_dialog)
# analysis_3_images_button.pack(side=tk.LEFT, padx=5)


# 2. Left Frame
left_frame = ctk.CTkFrame(root, width=200)
left_frame.pack(side=tk.LEFT, expand=False, fill=tk.Y, padx=(10, 0), pady=(5, 0))

# Noise Frame
noise_frame = ctk.CTkFrame(left_frame, bg_color=chatgpt_bg_color)  # Create a frame for noise operations
noise_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the noise frame

# Noise Label
noise_label = ctk.CTkLabel(noise_frame, text="Noise Operations", font=("Arial", 18, "bold"))
noise_label.pack(side=tk.TOP, pady=(5, 10))  # Add some padding

# Noise operation buttons
gaussian_button = ctk.CTkButton(noise_frame, font=("Palanquin Dark", 13, "bold"), text="Gaussian", command=lambda: [show_noise_message(), apply_noise(add_gaussian_noise)], fg_color="Navy Blue", text_color="white")
gaussian_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

rayleigh_button = ctk.CTkButton(noise_frame, font=("Palanquin Dark", 13, "bold"), text="Rayleigh", command=lambda: [show_noise_message(), apply_noise(add_rayleigh_noise)], fg_color="Navy Blue", text_color="white")
rayleigh_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

gamma_button = ctk.CTkButton(noise_frame, font=("Palanquin Dark", 13, "bold"), text="Gamma", command=lambda: [show_noise_message(), apply_noise(add_gamma_noise)], fg_color="Navy Blue", text_color="white")
gamma_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

exponential_button = ctk.CTkButton(noise_frame, font=("Palanquin Dark", 13, "bold"), text="Exponential", command=lambda: [show_noise_message(), apply_noise(add_exponential_noise)], fg_color="Navy Blue", text_color="white")
exponential_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

uniform_button = ctk.CTkButton(noise_frame, font=("Palanquin Dark", 13, "bold"), text="Uniform", command=lambda: [show_noise_message(), apply_noise(add_uniform_noise)], fg_color="Navy Blue", text_color="white")
uniform_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

poisson_button = ctk.CTkButton(noise_frame, font=("Palanquin Dark", 13, "bold"), text="Poisson", command=lambda: [show_noise_message(), apply_noise(add_poisson_noise)], fg_color="Navy Blue", text_color="white")
poisson_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Initially hide the button

quantization_button = ctk.CTkButton(noise_frame, font=("Palanquin Dark", 13, "bold"), text="Quantization", command=lambda: [show_noise_message(), apply_noise(add_quantization_noise)], fg_color="Navy Blue", text_color="white")
quantization_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Initially hide the button

salt_pepper_button = ctk.CTkButton(noise_frame , font=("Palanquin Dark", 13, "bold"), text="Salt & Pepper", command=lambda: [show_noise_message(), apply_noise(add_salt_and_pepper_noise)], fg_color="Navy Blue", text_color="white")
salt_pepper_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

# Denoise Frame
denoise_frame = ctk.CTkFrame(left_frame, bg_color=chatgpt_bg_color)  # Create a frame for denoise operations
denoise_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the denoise frame

# Denoise Label
denoise_label = ctk.CTkLabel(denoise_frame, text="Denoise Operations", font=("Arial", 18, "bold"))
denoise_label.pack(side=tk.TOP, pady=(5, 10))  # Add some padding

# Denoise operation buttons
gaussian_denoise_button = ctk.CTkButton(denoise_frame, font=("Palanquin Dark", 13, "bold"), text="Remove Gaussian", command=lambda: [show_denoise_message(), apply_denoise(remove_gaussian_noise, crop_image)], fg_color="Navy Blue", text_color="white")
gaussian_denoise_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

rayleigh_denoise_button = ctk.CTkButton(denoise_frame, font=("Palanquin Dark", 13, "bold"), text="Remove Rayleigh", command=lambda: [show_denoise_message(), apply_denoise(remove_rayleigh_noise, crop_image)], fg_color="Navy Blue", text_color="white")
rayleigh_denoise_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

gamma_denoise_button = ctk.CTkButton(denoise_frame, font=("Palanquin Dark", 13, "bold"), text="Remove Gamma", command=lambda: [show_denoise_message(), apply_denoise(remove_gamma_noise, crop_image)], fg_color="Navy Blue", text_color="white")
gamma_denoise_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

exponential_denoise_button = ctk.CTkButton(denoise_frame, font=("Palanquin Dark", 13, "bold"), text="Remove Exponential", command=lambda: [show_denoise_message(), apply_denoise(remove_exponential_noise, crop_image)], fg_color="Navy Blue", text_color="white")
exponential_denoise_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

uniform_denoise_button = ctk.CTkButton(denoise_frame, font=("Palanquin Dark", 13, "bold"), text="Remove Uniform", command=lambda: [show_denoise_message(), apply_denoise(remove_uniform_noise, crop_image)], fg_color="Navy Blue", text_color="white")
uniform_denoise_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

quantization_denoise_button = ctk.CTkButton(denoise_frame, font=("Palanquin Dark", 13, "bold"), text="Remove Quantization", command=lambda: [show_denoise_message(), apply_denoise(remove_quantization_noise, crop_image)], fg_color="Navy Blue", text_color="white")
quantization_denoise_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

poisson_denoise_button = ctk.CTkButton(denoise_frame, font=("Palanquin Dark", 13, "bold"), text="Remove Poisson", command=lambda: [show_denoise_message(), apply_denoise(remove_poisson_noise, crop_image)], fg_color="Navy Blue", text_color="white")
poisson_denoise_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

salt_pepper_denoise_button = ctk.CTkButton(denoise_frame, font=("Palanquin Dark", 13, "bold"), text="Remove Salt & Pepper", command=lambda: [show_denoise_message(), apply_denoise(remove_salt_and_pepper_noise, crop_image)], fg_color="Navy Blue", text_color="white")
salt_pepper_denoise_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)


# 3. Right Frame
right_frame = ctk.CTkFrame(root, width=200)
right_frame.pack(side=tk.RIGHT, expand=False, fill=tk.Y, padx=(0, 10), pady=(5, 0))

# Linear Filters Frame
linear_filters_frame = ctk.CTkFrame(right_frame, bg_color=chatgpt_bg_color)  # Create a frame for linear filters
linear_filters_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the linear filters frame

# Linear Filters Label
linear_filters_label = ctk.CTkLabel(linear_filters_frame, text="Linear Filters", font=("Arial", 18, "bold"))
linear_filters_label.pack(side=tk.TOP, pady=(5, 10))  # Add some padding

# Linear Filter Buttons
arithmetic_filter_button = ctk.CTkButton(linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Arithmetic", command=lambda: [show_filter_message(), apply_filter(apply_arithmetic_filter, crop_image)], fg_color="Navy Blue", text_color="white")
arithmetic_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

geometric_filter_button = ctk.CTkButton(linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Geometric", command=lambda: [show_filter_message(), apply_filter(apply_geometric_filter, crop_image)], fg_color="Navy Blue", text_color="white")
geometric_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

harmonic_filter_button = ctk.CTkButton(linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Harmonic", command=lambda: [show_filter_message(), apply_filter(apply_harmonic_filter, crop_image)], fg_color="Navy Blue", text_color="white")
harmonic_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

contraharmonic_filter_button = ctk.CTkButton(linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Contraharmonic", command=lambda: [show_filter_message(), apply_filter(apply_contraharmonic_filter, crop_image)], fg_color="Navy Blue", text_color="white")
contraharmonic_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button


# Non-Linear Filters Frame
non_linear_filters_frame = ctk.CTkFrame(right_frame, bg_color=chatgpt_bg_color)  # Create a frame for non-linear filters
non_linear_filters_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the non-linear filters frame

# Non-Linear Filters Label
non_linear_filters_label = ctk.CTkLabel(non_linear_filters_frame, text="Non-Linear Filters", font=("Arial", 18, "bold"))
non_linear_filters_label.pack(side=tk.TOP, pady=(5, 10))  # Add some padding

# Non-Linear Filter Buttons

median_filter_button = ctk.CTkButton(non_linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Median", command=lambda: [show_filter_message(), apply_filter(apply_median_filter, crop_image)], fg_color="Navy Blue", text_color="white")
median_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

min_filter_button = ctk.CTkButton(non_linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Min", command=lambda: [show_filter_message(), apply_filter(apply_min_filter, crop_image)], fg_color="Navy Blue", text_color="white")
min_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

max_filter_button = ctk.CTkButton(non_linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Max", command=lambda: [show_filter_message(), apply_filter(apply_max_filter, crop_image)], fg_color="Navy Blue", text_color="white")
max_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

midpoint_filter_button = ctk.CTkButton(non_linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Midpoint", command=lambda: [show_filter_message(), apply_filter(apply_midpoint_filter, crop_image)], fg_color="Navy Blue", text_color="white")
midpoint_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

alpha_filter_button = ctk.CTkButton(non_linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Alpha Trimmed Mean", command=lambda: [show_filter_message(), apply_filter(apply_alpha_trimmed_mean_filter, crop_image)], fg_color="Navy Blue", text_color="white")
alpha_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

adaptive_filter_button = ctk.CTkButton(non_linear_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Adaptive", command=lambda: [show_filter_message(), apply_filter(apply_adaptive_filter, crop_image)], fg_color="Navy Blue", text_color="white")
adaptive_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

# Frequency Filters Frame
frequency_filters_frame = ctk.CTkFrame(right_frame, bg_color=chatgpt_bg_color)  # Create a frame for frequency filters
frequency_filters_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the frequency filters frame

# Frequency Filters Label
frequency_filters_label = ctk.CTkLabel(frequency_filters_frame, font=("Arial", 18, "bold"), text="Frequency Filters")
frequency_filters_label.pack(side=tk.TOP, pady=(5, 10))  # Add some padding

# Frequency Filter Buttons

# Bandpass Filter Button
bandpass_filter_button = ctk.CTkButton(frequency_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Bandpass", command=lambda: [show_filter_message(), apply_filter(apply_bandpass_filter, crop_image)], fg_color="Navy Blue", text_color="white")
bandpass_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

bandreject_filter_button = ctk.CTkButton(frequency_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Bandreject", command=lambda: [show_filter_message(), apply_filter(apply_bandreject_filter, crop_image)], fg_color="Navy Blue", text_color="white")
bandreject_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

wiener_filter_button = ctk.CTkButton(frequency_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Wiener", command=lambda: [show_filter_message(), apply_filter(apply_wiener_filter, crop_image)], fg_color="Navy Blue", text_color="white")
wiener_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

kalman_filter_button = ctk.CTkButton(frequency_filters_frame, font=("Palanquin Dark", 13, "bold"), text="Kalman", command=lambda: [show_filter_message(), apply_filter(apply_kalman_filter, crop_image)], fg_color="Navy Blue", text_color="white")
kalman_filter_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button


# 4. Bottom Bar
bottom_bar = ctk.CTkFrame(root, height=10, bg_color="white")  # Increased height for buttons
bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Horizontal Frame for Coordinates, Buttons, and Resize Controls
controls_frame = ctk.CTkFrame(bottom_bar, bg_color=chatgpt_bg_color)  # Create a frame for controls
controls_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(0, 0))  # Pack the controls frame with no top padding

# Coordinates Display (on the left)
status_text = tk.StringVar()
status_label = ctk.CTkLabel(controls_frame,font=("Palanquin Dark", 13, "bold"), textvariable=status_text, anchor=tk.W, bg_color=chatgpt_bg_color, text_color=lighter_gray, width=160, height=40)  # Adjust width and height
status_label.pack(side=tk.LEFT, anchor='w', pady=(0, 0))  # Align to the left

# Undo Button with specified width and height
undo_button = ctk.CTkButton(controls_frame, font=("Palanquin Dark", 13, "bold"), text="Undo", command=undo, width=100, height=30, fg_color="Navy Blue", text_color="white")  # Adjust width and height
undo_button.pack(side=tk.LEFT, anchor='w', padx=(4, 0), pady=0)  # Align to the left

# Reset Button with specified width and height
reset_button = ctk.CTkButton(controls_frame, font=("Palanquin Dark", 13, "bold"), text="Reset", command=reset_process, width=100, height=30, fg_color="Navy Blue", text_color="white")  # Adjust width and height
reset_button.pack(side=tk.LEFT, anchor='w', padx=(4, 0), pady=0)  # Align to the left

# Redo Button with specified width and height
redo_button = ctk.CTkButton(controls_frame, font=("Palanquin Dark", 13, "bold"),text="Redo", command=redo, width=100, height=30, fg_color="Navy Blue", text_color="white")  # Adjust width and height
redo_button.pack(side=tk.LEFT, anchor='w', padx=(4, 0), pady=0)  # Align to the left

# Resize Controls
resize_label = ctk.CTkLabel(controls_frame, font=("Palanquin Dark", 13, "bold"), text="Resize Image", anchor=tk.W, bg_color=chatgpt_bg_color, text_color=lighter_gray)
resize_label.pack(side=tk.LEFT, padx=5, pady=0)  # Remove extra padding

resize_label_width = ctk.CTkLabel(controls_frame, font=("Palanquin Dark", 13, "bold"), text="Width:", anchor=tk.W, bg_color=chatgpt_bg_color, text_color=lighter_gray)
resize_label_width.pack(side=tk.LEFT, padx=5, pady=0)  # Remove extra padding

resize_entry_width = ctk.CTkEntry(controls_frame, font=("Palanquin Dark", 13, "bold"), placeholder_text="Width in pixels", width=50)
resize_entry_width.pack(side=tk.LEFT, padx=5, pady=0)  # Remove extra padding

resize_label_height = ctk.CTkLabel(controls_frame, font=("Palanquin Dark", 13, "bold"), text="Height:", anchor=tk.W, bg_color=chatgpt_bg_color, text_color=lighter_gray)
resize_label_height.pack(side=tk.LEFT, padx=5, pady=0)  # Remove extra padding

resize_entry_height = ctk.CTkEntry(controls_frame, font=("Palanquin Dark", 13, "bold"), placeholder_text="Height in pixels", width=50)
resize_entry_height.pack(side=tk.LEFT, padx=5, pady=0)  # Remove extra padding

apply_resize_button = ctk.CTkButton(controls_frame, font=("Palanquin Dark", 13, "bold"), text="Apply Resize", command=apply_resize, width=100, height=30, fg_color="Navy Blue", text_color="white")  # Adjust width and height
apply_resize_button.pack(side=tk.LEFT, padx=5, pady=0)  # Remove extra padding

# Operations Frame
operations_frame = ctk.CTkFrame(bottom_bar, bg_color=chatgpt_bg_color)  # Create a frame for operations
operations_frame.pack(side=tk.LEFT, fill=tk.Y, padx=150, pady=5)  # Pack the operations frame

# Crop Image Button
crop_button = ctk.CTkButton(operations_frame, font=("Palanquin Dark", 13, "bold"), text="Crop Image", command=crop_image, fg_color="Navy Blue", text_color="white")
crop_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

# Convert to Black & White Button
black_white_button = ctk.CTkButton(operations_frame, font=("Palanquin Dark", 13, "bold"), text="Convert to Grayscale", command=convert_to_black_white, fg_color="Navy Blue", text_color="white")
black_white_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

# # Convert to Color Button
# color_button = ctk.CTkButton(operations_frame, text="Convert to Color", command=convert_to_color, fg_color="Navy Blue", text_color="white")
# color_button.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)  # Pack the button

right_frame = ctk.CTkFrame(bottom_bar, bg_color=chatgpt_bg_color)  # Create a frame for the right side
right_frame.pack(side=tk.RIGHT, padx=10)  # Pack the right frame

# Brightness Slider and Button
brightness_frame = ctk.CTkFrame(right_frame)  # Create a frame for brightness slider and button
brightness_frame.pack(side=tk.TOP, pady=5)  # Pack the brightness frame

apply_brightness_button = ctk.CTkButton(brightness_frame, font=("Palanquin Dark", 13, "bold"), text="Brightness", command=apply_brightness, fg_color="Navy Blue", text_color="white")
apply_brightness_button.pack(side=tk.LEFT)  # Place the button on the left

brightness_slider = ctk.CTkSlider(brightness_frame, from_=-100, to=100, number_of_steps=200, command=lambda val: update_brightness(val))
brightness_slider.pack(side=tk.LEFT, padx=5)  # Place the slider to the right of the button

# Contrast Slider and Button
contrast_frame = ctk.CTkFrame(right_frame)  # Create a frame for contrast slider and button
contrast_frame.pack(side=tk.TOP, pady=5)  # Pack the contrast frame

apply_contrast_button = ctk.CTkButton(contrast_frame, font=("Palanquin Dark", 13, "bold"), text="Contrast", command=apply_contrast, fg_color="Navy Blue", text_color="white")
apply_contrast_button.pack(side=tk.LEFT)  # Place the button on the left

contrast_slider = ctk.CTkSlider(contrast_frame, from_=-100, to=100, number_of_steps=200, command=lambda val: update_contrast(val))
contrast_slider.pack(side=tk.LEFT, padx=5)  # Place the slider to the right of the button

# Sharpness Slider and Button
sharpness_frame = ctk.CTkFrame(right_frame)  # Create a frame for sharpness slider and button
sharpness_frame.pack(side=tk.TOP, pady=5)  # Pack the sharpness frame

apply_sharpness_button = ctk.CTkButton(sharpness_frame, font=("Palanquin Dark", 13, "bold"), text="Sharpness", command=apply_sharpness, fg_color="Navy Blue", text_color="white")
apply_sharpness_button.pack(side=tk.LEFT)  # Place the button on the left

sharpness_slider = ctk.CTkSlider(sharpness_frame, from_=-100, to=100, number_of_steps=200, command=lambda val: update_sharpness(val))
sharpness_slider.pack(side=tk.LEFT, padx=5)  # Place the slider to the right of the button

# Blurriness Slider and Button
blur_frame = ctk.CTkFrame(right_frame)  # Create a frame for blurriness slider and button
blur_frame.pack(side=tk.TOP, pady=5)  # Pack the blur frame

apply_blurriness_button = ctk.CTkButton(blur_frame, font=("Palanquin Dark", 13, "bold"), text="Blurriness" , command=apply_blurriness, fg_color="Navy Blue", text_color="white")
apply_blurriness_button.pack(side=tk.LEFT)  # Place the button on the left

blur_slider = ctk.CTkSlider(blur_frame, from_=0, to=10, number_of_steps=100, command=lambda val: update_blurriness(val)) 
blur_slider.pack(side=tk.LEFT, padx=5)  # Place the slider to the right of the button
blur_slider.set(0)

# Create a canvas for displaying the edited image
edited_canvas = ctk.CTkCanvas(root, bg="white")
edited_canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=(0, 0), pady=(5, 0))

background_image_path = r"/Users/sonubodat/Downloads/1.png"  # Using raw string notation
background_image = Image.open(background_image_path)

def display_background_image():
    global background_image_id
    # Get the dimensions of the canvas
    canvas_width = edited_canvas.winfo_width()
    canvas_height = edited_canvas.winfo_height()

    # Resize the background image to fit the canvas dimensions
    resized_image = background_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

    # Clear the canvas
    edited_canvas.delete("all")

    # Convert the resized image to PhotoImage
    bg_image_tk = ImageTk.PhotoImage(resized_image)

    # Display the image on the canvas
    background_image_id = edited_canvas.create_image(0, 0, anchor='nw', image=bg_image_tk)
    edited_canvas.image = bg_image_tk  # Keep a reference to avoid garbage collection

# Ensure the image adjusts when the canvas is resized
edited_canvas.bind("<Configure>", lambda event: display_background_image())

# Display the images if loaded
if loaded_image:
    display_image(loaded_image, edited_canvas, fit_to_canvas=True)  # Display only in edited_canvas

if cropped_image:
    display_image(cropped_image, edited_canvas, fit_to_canvas=True)  # Display cropped image in edited_canvas

# Bind mouse events for cropping and updates
edited_canvas.bind('<ButtonPress-1>', start_crop)
edited_canvas.bind('<B1-Motion>', draw_crop)
edited_canvas.bind("<MouseWheel>", zoom)
edited_canvas.bind("<B3-Motion>", move_image)
edited_canvas.bind('<Motion>', lambda event: update_status_bar(event.x, event.y))

# @show_description("This function allows you to adjust the brightness of the current image.")
def show_brightness_button():
    brightness_slider.pack(side=tk.LEFT, padx=5, pady=5)
    apply_brightness_button.pack(side=tk.LEFT, padx=5, pady=5)

# @show_description("This function allows you to adjust the contrast of the current image.")
def show_contrast_button():
    contrast_slider.pack(side=tk.LEFT, padx=5, pady=5)
    apply_contrast_button.pack(side=tk.LEFT, padx=5, pady=5)

# @show_description("This function allows you to adjust the sharpness of the current image.")
def show_sharpness_button():
    sharpness_slider.pack(side=tk.LEFT, padx=5, pady=5)
    apply_sharpness_button.pack(side=tk.LEFT, padx=5, pady=5)

# @show_description("This function allows you to adjust the blurriness of the current image.")
def show_blurriness_button():
    blur_slider.set(0)
    blur_slider.pack(side=tk.LEFT, padx=5, pady=5)
    apply_blurriness_button.pack(side=tk.LEFT, padx=5, pady=5)

# Apply filter function for the chosen filter type
def apply_filter(filter_function, image):
    global cropped_image, undo_stack, redo_stack
    
    if cropped_image is None:
        print("No image loaded to apply filter.")
        return
    try:
        filtered_image = filter_function(image)
        if filtered_image is None:
            print("Filter function returned None, unable to apply filter.")
            return
        # Push the current image to undo stack before changing
        push_to_undo_stack(cropped_image)
        cropped_image = filtered_image
        display_image(cropped_image, edited_canvas, fit_to_canvas=True)
    except Exception as e:
        print(f"Error applying filter: {e}")
    # Clear redo stack after applying filter
    redo_stack.clear()

# Call this function at the start of your application
display_background_image() 

# Update status bar coordinates to (0, 0) initially
update_status_bar(0, 0)

# Start the application
root.mainloop()
