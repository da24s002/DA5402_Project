import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import requests
import json
import sys
# import torchvision.transforms as transforms
import base64

## default url
url = "http://localhost:5002/invocations/"

## image saving url
img_save_url = "http://localhost:8000/save-image/"

all_predictions = []

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.float()),
#     transforms.Resize((224, 224)),
#     transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
# ])


def save_image(image_vector, class_name):
    base64_encoded_vector = base64.b64encode(image_vector)
    
    # Convert bytes to string
    base64_string = base64_encoded_vector.decode('utf-8')
    payload = {
        "image": base64_string,
        "category": str(class_name)
    }

    response = requests.post(img_save_url, json=payload)
    print(response.text)

def predict(image_vector):
    # Uses the loaded model to make a prediction
    print(image_vector.shape, image_vector.dtype)
    
    print(image_vector.shape, image_vector.dtype)
    print(sys.getsizeof(image_vector))
    base64_encoded_vector = base64.b64encode(image_vector)
    
    # Convert bytes to string
    base64_string = base64_encoded_vector.decode('utf-8')
    payload = {
        "inputs": base64_string
    }

    response = requests.post(url, json=payload)
    # plt.imshow(image_vector.reshape((28,28)))
    # plt.show()
    print(response.text)
    
    global all_predictions
    
    if response.status_code == 200:
        prediction_response = json.loads(response.text)["predictions"]
        all_predictions = prediction_response[0]  # Store all predictions
        return prediction_response[0][0]  # Return the top prediction
    else:
        messagebox.showerror("Error", "Error in fetching prediction from API")
        return None


# Drawing application class
class DrawingApp:
    def __init__(self, root):
        
        self.root = root
        self.root.title("My Drawing Canvas 28x28")
        self.all_predictions = []

        # Canvas settings
        self.canvas_size = 680  # Canvas size in pixels
        self.image_size = 28  # Image size for vectorization
        self.brush_size = 1  # Size of the white brush

        # Add drawing mode state
        self.drawing_mode = "draw"  # Can be "draw" or "erase"

        # Canvas for drawing
        self.canvas = tk.Canvas(root, bg="black", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Creation of the image and the object for drawing
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        # Action buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        
        self.predict_button = tk.Button(self.button_frame, text="  Predict The Drawing  ", command=self.predict_image)
        self.predict_button.pack(side="left")

        # Add eraser button
        self.eraser_button = tk.Button(self.button_frame, text="  Eraser  ", command=self.toggle_eraser)
        self.eraser_button.pack(side="left", padx=10)

        self.clear_button = tk.Button(self.button_frame, text="  Erase All  ", command=self.clear_canvas)
        self.clear_button.pack(side="right")

        # Drawing events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)
        
        # Track the last prediction for feedback
        self.last_prediction = None
        self.last_image_data = None
        # Add these to your initialization
        self.prev_x = None
        self.prev_y = None
        
        # Also bind to mouse release to reset tracking
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)

    # def paint(self, event):
    #     # Draw on the screen and on the image
    #     x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
    #     x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        
    #     # Draw on the canvas (screen) with a white brush
    #     self.canvas.create_oval(x1, y1, x2, y2, fill="yellow", outline="yellow")

    #     # Draw on the 28x28 image for vectorization
    #     scaled_x1, scaled_y1 = (x1 * self.image_size // self.canvas_size), (y1 * self.image_size // self.canvas_size)
    #     scaled_x2, scaled_y2 = (x2 * self.image_size // self.canvas_size), (y2 * self.image_size // self.canvas_size)
    #     self.draw.ellipse([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill="yellow")
    def paint(self, event):
        x, y = event.x, event.y
        
        # Set colors based on drawing mode
        fill_color = "black" if self.drawing_mode == "erase" else "white"
        self.brush_size = 10 if self.drawing_mode == "erase" else 1
        
        # Draw on the canvas (screen)
        x1, y1 = (x - self.brush_size), (y - self.brush_size)
        x2, y2 = (x + self.brush_size), (y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill=fill_color, outline=fill_color)
        
        # Connect points with lines for smooth drawing
        if self.prev_x is not None and self.prev_y is not None:
            # Draw a line connecting the previous point to the current point
            self.canvas.create_line(self.prev_x, self.prev_y, x, y, 
                                width=self.brush_size*2, fill=fill_color, 
                                capstyle=tk.ROUND, smooth=True)
            
            # Also draw the line on the 28x28 image
            prev_scaled_x = self.prev_x * self.image_size // self.canvas_size
            prev_scaled_y = self.prev_y * self.image_size // self.canvas_size
            scaled_x = x * self.image_size // self.canvas_size
            scaled_y = y * self.image_size // self.canvas_size
            
            # Draw line on the image
            self.draw.line([(prev_scaled_x, prev_scaled_y), (scaled_x, scaled_y)], 
                        fill=fill_color, width=max(1, self.brush_size * self.image_size // self.canvas_size))
        
        # Draw the current point on the 28x28 image
        scaled_x1 = x1 * self.image_size // self.canvas_size
        scaled_y1 = y1 * self.image_size // self.canvas_size
        scaled_x2 = x2 * self.image_size // self.canvas_size
        scaled_y2 = y2 * self.image_size // self.canvas_size
        self.draw.ellipse([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill=fill_color)
        
        # Update previous position
        self.prev_x = x
        self.prev_y = y


    def toggle_eraser(self):
        if self.drawing_mode == "draw":
            self.drawing_mode = "erase"
            self.eraser_button.config(relief=tk.SUNKEN, text="  Drawing  ")
        else:
            self.drawing_mode = "draw"
            self.eraser_button.config(relief=tk.RAISED, text="  Eraser  ")

    


    def reset_position(self, event):
        # Reset tracking when mouse is released
        self.prev_x = None
        self.prev_y = None

    def predict_image(self):
        # Convert the image to a vector and normalize the values (0 to 1)
        image_data = np.array(self.image).reshape(1, -1) / 255.0
        image_data = image_data * 255
        image_data = np.uint8(image_data.reshape(784))
        self.last_image_data = image_data.copy()
        
        prediction = predict(image_data)
        if prediction is not None:
            self.last_prediction = prediction
            
            # Store all predictions in the instance
            global all_predictions
            self.all_predictions = all_predictions  # Use the global all_predictions
            
            self.show_prediction_confirmation(prediction)


    def show_prediction_confirmation(self, prediction):
        # Create a custom dialog for prediction confirmation
        confirmation_dialog = tk.Toplevel(self.root)
        confirmation_dialog.title("Prediction Result")
        # confirmation_dialog.geometry("500x500")
        confirmation_dialog.transient(self.root)
        confirmation_dialog.grab_set()
        
        # Add message and buttons
        message_label = tk.Label(
            confirmation_dialog, 
            text=f"The model predicts: {prediction}\nIs this correct?",
            font=("Arial", 10)  # Smaller font size
        )
        message_label.pack(pady=10)
        
        button_frame = tk.Frame(confirmation_dialog)
        button_frame.pack(pady=10)
        
        accept_button = tk.Button(
            button_frame, 
            text="Yes, correct", 
            command=lambda: self.handle_prediction_response(confirmation_dialog, True),
            font=("Arial", 9)  # Smaller font for buttons
        )
        accept_button.pack(side="left", padx=10)
        
        reject_button = tk.Button(
            button_frame, 
            text="No, incorrect", 
            command=lambda: self.handle_prediction_response(confirmation_dialog, False),
            font=("Arial", 9)  # Smaller font for buttons
        )
        reject_button.pack(side="right", padx=10)
        confirmation_dialog.minsize(300, 150)

    def handle_prediction_response(self, dialog, is_accepted):
        dialog.destroy()
        
        if not is_accepted:
            # Create a new dialog with buttons for each prediction
            correction_dialog = tk.Toplevel(self.root)
            correction_dialog.title("Choose Correct Class")
            correction_dialog.transient(self.root)
            correction_dialog.grab_set()
            
            # Add instructions
            instruction_label = tk.Label(
                correction_dialog,
                text="Select the correct class:",
                font=("Arial", 10)
            )
            instruction_label.pack(pady=10)
            
            # Create a frame for the buttons
            buttons_frame = tk.Frame(correction_dialog)
            buttons_frame.pack(pady=10)
            
            # Get the top predictions from all_predictions
            # If all_predictions is empty or not available, show buttons for all digits 0-9
            if not hasattr(self, 'all_predictions') or not self.all_predictions:
                predictions_to_show = list(range(10))
            else:
                predictions_to_show = self.all_predictions
            predictions_to_show.append("other")
            
            # Create a button for each prediction
            for value in predictions_to_show[1:]:
                width = 3
                # Use a lambda with a default argument to capture the current value
                if (value == "other"):
                    width = 6
                
                button = tk.Button(
                    buttons_frame,
                    text=str(value),
                    width=width,
                    font=("Arial", 12, "bold"),
                    command=lambda v=value: self.submit_correction(correction_dialog, v)
                )
                button.pack(side="left", padx=5)

            # button = tk.Button(
            #         buttons_frame,
            #         text=str("other"),
            #         width=6,
            #         font=("Arial", 12, "bold"),
            #         command=self.submit_correction(correction_dialog, "other")
            #     )
            # button.pack(side="left", padx=5)            
            # Set a minimum size for the dialog
            correction_dialog.minsize(300, 150)
            
            # Center the dialog
            correction_dialog.update_idletasks()
            width = correction_dialog.winfo_width()
            height = correction_dialog.winfo_height()
            x = (correction_dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (correction_dialog.winfo_screenheight() // 2) - (height // 2)
            correction_dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        else:
            messagebox.showinfo("Thank You", "Thank you for confirming the prediction!")

        self.clear_canvas()

    def submit_correction(self, dialog, correct_value):
        dialog.destroy()
        save_image(self.last_image_data, correct_value)
        messagebox.showinfo("Thank You", f"Thank you for your feedback. The correct class is {correct_value}.")
        
        # Here you could add code to save this feedback for model improvement


    def clear_canvas(self):
        # Clears the canvas and creates a new black image
        self.canvas.delete("all")
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.last_prediction = None
        # self.last_image_data = None

# Application initialization
if __name__ == "__main__":
    print(sys.argv)
    if (len(sys.argv) == 2):
        url = sys.argv[1]
    root = tk.Tk()
    root.tk.call('tk','scaling',4.0)
    app = DrawingApp(root)
    root.mainloop()
