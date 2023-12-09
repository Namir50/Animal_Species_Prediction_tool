import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Define a function to predict the animal species from an image
def predict_animal_species(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array =np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    return decoded_predictions

# Create a Tkinter window with a larger size
root = tk.Tk()
root.title("Animal Species Detection")

# Set the size of the window (width x height)
root.geometry("800x600")

root.configure(bg="#E6E6E6")
title_label = tk.Label(root, text="Welcome Animal Lover \n This is Animal Species Prediction Tool", font=("Helvetica", 16), bg="#E6E6E6")
title_label.pack(pady=20)

# Function to handle image selection
def select_image():
    image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        predictions = predict_animal_species(image_path)

        # Display the selected image
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

        # Display predictions
        predictions_text.set("Predicted animal species:\n")
        for i, (_, label, score) in enumerate(predictions):
            predictions_text.set(predictions_text.get() + f"{i + 1}: {label} ({score*100:.2f})\n")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image ,bg="#4CAF50", fg="white", font=("Helvetica", 14))
select_button.pack()

button_frame = tk.Frame(root, bg="#E6E6E6")
button_frame.pack(pady=20)  


# Create a label to display the selected image
img_label = tk.Label(root)
img_label.pack()

# Create a label to display predictions
predictions_text = tk.StringVar()
predictions_label = tk.Label(root, textvariable=predictions_text)
predictions_label.pack()

root.mainloop()
