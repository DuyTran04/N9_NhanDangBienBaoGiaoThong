import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
import numpy as np
import cv2
import pickle
from PIL import Image, ImageTk
from tkinter import BOTTOM, TOP

# Load the trained model
with open("model_trained.h5", "rb") as model_file:
    model = pickle.load(model_file)

# Dictionary to label all traffic sign classes
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Nhận dạng biển báo giao thông')
top.configure(background='#ffffff')

label = Label(top, background='#ffffff', font=('arial', 15, 'bold'))
sign_image = Label(top)

def preprocess_image(img):
    """
    Preprocess the uploaded image (resize, grayscale, normalize).
    """
    img = img.resize((32, 32))  # Resize to 32x32 like the training data
    img = np.array(img)
    
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Normalize the image
    img = img / 255.0
    
    # Reshape image for model input (1, 32, 32, 1)
    img = img.reshape(1, 32, 32, 1)
    return img

def classify(file_path):
    """
    Function to classify the image using the loaded model.
    """
    try:
        img = Image.open(file_path)
        img = preprocess_image(img)
        
        # Make prediction using the model
        pred = model.predict(img)
        class_index = np.argmax(pred)  # Get the index with the highest probability
        sign = classes[class_index]  # Get the class label
        label.configure(foreground='#011638', text=f'{sign}\nĐộ chính xác: {np.max(pred)*100:.2f}%')
    except Exception as e:
        print(f"Error during prediction: {e}")
        label.configure(foreground='#011638', text="Prediction Error")

def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận dạng", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(e)

# Function to show results
def show_results():
    """Display the recognition results."""
    messagebox.showinfo("Kết quả", label.cget("text"))

# Function for settings
def open_settings():
    """Open a settings window to adjust image quality, processing speed."""
    settings_window = tk.Toplevel(top)
    settings_window.title("Cài đặt")
    settings_window.geometry('400x300')
    Label(settings_window, text="Cài đặt hệ thống", font=('arial', 16)).pack(pady=20)
    # You can add additional settings options here.
    Label(settings_window, text="Tùy chọn cài đặt chất lượng hình ảnh").pack(pady=10)

# Exit the program
def exit_program():
    """Close the application."""
    if messagebox.askyesno("Thoát", "Bạn có chắc chắn muốn thoát?"):
        top.quit()

# Function to run the Traffic Sign Classification using camera
def run_traffic_sign_classification():
    """Start video stream and classify traffic signs from live camera feed."""
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height
    cap.set(10, 180)  # Set brightness

    while cap.isOpened():  # Check if the camera opened successfully
        success, imgOriginal = cap.read()

        if not success:
            break

        # Preprocess the frame from the camera
        img = np.asarray(imgOriginal)
        img = cv2.resize(img, (32, 32))  # Resize to match model input
        img = preprocess_image_for_camera(img)  # Apply preprocessing
        img = img.reshape(1, 32, 32, 1)

        # Perform prediction
        predictions = model.predict(img)
        class_index = np.argmax(predictions)  # Get predicted class index
        probabilityValue = np.max(predictions)  # Get probability of prediction

        # If probability is above threshold, display the result
        if probabilityValue > 0.75:
            class_name = classes[class_index]
            cv2.putText(imgOriginal, f"{class_index}: {class_name}", (120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(imgOriginal, f"{round(probabilityValue * 100, 2)}%", (180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display the original image with predictions
        cv2.imshow("Traffic Sign Classification", imgOriginal)
        

        # Press 'q' to quit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()  # Ensure all windows are closed after exiting


def preprocess_image_for_camera(img):
    """Preprocess the image for real-time classification."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# Add Traffic Sign Classification button
traffic_sign_classification_btn = Button(top, text="Phân loại biển báo giao thông", command=run_traffic_sign_classification, padx=10, pady=5)
traffic_sign_classification_btn.configure(background='#27ae60', foreground='white', font=('arial', 10, 'bold'))
traffic_sign_classification_btn.pack(side=BOTTOM, pady=10)
# Button to upload image
upload = Button(top, text="Tải ảnh", command=upload_image, padx=10, pady=5)
upload.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=10)

# Button to show results
results_btn = Button(top, text="Xuất kết quả", command=show_results, padx=10, pady=5)
results_btn.configure(background='#f39c12', foreground='white', font=('arial', 10, 'bold'))
results_btn.pack(side=BOTTOM, pady=10)

# Button for settings
settings_btn = Button(top, text="Cài đặt", command=open_settings, padx=10, pady=5)
settings_btn.configure(background='#8e44ad', foreground='white', font=('arial', 10, 'bold'))
settings_btn.pack(side=BOTTOM, pady=10)

# Button to exit
exit_btn = Button(top, text="Thoát", command=exit_program, padx=10, pady=5)
exit_btn.configure(background='#e74c3c', foreground='white', font=('arial', 10, 'bold'))
exit_btn.pack(side=BOTTOM, pady=10)

# Display area for image and results
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="Nhận dạng biển báo giao thông", pady=10, font=('arial', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#364156')

heading.pack()
top.mainloop()
