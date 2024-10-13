import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
import numpy as np
import cv2
import pickle
from PIL import Image, ImageTk, ImageFilter
from tkinter import BOTTOM

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

# Global variables to store the blur level and image dimensions
blur_level = 0  
image_width = 32  
image_height = 32  

# Initialize the GUI
top = tk.Tk()
top.geometry('900x650')
top.title('Nhận dạng biển báo giao thông')
top.configure(background='#E8EAF6')

# Heading label for the title of the application
heading = Label(top, text="Nhận dạng biển báo giao thông", pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#E8EAF6', foreground='#5E35B1')
heading.pack()

# Frame for displaying the uploaded image
sign_image = Label(top, background='#E8EAF6', borderwidth=2, relief="groove", width=60, height=15)
sign_image.pack(pady=20)

# Label for displaying classification result
label = Label(top, background='#E8EAF6', font=('arial', 15, 'bold'))
label.pack()

# Global variables to store classification result and uploaded image
result_text = ""
uploaded_image = None  

def preprocess_image(img):
    """
    Preprocess the uploaded image for classification.
    - Resize the image
    - Convert to grayscale
    - Normalize pixel values
    """
    img = img.resize((image_width, image_height))  # Resize image
    img = np.array(img)  # Convert to NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, image_width, image_height, 1)  # Reshape for model input
    return img

def classify(file_path):
    """
    Classify the uploaded image using the trained model.
    Display the prediction and accuracy on the screen.
    """
    global result_text, blur_level
    try:
        img = Image.open(file_path)  # Open the image file

        # Apply blur effect if specified
        if blur_level > 0:
            img = img.filter(ImageFilter.GaussianBlur(blur_level))

        img = preprocess_image(img)  # Preprocess the image
        pred = model.predict(img)  # Make prediction
        class_index = np.argmax(pred)  # Get the class index
        sign = classes[class_index]  # Get the class name
        accuracy = np.max(pred) * 100  # Calculate the accuracy
        result_text = f'{sign}\nĐộ chính xác: {accuracy:.2f}%'
        label.configure(foreground='#5E35B1', text=result_text)  # Display result
    except Exception as e:
        print(f"Error during prediction: {e}")
        label.configure(foreground='#5E35B1', text="Prediction Error")

def show_classify_button(file_path):
    """
    Display the 'Nhận dạng' button after the image is uploaded.
    """
    classify_b = Button(top, text="Nhận dạng", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#5E35B1', foreground='white', font=('arial', 10, 'bold'), width=15)
    classify_b.place(relx=0.79, rely=0.46)

def update_blurred_image():
    """
    Update the displayed image with the current blur effect applied.
    """
    global uploaded_image
    if uploaded_image is not None:
        resized_img = uploaded_image.copy()
        resized_img = resized_img.resize((image_width, image_height))
        if blur_level > 0:
            resized_img = resized_img.filter(ImageFilter.GaussianBlur(blur_level))
        im = ImageTk.PhotoImage(resized_img)
        sign_image.configure(image=im)
        sign_image.image = im

def upload_image():
    """
    Upload an image file and display it on the screen.
    """
    global uploaded_image
    try:
        file_path = filedialog.askopenfilename()
        uploaded_image = Image.open(file_path)
        update_blurred_image()  # Display the uploaded image
        label.configure(text='')
        show_classify_button(file_path)  # Show the 'Nhận dạng' button
    except Exception as e:
        print(e)

def show_results():
    """
    Display and allow saving the classification results to a text file.
    """
    global result_text
    if result_text:
        messagebox.showinfo("Kết quả", result_text)
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if save_path:
            with open(save_path, "w") as f:
                f.write(result_text)
                messagebox.showinfo("Thông báo", f"Kết quả đã được lưu vào {save_path}")
    else:
        messagebox.showinfo("Kết quả", "Chưa có kết quả nào để xuất.")

def open_settings():
    """
    Open the settings window to adjust image processing options such as blur level and image size.
    """
    settings_window = tk.Toplevel(top)
    settings_window.title("Cài đặt")
    settings_window.geometry('400x400')
    settings_window.configure(background='#E8EAF6')
    Label(settings_window, text="Cài đặt hệ thống", font=('arial', 16)).pack(pady=20)
    
    global blur_slider, width_slider, height_slider

    # Blur level slider
    blur_slider = tk.Scale(settings_window, from_=0, to=10, orient="horizontal", label="Độ mờ (0-10)", length=300)
    blur_slider.set(blur_level)
    blur_slider.pack(pady=10)

    # Image width slider
    width_slider = tk.Scale(settings_window, from_=32, to=128, orient="horizontal", label="Chiều rộng ảnh", length=300)
    width_slider.set(image_width)
    width_slider.pack(pady=10)

    # Image height slider
    height_slider = tk.Scale(settings_window, from_=32, to=128, orient="horizontal", label="Chiều cao ảnh", length=300)
    height_slider.set(image_height)
    height_slider.pack(pady=10)

    save_button = Button(settings_window, text="Lưu cài đặt", command=save_settings)
    save_button.pack(pady=20)

def save_settings():
    """
    Save the user settings (blur level and image size) to a file.
    """
    global blur_level, image_width, image_height
    blur_level = blur_slider.get()
    image_width = width_slider.get()
    image_height = height_slider.get()
    with open("settings.txt", "w") as f:
        f.write(f"{blur_level}\n")
        f.write(f"{image_width}\n")
        f.write(f"{image_height}\n")
    update_blurred_image()  # Apply the new settings to the displayed image
    messagebox.showinfo("Cài đặt", "Cài đặt đã được lưu.")

def exit_program():
    """
    Exit the application when the user confirms.
    """
    if messagebox.askyesno("Thoát", "Bạn có chắc chắn muốn thoát?"):
        top.quit()

def run_traffic_sign_classification():
    """
    Open the camera, stream video, and classify traffic signs in real-time.
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set camera width
    cap.set(4, 480)  # Set camera height
    cap.set(10, 180)  # Set camera brightness

    while cap.isOpened():
        success, imgOriginal = cap.read()
        if not success:
            break

        img = np.asarray(imgOriginal)
        img = cv2.resize(img, (32, 32))  # Resize for model input
        img = preprocess_image_for_camera(img)  # Preprocess image
        img = img.reshape(1, 32, 32, 1)

        # Perform prediction
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        probabilityValue = np.max(predictions)

        # If confidence level is high, display the result on the video feed
        if probabilityValue > 0.75:
            class_name = classes[class_index]
            cv2.putText(imgOriginal, f"{class_index}: {class_name}", (120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(imgOriginal, f"{round(probabilityValue * 100, 2)}%", (180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Traffic Sign Classification", imgOriginal)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def preprocess_image_for_camera(img):
    """
    Preprocess the real-time camera frame (grayscale and normalization).
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# Buttons area
button_frame = tk.Frame(top, bg='#E8EAF6')
button_frame.pack(side=BOTTOM, pady=20)

# Button for uploading an image
upload = Button(button_frame, text="Tải ảnh", command=upload_image, padx=10, pady=5)
upload.configure(background='#81C784', foreground='black', font=('arial', 10, 'bold'), width=15)
upload.grid(row=0, column=0, padx=10)

# Button for running real-time camera classification
camera_btn = Button(button_frame, text="Sử dụng camera", command=run_traffic_sign_classification, padx=10, pady=5)
camera_btn.configure(background='#81C784', foreground='black', font=('arial', 10, 'bold'), width=15)
camera_btn.grid(row=0, column=2, padx=10)

# Button for opening settings
settings_btn = Button(button_frame, text="Cài đặt", command=open_settings, padx=10, pady=5)
settings_btn.configure(background='#81C784', foreground='black', font=('arial', 10, 'bold'), width=15)
settings_btn.grid(row=0, column=3, padx=10)

# Button to show and save results
results_btn = Button(button_frame, text="Xuất kết quả", command=show_results, padx=10, pady=5)
results_btn.configure(background='#81C784', foreground='black', font=('arial', 10, 'bold'), width=15)
results_btn.grid(row=0, column=4, padx=10)

# Button to exit the application
exit_btn = Button(button_frame, text="Thoát", command=exit_program, padx=10, pady=5)
exit_btn.configure(background='#81C784', foreground='black', font=('arial', 10, 'bold'), width=15)
exit_btn.grid(row=0, column=5, padx=10)

top.mainloop()
