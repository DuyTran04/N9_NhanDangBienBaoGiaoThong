import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
import os
from PIL import Image, ImageTk
from subprocess import Popen


# Initialize the main GUI window
root = tk.Tk()
root.title("Giao diện chính - Nhận dạng biển báo giao thông")
root.geometry('800x600')
root.configure(background='#f2f2f2')

# Heading for the interface
heading = Label(root, text="Chương trình nhận dạng biển báo giao thông", font=('arial', 24, 'bold'))
heading.configure(background='#f2f2f2', foreground='#364156')
heading.pack(pady=20)

# Define functions for each button

def open_image():
    """This function allows the user to upload an image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img_display = ImageTk.PhotoImage(img)

            # Display uploaded image in the interface
            img_label = Label(root, image=img_display)
            img_label.image = img_display
            img_label.pack(pady=10)
            messagebox.showinfo("Thông báo", "Ảnh đã được tải lên thành công.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")

def run_classification():
    """This function runs the recognition program."""
    try:
        # Giả sử file nhận dạng có tên là "recognition.py"
        Popen(['python', 'recognition.py'])
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể chạy chương trình nhận dạng: {str(e)}")

def show_results():
    """This function shows the result of the classification."""
    # Giả sử kết quả được lưu vào file hoặc hiển thị trực tiếp
    messagebox.showinfo("Kết quả", "Kết quả nhận dạng biển báo sẽ được hiển thị tại đây.")

def open_settings():
    """This function opens the settings window."""
    settings_window = tk.Toplevel(root)
    settings_window.title("Cài đặt")
    settings_window.geometry('400x300')
    settings_label = Label(settings_window, text="Cài đặt hệ thống", font=('arial', 16))
    settings_label.pack(pady=20)

    # Bạn có thể thêm các cài đặt tùy chỉnh ở đây (ví dụ như điều chỉnh độ sáng, độ phân giải hình ảnh)
    messagebox.showinfo("Thông báo", "Cài đặt sẽ được cấu hình tại đây.")

def exit_program():
    """This function exits the program."""
    if messagebox.askyesno("Thoát", "Bạn có chắc chắn muốn thoát?"):
        root.quit()


# Buttons for the main functions
btn_upload = Button(root, text="Tải ảnh", command=open_image, padx=10, pady=5, font=('arial', 12, 'bold'))
btn_upload.configure(background='#4CAF50', foreground='white')
btn_upload.pack(pady=10)

btn_classify = Button(root, text="Nhận dạng", command=run_classification, padx=10, pady=5, font=('arial', 12, 'bold'))
btn_classify.configure(background='#008CBA', foreground='white')
btn_classify.pack(pady=10)

btn_results = Button(root, text="Xuất kết quả", command=show_results, padx=10, pady=5, font=('arial', 12, 'bold'))
btn_results.configure(background='#f39c12', foreground='white')
btn_results.pack(pady=10)

btn_settings = Button(root, text="Cài đặt", command=open_settings, padx=10, pady=5, font=('arial', 12, 'bold'))
btn_settings.configure(background='#E74C3C', foreground='white')
btn_settings.pack(pady=10)

btn_exit = Button(root, text="Thoát", command=exit_program, padx=10, pady=5, font=('arial', 12, 'bold'))
btn_exit.configure(background='#555555', foreground='white')
btn_exit.pack(pady=10)

# Run the main loop for the GUI
root.mainloop()
