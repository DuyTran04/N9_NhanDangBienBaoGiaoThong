import numpy as np
import cv2
import pickle
 
#############################################
 
frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################
 
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
pickle_in=open("model_trained.h5","rb")  ## rb = READ BYTE
model=pickle.load(pickle_in)

# FUNCTION DEFINITIONS

def grayscale(img):
    """Convert image to grayscale."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    """Apply histogram equalization to standardize lighting."""
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    """Preprocess the input image (grayscale, equalize, normalize)."""
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

def getClassName(classNo):
    """Return the class name corresponding to a class number."""
    classNames = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
        'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
        'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
        'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classNames[classNo]

# MAIN LOOP
while True:
    # READ IMAGE FROM CAMERA
    success, imgOriginal = cap.read()
    
    # PREPROCESS IMAGE
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))  # Resize the image to match model input size
    img = preprocessing(img)  # Apply preprocessing (grayscale, equalize, normalize)
    img = img.reshape(1, 32, 32, 1)  # Reshape to the model's input shape (batch, height, width, depth)

    # PREDICT CLASS USING THE MODEL
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)  # Get the predicted class
    probabilityValue = np.max(predictions)  # Get the probability of the prediction

    # DISPLAY PREDICTED CLASS AND PROBABILITY IF ABOVE THRESHOLD
    if probabilityValue > threshold:
        className = getClassName(classIndex)
        cv2.putText(imgOriginal, f"{classIndex} {className}", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"{round(probabilityValue * 100, 2)}%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # DISPLAY THE ORIGINAL IMAGE WITH PREDICTIONS
    cv2.imshow("Result", imgOriginal)

    # BREAK LOOP ON 'q' KEY PRESS
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEAN UP
cap.release()
cv2.destroyAllWindows()
