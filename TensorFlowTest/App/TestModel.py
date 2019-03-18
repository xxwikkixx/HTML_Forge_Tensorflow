import cv2
import tensorflow as tf

CATEGORIES = ['Header', 'Title', 'Plain_Image_Gallery',
              'Paragraph', 'IMG_Top_Text_Bottom', 'IMG_Right_Text_Left',
              'IMG_Left_Text_Right', 'ImageFlipWithPreview', 'Image_Flip', 'Footer']

def prepareImage(path):
    IMG_SIZE = 90  # 50 in txt-based
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

model = tf.keras.models.load_model("64x3-CNN.model")
prediction1 = model.predict([prepareImage('header.png')])
print(prediction1)  # will be a list in a list.
print(CATEGORIES[int(prediction1[0][0])])