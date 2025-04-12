IMG_SIZE = 32

def preprocess_image(img):
    # Resize, normalize and expand dims
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # img=cv2.imread(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img