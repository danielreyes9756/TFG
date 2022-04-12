from flask import Flask, render_template
from flask import request
import os
import numpy as np
from tensorflow import keras
from skimage import io

app = Flask(__name__)

## -------------------- Load Models -------------------


def model_predict(img_path):
    
    if("gs" in MODEL_CNN_PATH ):
        img = keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=(128,128))
    else:
        img = keras.preprocessing.image.load_img(img_path, target_size=(128,128))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = clone(model).predict(x)
   
    return preds

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename 
        print('The filename that has been uploaded =',filename)
        # know the extension of filename
        # all only .jpg, .png, .jpeg, PNG
        ext = filename.split('.')[-1]
        print('The extension of the filename =',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            # saving the image
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File saved sucessfully')
            print(path_save)
            preds = model_predict(path_save)
            print(preds)
            results = [(classes[i], preds[0][i]) for i in range(len(classes))]
            
            return render_template('upload.html', fileupload=True, extension=False, data=results, image_filename=filename, height=getheight(path_save))
        else:
            print('Use only the extension with .jpg, .png, .jpeg')
            return render_template('upload.html',fileupload=False, extension=True)
            
    else:
        return render_template('upload.html', fileupload=False, extension=False)

def getheight(path):
    img = io.imread(path)
    h,w,_ =img.shape 
    aspect = h/w
    given_width = 300
    height = given_width*aspect
    return height

def clone(model):
    model_cp = keras.models.clone_model(model) # Necesario bug de TNSF
    model_cp.make_predict_function()
    return model_cp 

@app.route('/about/')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    
    BASE_PATH = os.getcwd()
    UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
    MODEL_PATH = os.path.join(BASE_PATH,'static/models/')
    MODEL_CNN_PATH = os.path.join(MODEL_PATH,'modelCNN_gs.h5')
    model = keras.models.load_model(MODEL_CNN_PATH)
    model.make_predict_function()

    classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    app.run(debug=False) 