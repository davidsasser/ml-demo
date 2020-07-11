from django.shortcuts import render
from joblib import load
import numpy as np
import os
import base64



def index(request):

    return render(request,'index.html')

def predict(request):
    if request.method == 'POST':
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../numbers_model.joblib')
        model = load(filename)
        x_val = [   
                    [ 0.,  0.,  3., 10., 15.,  8.,  0.,  0.],  
                    [ 0.,  0., 12., 14.,  8.,  1.,  0.,  0.],  
                    [ 0.,  1., 16.,  3.,  0.,  0.,  0.,  0.],  
                    [ 0.,  2., 16.,  9., 11., 16.,  3.,  0.], 
                    [ 0.,  4., 16., 14.,  9., 15.,  7.,  0.], 
                    [ 0.,  1.,  4.,  0.,  0., 15.,  3.,  0.],  
                    [ 0.,  0.,  0.,  3., 12.,  8.,  0.,  0.],  
                    [ 0.,  0.,  2., 10.,  8.,  0.,  0.,  0.]
                ]
        x = np.asarray(x_val)
        prediction = model.predict(x.reshape(1, -1))

        base64str = request.POST.get("imgBase64", None).split(',')[1]
        # print(base64str)
        imgdata = base64.b64decode(base64str)
        filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(imgdata)

    # if a GET (or any other method) we'll create a blank form
    else:
        print("uh oh")

    return render(request, 'index.html', {'prediction': prediction})