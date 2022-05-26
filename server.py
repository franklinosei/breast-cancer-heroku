from typing import List
from fastapi import FastAPI, UploadFile
import PIL.Image as Image
import io
from tensorflow import keras
from fastapi.middleware.cors import CORSMiddleware


model_path = "model/model/"

MODEL = keras.models.load_model(model_path)


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Message": "Server is running"}



@app.post("/predict/")
async def create_upload_files(file: UploadFile):
    try:
        # get image from request
        image = await file.read()

        # read image and preprocess it
        pred_image = np.array(Image.open(io.BytesIO(image)).resize((50, 50)))
        images_list = []
        images_list.append(np.array(pred_image))
        img = np.asarray(images_list)

        # run prediction on processed image
        prediction = MODEL.predict(img)

        message = ''

        if prediction[0][0] > 0.5:
            message = 'Positive'
        elif(prediction[0][0] < 0.5):
            message = 'Negative'

        # returning prediction
        return {"prediction": float(prediction), "message": message}

    except Exception as e:
        print(e)
        return {"error": str(e)}


