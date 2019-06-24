from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates
from fastai.vision import *
import uvicorn


app = Starlette()
app.debug = True

templates = Jinja2Templates(directory='templates')

path = Path("./")
classes = ["mamals", "reptiles"]

#data = ImageDataBunch.single_from_classes(path,classes,size=224).normalize(imagenet_stats)
async def init():
    return load_learner(path,"export.pkl")


@app.route('/')
def index(request):
    return templates.TemplateResponse("index.html",{'request':request})

@app.route('/submit',methods = ["POST"])
async def submit(request):
    learn = await init()
    data = await request.form()
    bytes = await data["file"].read()
    return pred_img_from_bytes(learn,bytes)


def pred_img_from_bytes(learn,bytes):

    img = open_image(BytesIO(bytes))
    losses = learn.predict(img)
    return  PlainTextResponse(str(losses[0]))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
