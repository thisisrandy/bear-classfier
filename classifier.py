# mostly stolen from
# https://github.com/simonw/cougar-or-not/blob/master/cougar.py

from fastai.basic_train import load_learner
from fastai.vision import open_image
import torch
from io import BytesIO
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import sys
import aiohttp
import asyncio

fastai.torch_core.defaults.device = torch.device("cpu")
learner = load_learner(".", "bear-classifier.pkl")

app = Starlette()


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await data["file"].read()
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _, _, losses = learner.predict(img)
    return JSONResponse(
        {
            "predictions": sorted(
                zip(learner.data.classes, map(float, losses)),
                key=lambda p: p[1],
                reverse=True,
            )
        }
    )


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post", enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
        """
    )


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
