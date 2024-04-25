from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib


app = FastAPI(
    title="Deploy trabajo practico",
    version="0.0.1"
)
print()
# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
model = joblib.load("model/logistic_regression_model_v01.pkl")

@app.post("/api/v1/trabajo-practico", tags=["trabajo-practico"])
async def predict(
    radius: float,
    texture: float,
    perimeter: float,
    area: float,
    smoothness: float,
    compactness: float,
    symmetry: float,
    fractal_dimension: float
):
    dictionary = {
        'radius': radius,
        'texture': texture,
        'perimeter': perimeter,
        'area': area,
        'smoothness': smoothness,
        'compactness': compactness,
        'symmetry': symmetry,
        'fractal_dimension': fractal_dimension
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=prediction[0]
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )