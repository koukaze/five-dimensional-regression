'''Expose a local upload, training, and prediction workflow through FastAPI.'''

from dataclasses import dataclass, field
from io import BytesIO
from threading import Lock
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from fivedreg.data import MAX_UPLOAD_BYTES, DatasetSplit, load_dataset
from fivedreg.model import MLP


class TrainRequest(BaseModel):
    '''Specify bounded model settings for a fixed-epoch training run.'''

    model_config = ConfigDict(extra='forbid')
    n_layers: int = Field(default=2, ge=1, le=5, strict=True)
    n_nodes: int = Field(default=64, ge=1, le=256, strict=True)
    lr: float = Field(default=0.001, gt=0, le=0.1, allow_inf_nan=False)
    max_epochs: int = Field(default=100, ge=1, le=500, strict=True)
    seed: int = Field(default=42, ge=0, le=2**31 - 1, strict=True)


class PredictRequest(BaseModel):
    '''Supply five raw feature values in training-column order.'''

    model_config = ConfigDict(extra='forbid')
    X: list[Annotated[float, Field(strict=True, allow_inf_nan=False)]] = Field(min_length=5, max_length=5)


@dataclass
class WorkflowState:
    '''Keep one dataset and fitted model for a local application process.'''

    data: DatasetSplit | None = None
    model: MLP | None = None
    training: bool = False
    lock: Lock = field(default_factory=Lock)


def create_app() -> FastAPI:
    '''Create an API instance with independent in-memory workflow state.

    Returns
        FastAPI: Application intended for one local user and one worker.
    '''
    app = FastAPI(title='Five-Dimensional Regression', version='0.1.0')
    app.state.workflow = WorkflowState()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
        allow_methods=['GET', 'POST'],
        allow_headers=['Content-Type'],
    )

    @app.get('/health')
    def health_check() -> dict[str, str]:
        '''Return service health without changing workflow state.'''
        return {'status': 'ok'}

    @app.get('/status')
    def status() -> dict:
        '''Report dataset availability and the current training state.'''
        state = app.state.workflow
        data = state.data
        return {
            'dataset_loaded': data is not None,
            'model_trained': state.model is not None,
            'n_samples': sum(data.split_sizes.values()) if data is not None else 0,
            'split_sizes': data.split_sizes if data is not None else {},
            'training': state.training,
        }

    @app.post('/upload')
    async def upload_dataset(file: UploadFile = File(...)) -> dict:
        '''Validate an NPZ upload and begin a new dataset workflow.'''
        state = app.state.workflow
        if not state.lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail='Another operation is in progress.')
        try:
            if not file.filename or not file.filename.lower().endswith('.npz'):
                raise HTTPException(status_code=400, detail='Upload an NPZ file containing X and y arrays.')
            contents = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail='Upload exceeds the 20 MiB limit.')
            try:
                data = load_dataset(BytesIO(contents), seed=42)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            state.data = data
            state.model = None
            return {
                'message': 'Dataset ready for training.',
                'n_samples': sum(data.split_sizes.values()),
                'split_sizes': data.split_sizes,
            }
        finally:
            await file.close()
            state.lock.release()

    @app.post('/train')
    def train_model(params: TrainRequest) -> dict:
        '''Train on the active dataset and evaluate the final model weights.'''
        state = app.state.workflow
        if not state.lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail='Another operation is in progress.')
        try:
            if state.data is None:
                raise HTTPException(status_code=400, detail='Upload a dataset before training.')
            state.training = True
            data = state.data
            configuration = params.model_dump()
            model = MLP(**configuration)
            try:
                model.fit(data.X_train, data.y_train, validation_data=(data.X_val, data.y_val))
                metrics = {
                    'train': model.metrics(data.X_train, data.y_train),
                    'validation': model.metrics(data.X_val, data.y_val),
                    'test': model.metrics(data.X_test, data.y_test),
                }
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            state.model = model
            return {
                'message': 'Model trained.',
                'metrics': metrics,
                'history': {name: [float(value) for value in values]
                            for name, values in model.history.history.items()},
                'configuration': configuration,
                'split_sizes': data.split_sizes,
            }
        finally:
            state.training = False
            state.lock.release()

    @app.post('/predict')
    def predict(params: PredictRequest) -> dict[str, float]:
        '''Standardise one raw input with training statistics and predict.'''
        state = app.state.workflow
        if not state.lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail='Another operation is in progress.')
        try:
            if state.model is None or state.data is None:
                raise HTTPException(status_code=400, detail='Train a model before prediction.')
            raw = np.asarray(params.X, dtype=np.float64).reshape(1, 5)
            try:
                transformed = state.data.scaler.transform(raw)
                prediction = state.model.predict(transformed)[0]
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {'prediction': float(prediction)}
        finally:
            state.lock.release()

    return app


app = create_app()
