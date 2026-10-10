from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fivedreg.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def dataset_bytes() -> bytes:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(100, 5)) * [2, 10, 0.1, 4, 3] + [100, -20, 4, 8, 0]
    y = X[:, 0] - X[:, 1] / 10
    buffer = BytesIO()
    np.savez(buffer, X=X, y=y)
    return buffer.getvalue()


def upload(client: TestClient) -> None:
    response = client.post('/upload', files={'file': ('dataset.npz', dataset_bytes(), 'application/octet-stream')})
    assert response.status_code == 200


def test_initial_workflow_and_validation(client: TestClient) -> None:
    assert client.get('/health').json() == {'status': 'ok'}
    assert client.get('/status').json() == {
        'dataset_loaded': False, 'model_trained': False, 'n_samples': 0,
        'split_sizes': {}, 'training': False,
    }
    assert client.post('/train', json={}).status_code == 400
    assert client.post('/predict', json={'X': [1, 2, 3, 4, 5]}).status_code == 400
    assert client.post('/predict', json={'X': [1, 2]}).status_code == 422
    assert client.post('/train', json={'n_nodes': 0}).status_code == 422
    assert client.post('/train', json={'n_layers': True}).status_code == 422
    assert client.post('/upload', files={'file': ('dataset.pkl', b'pickle')}).status_code == 400
    assert client.post('/upload', files={'file': ('dataset.npz', b'invalid')}).status_code == 400


def test_training_raw_prediction_metrics_and_upload_reset(client: TestClient) -> None:
    upload(client)
    response = client.post('/train', json={'n_layers': 1, 'n_nodes': 8, 'max_epochs': 2, 'seed': 21})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['split_sizes'] == {'train': 60, 'validation': 20, 'test': 20}
    assert set(result['metrics']) == {'train', 'validation', 'test'}
    assert len(result['history']['val_loss']) == 2
    state = client.app.state.workflow
    np.testing.assert_allclose(result['metrics']['test']['mse'], state.model.metrics(state.data.X_test, state.data.y_test)['mse'])
    raw = np.array([[100, -20, 4, 8, 0]], dtype=float)
    expected = state.model.predict(state.data.scaler.transform(raw))[0]
    predicted = client.post('/predict', json={'X': raw[0].tolist()})
    np.testing.assert_allclose(predicted.json()['prediction'], expected, rtol=1e-6)
    assert client.get('/status').json()['model_trained'] is True
    upload(client)
    assert client.get('/status').json()['model_trained'] is False
    assert client.post('/predict', json={'X': raw[0].tolist()}).status_code == 400


def test_busy_workflow_and_failed_upload_preserve_data(client: TestClient) -> None:
    upload(client)
    state = client.app.state.workflow
    previous = state.data
    with state.lock:
        assert client.post('/train', json={}).status_code == 409
        assert client.post('/upload', files={'file': ('dataset.npz', dataset_bytes())}).status_code == 409
    assert client.post('/upload', files={'file': ('bad.npz', b'invalid')}).status_code == 400
    assert state.data is previous
    assert not state.lock.locked()


def test_upload_size_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('fivedreg.main.MAX_UPLOAD_BYTES', 10)
    response = client.post('/upload', files={'file': ('dataset.npz', b'x' * 11)})
    assert response.status_code == 413
    assert client.get('/status').json()['dataset_loaded'] is False
    assert not client.app.state.workflow.lock.locked()
