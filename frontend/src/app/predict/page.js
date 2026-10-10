'use client';

import Link from 'next/link';
import { useState } from 'react';
import SessionStatus from '../../components/session-status';
import { apiRequest, formatMetric } from '../../lib/api.mjs';
import { useStatus } from '../../lib/use-status';

export default function Predict() {
    const [prediction, setPrediction] = useState(null);
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);
    const { status, error, refresh } = useStatus();

    async function handlePredict(event) {
        event.preventDefault();
        const form = new FormData(event.currentTarget);
        const features = Array.from({ length: 5 }, (_, index) => Number(form.get(`x${index + 1}`)));
        setLoading(true);
        setMessage('');
        setPrediction(null);
        try {
            const data = await apiRequest('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ X: features }) });
            if (!Number.isFinite(data.prediction)) throw new Error('The server returned a non-finite prediction.');
            setPrediction(data.prediction);
        } catch (error) {
            setMessage(error.message);
            await refresh();
        } finally {
            setLoading(false);
        }
    }

    return (
        <main id="main-content">
            <div className="intro compact"><p className="eyebrow">Step 03</p><h1>Make a prediction</h1><p className="lead">Enter one feature vector in the column order used by your dataset.</p></div>
            <SessionStatus status={status} error={error} onRefresh={refresh} />
            <form className="panel" onSubmit={handlePredict}>
                <h2>Input features</h2>
                <p>Use the original feature units. The model applies its training-set scaling automatically.</p>
                <fieldset disabled={loading || status?.training}>
                    <legend className="sr-only">Five feature values</legend>
                    <div className="features-grid">
                        {Array.from({ length: 5 }, (_, index) => <div className="field" key={index}><label htmlFor={`x${index + 1}`}>X{index + 1}</label><input id={`x${index + 1}`} name={`x${index + 1}`} type="number" step="any" placeholder="0.0" required /></div>)}
                    </div>
                </fieldset>
                {!status?.model_trained && <p className="note"><Link href="/train">Train a model</Link> before making a prediction.</p>}
                <div className="button-row"><button disabled={loading || !status?.model_trained || status?.training}>{loading ? 'Predicting…' : 'Predict target'}</button></div>
                <p className="note">Predictions far outside the training feature range may be unreliable.</p>
            </form>
            {prediction !== null && <section className="result-box" aria-live="polite"><p className="eyebrow">Predicted target</p><p className="prediction-value">{formatMetric(prediction)}</p></section>}
            {message && <p className="message error" role="alert">{message}</p>}
        </main>
    );
}
