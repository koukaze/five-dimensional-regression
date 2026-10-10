'use client';

import Link from 'next/link';
import { useState } from 'react';
import SessionStatus from '../../components/session-status';
import { apiRequest } from '../../lib/api.mjs';
import { useStatus } from '../../lib/use-status';

export default function Upload() {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState('');
    const [failed, setFailed] = useState(false);
    const [loading, setLoading] = useState(false);
    const { status, error, refresh } = useStatus();

    async function uploadFile(dataset) {
        setLoading(true);
        setMessage('');
        setFailed(false);
        const formData = new FormData();
        formData.append('file', dataset);
        try {
            const data = await apiRequest('/upload', { method: 'POST', body: formData });
            setMessage(`${data.message} ${data.n_samples.toLocaleString()} samples are ready for training.`);
        } catch (error) {
            setMessage(error.message);
            setFailed(true);
        } finally {
            setLoading(false);
            await refresh();
        }
    }

    async function handleUpload(event) {
        event.preventDefault();
        await uploadFile(file);
    }

    async function handleExample() {
        setLoading(true);
        setFailed(false);
        setMessage('');
        try {
            const response = await fetch('/example-dataset.npz');
            if (!response.ok) throw new Error('The example dataset could not be loaded.');
            const blob = await response.blob();
            await uploadFile(new File([blob], 'example-dataset.npz'));
        } catch (error) {
            setMessage(error.message);
            setFailed(true);
            setLoading(false);
        }
    }

    return (
        <main id="main-content">
            <div className="intro compact"><p className="eyebrow">Step 01</p><h1>Upload a dataset</h1><p className="lead">Each sample has five numeric features and one continuous target.</p></div>
            <SessionStatus status={status} error={error} onRefresh={refresh} />
            <form className="panel" onSubmit={handleUpload}>
                <h2>NumPy archive</h2>
                <p>Choose a <code>.npz</code> file containing <code>X</code> with shape <code>(N, 5)</code> and <code>y</code> with shape <code>(N,)</code>. All values must be finite numbers.</p>
                <label htmlFor="dataset">Dataset file</label>
                <input id="dataset" type="file" accept=".npz" required disabled={loading} onChange={(event) => { setFile(event.target.files[0] || null); setMessage(''); }} />
                <p className="note">A successful upload replaces the current dataset and clears the trained model. Data is split into 60% training, 20% validation and the remaining samples for testing.</p>
                <div className="button-row"><button disabled={loading || !file || status?.training}>{loading ? 'Uploading…' : 'Upload dataset'}</button><Link href="/train">Continue to training →</Link></div>
                <div className="example-data">
                    <h2>Try the example</h2>
                    <p className="note">A synthetic dataset with 1,000 samples and a nonlinear target is included for a complete training and prediction run.</p>
                    <button type="button" onClick={handleExample} disabled={loading || status?.training}>Use example data</button>
                </div>
            </form>
            {message && <p className={`message ${failed ? 'error' : ''}`} role={failed ? 'alert' : 'status'}>{message}</p>}
        </main>
    );
}
