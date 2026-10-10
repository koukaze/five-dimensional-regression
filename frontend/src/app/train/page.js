'use client';

import Link from 'next/link';
import { useState } from 'react';
import SessionStatus from '../../components/session-status';
import Metrics from '../../components/metrics';
import { apiRequest } from '../../lib/api.mjs';
import { useStatus } from '../../lib/use-status';

const fields = [
    { name: 'n_layers', label: 'Hidden layers', min: 1, max: 5, step: 1, defaultValue: 2 },
    { name: 'n_nodes', label: 'Nodes per layer', min: 1, max: 256, step: 1, defaultValue: 64 },
    { name: 'lr', label: 'Learning rate', min: 0.00000001, max: 0.1, step: 'any', defaultValue: 0.001 },
    { name: 'max_epochs', label: 'Epochs', min: 1, max: 500, step: 1, defaultValue: 100 },
    { name: 'seed', label: 'Model seed', min: 0, max: 2147483647, step: 1, defaultValue: 42 },
];

export default function Train() {
    const [message, setMessage] = useState('');
    const [failed, setFailed] = useState(false);
    const [loading, setLoading] = useState(false);
    const [metrics, setMetrics] = useState(null);
    const { status, error, refresh } = useStatus();

    async function handleTrain(event) {
        event.preventDefault();
        const form = new FormData(event.currentTarget);
        const parameters = Object.fromEntries(fields.map(({ name }) => [name, Number(form.get(name))]));
        setLoading(true);
        setFailed(false);
        setMessage('Training is running. Keep this page open until the result appears.');
        setMetrics(null);
        try {
            const data = await apiRequest('/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(parameters) });
            setMessage(data.message);
            setMetrics(data.metrics);
        } catch (error) {
            setMessage(error.message);
            setFailed(true);
        } finally {
            setLoading(false);
            await refresh();
        }
    }

    return (
        <main id="main-content">
            <div className="intro compact"><p className="eyebrow">Step 02</p><h1>Train the model</h1><p className="lead">A fully connected network with ReLU hidden layers, a linear output and Adam optimisation.</p></div>
            <SessionStatus status={status} error={error} onRefresh={refresh} />
            <form className="panel" onSubmit={handleTrain}>
                <h2>Training configuration</h2>
                <p>Feature scaling is fitted on the training split and applied to every prediction. Training runs for the requested number of epochs.</p>
                <fieldset disabled={loading || status?.training}>
                    <legend className="sr-only">Network and optimiser settings</legend>
                    <div className="settings-grid">
                        {fields.map(({ name, label, ...properties }) => <div className="field" key={name}><label htmlFor={name}>{label}</label><input id={name} name={name} type="number" required {...properties} /></div>)}
                    </div>
                </fieldset>
                {!status?.dataset_loaded && <p className="note"><Link href="/upload">Upload a dataset</Link> before starting training.</p>}
                <div className="button-row"><button disabled={loading || !status?.dataset_loaded || status?.training}>{loading ? 'Training…' : 'Train model'}</button><Link href="/predict">Continue to prediction →</Link></div>
                <p className="note">Use validation metrics to compare settings. Repeatedly choosing settings from test results makes the test estimate less independent.</p>
            </form>
            {message && <p className={`message ${failed ? 'error' : ''}`} role={failed ? 'alert' : 'status'}>{message}</p>}
            {metrics && <Metrics metrics={metrics} />}
        </main>
    );
}
