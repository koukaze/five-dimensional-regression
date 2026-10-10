'use client';

import Link from 'next/link';
import SessionStatus from '../components/session-status';
import { useStatus } from '../lib/use-status';

export default function Home() {
    const { status, error, refresh } = useStatus();
    return (
        <main id="main-content">
            <div className="intro">
                <p className="eyebrow">Multilayer perceptron · Regression</p>
                <h1>From five inputs<br />to a prediction.</h1>
                <p className="lead">Fit a nonlinear relationship, inspect its performance on held-out samples, and predict a continuous target from five feature values.</p>
            </div>
            <SessionStatus status={status} error={error} onRefresh={refresh} />
            <div className="workflow">
                <Link className="workflow-card" href="/upload">
                    <span className="step">01</span><h2>Upload a dataset</h2>
                    <p>Load a NumPy archive containing five features per sample and one target value.</p>
                    <span className="action">Choose data <span aria-hidden="true">→</span></span>
                </Link>
                <Link className="workflow-card" href="/train">
                    <span className="step">02</span><h2>Train the model</h2>
                    <p>Set the network size and learning rate, then inspect train, validation and test metrics.</p>
                    <span className="action">Configure training <span aria-hidden="true">→</span></span>
                </Link>
                <Link className="workflow-card" href="/predict">
                    <span className="step">03</span><h2>Make a prediction</h2>
                    <p>Enter a feature vector in the same units and column order as the training data.</p>
                    <span className="action">Enter features <span aria-hidden="true">→</span></span>
                </Link>
            </div>
            <p className="note">The dataset and model are kept in the current server session. Restarting the backend clears the session.</p>
        </main>
    );
}
