export default function SessionStatus({ status, error, onRefresh }) {
    return (
        <section className="session-status" aria-label="Current session">
            <div role="status">
                {error ? <p className="error">{error}</p> : status ? (
                    <p>
                        <span className={`status-dot ${status.dataset_loaded ? 'ready' : ''}`} />
                        {status.dataset_loaded ? `${status.n_samples.toLocaleString()} samples loaded` : 'No dataset loaded'}
                        <span className="separator">/</span>
                        {status.training ? 'Training in progress' : status.model_trained ? 'Model ready' : 'No trained model'}
                    </p>
                ) : <p>Checking session…</p>}
            </div>
            <button type="button" className="quiet" onClick={onRefresh}>Refresh status</button>
        </section>
    );
}
