import { formatMetric } from '../lib/api.mjs';

export default function Metrics({ metrics }) {
    return (
        <section className="panel" aria-labelledby="metrics-title">
            <h2 id="metrics-title">Model evaluation</h2>
            <p>MAE uses the target units; MSE uses squared target units. R² is unavailable when the target variance is zero.</p>
            <div className="table-scroll">
                <table>
                    <thead><tr><th scope="col">Split</th><th scope="col">MSE</th><th scope="col">MAE</th><th scope="col">R²</th></tr></thead>
                    <tbody>
                        {['train', 'validation', 'test'].map((split) => (
                            <tr key={split}>
                                <th scope="row">{split === 'validation' ? 'Validation' : split === 'train' ? 'Train' : 'Test'}</th>
                                <td>{formatMetric(metrics[split]?.mse)}</td>
                                <td>{formatMetric(metrics[split]?.mae)}</td>
                                <td>{formatMetric(metrics[split]?.r2)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}
