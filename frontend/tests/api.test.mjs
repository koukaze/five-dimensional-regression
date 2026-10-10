import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { apiRequest, errorMessage, formatMetric } from '../src/lib/api.mjs';

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

test('formats API validation errors as plain text', () => {
    assert.equal(errorMessage([{ loc: ['body', 'X', 1], msg: 'Input should be a finite number' }]), 'X.1: Input should be a finite number');
    assert.equal(errorMessage('No model trained.'), 'No model trained.');
    assert.equal(errorMessage({ unexpected: true }, 'Request failed'), 'Request failed');
});

test('uses the same-origin API and forwards successful JSON', async () => {
    globalThis.fetch = async (url, options) => {
        assert.equal(url, '/api/predict');
        assert.equal(options.cache, 'no-store');
        assert.equal(options.method, 'POST');
        return Response.json({ prediction: 1.25 });
    };
    assert.deepEqual(await apiRequest('/predict', { method: 'POST' }), { prediction: 1.25 });
});

test('rejects validation and non-JSON server failures with readable messages', async () => {
    globalThis.fetch = async () => Response.json({ detail: [{ loc: ['body', 'n_layers'], msg: 'Input should be greater than 0' }] }, { status: 422 });
    await assert.rejects(apiRequest('/train'), /n_layers: Input should be greater than 0/);
    globalThis.fetch = async () => new Response('Bad gateway', { status: 502 });
    await assert.rejects(apiRequest('/status'), /did not return a valid response/);
});

test('handles an unavailable backend', async () => {
    globalThis.fetch = async () => { throw new TypeError('fetch failed'); };
    await assert.rejects(apiRequest('/status'), /Cannot reach the server/);
});

test('formats unavailable and zero-valued regression metrics', () => {
    assert.equal(formatMetric(null), '—');
    assert.equal(formatMetric(NaN), '—');
    assert.equal(formatMetric(0), '0.0000');
    assert.equal(formatMetric(-12.5), '-12.500');
});
