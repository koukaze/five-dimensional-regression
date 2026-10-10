export function errorMessage(detail, fallback = 'The request could not be completed.') {
    if (typeof detail === 'string' && detail) return detail;
    if (Array.isArray(detail)) {
        return detail.map((item) => {
            const field = (item.loc || []).filter((part) => part !== 'body').join('.');
            return `${field ? `${field}: ` : ''}${item.msg || 'Invalid value'}`;
        }).join('; ') || fallback;
    }
    return fallback;
}

export async function apiRequest(path, options = {}) {
    let response;
    try {
        response = await fetch(`/api${path}`, { cache: 'no-store', ...options });
    } catch {
        throw new Error('Cannot reach the server. Check that the application is running.');
    }
    let data;
    try {
        data = await response.json();
    } catch {
        throw new Error('The server did not return a valid response. Check the backend and try again.');
    }
    if (!response.ok) {
        throw new Error(errorMessage(data.detail, `Request failed (${response.status}).`));
    }
    return data;
}

export function formatMetric(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toPrecision(5) : '—';
}
