'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiRequest } from './api.mjs';

export function useStatus() {
    const [status, setStatus] = useState(null);
    const [error, setError] = useState('');
    const refresh = useCallback(async () => {
        try {
            const data = await apiRequest('/status');
            setStatus(data);
            setError('');
        } catch (error) {
            setStatus(null);
            setError(error.message);
        }
    }, []);

    useEffect(() => {
        let active = true;
        apiRequest('/status').then((data) => {
            if (active) setStatus(data);
        }).catch((error) => {
            if (active) setError(error.message);
        });
        return () => { active = false; };
    }, []);

    return { status, error, refresh };
}
