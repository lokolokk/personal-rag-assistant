import React from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';

vi.mock('../../App', () => ({ default: () => <div>Workspace ready</div> }));

describe('invitation access gate', () => {
  beforeEach(() => {
    vi.resetModules();
    sessionStorage.clear();
    vi.stubEnv('VITE_INVITED_ACCESS', '1');
    vi.stubEnv('VITE_API_URL', 'https://backend.example/api/v1');
    window.history.replaceState(null, '', '/#invite=synthetic-invitation');
  });
  afterEach(() => { cleanup(); vi.unstubAllEnvs(); });

  const successfulSession = (): Response => new Response(JSON.stringify({
    sessionToken: 'synthetic-session', workspaceId: 'a', expiresAt: new Date(Date.now()+3600000).toISOString(), limits: {},
  }), { status: 200 });

  it('removes the fragment before redemption and redeems only once in StrictMode', async () => {
    const spy = vi.fn().mockResolvedValue(successfulSession());
    vi.stubGlobal('fetch', spy);
    const { default: AccessGate } = await import('./AccessGate');
    expect(window.location.hash).toBe('');
    render(<React.StrictMode><AccessGate /></React.StrictMode>);
    await screen.findByText('Workspace ready');
    expect(spy).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem('rag-session-v1')).not.toContain('synthetic-invitation');
  });

  it('retains an offline invitation only in memory and retries redemption', async () => {
    const spy = vi.fn().mockRejectedValueOnce(new Error('Connection failed'))
      .mockResolvedValueOnce(successfulSession());
    vi.stubGlobal('fetch', spy);
    const { default: AccessGate } = await import('./AccessGate');
    render(<AccessGate />);
    await screen.findByText('Backend unavailable. Check the host connection and retry.');
    expect(sessionStorage.length).toBe(0);
    expect(window.location.hash).toBe('');
    fireEvent.click(screen.getByRole('button', { name: 'Retry connection' }));
    await screen.findByText('Workspace ready');
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it('clears revoked access and shows its state without workspace content', async () => {
    window.history.replaceState(null, '', '/');
    sessionStorage.setItem('rag-session-v1', JSON.stringify({
      sessionToken:'synthetic-session', workspaceId:'a', expiresAt:new Date(Date.now()+3600000).toISOString(), limits:{},
    }));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: { code:'ACCESS_REVOKED' },
    }), { status:401 })));
    const { default: AccessGate } = await import('./AccessGate');
    render(<AccessGate />);
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('revoked'));
    expect(screen.queryByText('Workspace ready')).not.toBeInTheDocument();
    expect(sessionStorage.length).toBe(0);
  });
});
