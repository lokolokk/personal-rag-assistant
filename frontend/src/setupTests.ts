import '@testing-library/jest-dom/vitest';
import { beforeEach, vi } from 'vitest';

window.HTMLElement.prototype.scrollIntoView = vi.fn();

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('Unmocked fetch is disabled'))));
});

