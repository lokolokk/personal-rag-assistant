import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { MessageBubble } from './MessageBubble';
import { DisplayMessage } from '../../types';

afterEach(() => { cleanup(); vi.unstubAllEnvs(); });

it('makes only validated live citations clickable', () => {
  vi.stubEnv('VITE_INVITED_ACCESS','1');
  const message: DisplayMessage = { id:'draft',role:'assistant',content:'Draft [1].',citations:[],
    createdAt:new Date().toISOString(),isStreaming:true };
  const { rerender } = render(<MessageBubble message={message} />);
  expect(screen.queryByRole('button',{name:/Citation 1/})).not.toBeInTheDocument();
  rerender(<MessageBubble message={{...message,citations:[{
    citationIndex:1,chunkId:'owned_chunk',documentId:'owned_doc',filename:'owned.pdf',
    pageNumber:1,sectionTitle:'Test',sourceText:'Synthetic evidence.',
  }]}} />);
  expect(screen.getByRole('button',{name:'Citation 1: owned.pdf, page 1'})).toBeInTheDocument();
});
