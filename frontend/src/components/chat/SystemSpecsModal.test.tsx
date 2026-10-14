import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { SystemSpecsModal } from './SystemSpecsModal';

describe('SystemSpecsModal component', () => {
  it('does not render when isOpen is false', () => {
    render(
      <SystemSpecsModal
        isOpen={false}
        onClose={vi.fn()}
        selectedDocCount={0}
      />
    );

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('renders modal dialog with accessible attributes and specifications when isOpen is true', () => {
    render(
      <SystemSpecsModal
        isOpen={true}
        onClose={vi.fn()}
        selectedDocCount={2}
      />
    );

    const dialog = screen.getByRole('dialog', { name: /system architecture & telemetry/i });
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveAttribute('aria-modal', 'true');

    // Architecture content assertions
    expect(screen.getByText(/Hybrid Dense \+ Sparse RRF/i)).toBeInTheDocument();
    expect(screen.getAllByText(/ChromaDB/i)[0]).toBeInTheDocument();
    expect(screen.getByText(/qwen3\.7-text-embedding/i)).toBeInTheDocument();
    expect(screen.getAllByText(/BM25/i)[0]).toBeInTheDocument();
    expect(screen.getByText(/RRF k=60/i)).toBeInTheDocument();

    // Verifiable Grounding assertions
    expect(screen.getByText(/Verifiable Grounding & Attribution/i)).toBeInTheDocument();
    expect(screen.getByText(/Source-prefixed chunk framing/i)).toBeInTheDocument();

    // Multi-Doc Scoping status
    expect(screen.getByText(/2 documents active in scope/i)).toBeInTheDocument();

    // Keyboard Shortcuts
    expect(screen.getByText(/Ctrl\+Enter/i)).toBeInTheDocument();
    expect(screen.getByText(/Shift\+Enter/i)).toBeInTheDocument();
    expect(screen.getByText(/Ctrl\+N/i)).toBeInTheDocument();
  });

  it('displays global scoping status when selectedDocCount is 0', () => {
    render(
      <SystemSpecsModal
        isOpen={true}
        onClose={vi.fn()}
        selectedDocCount={0}
      />
    );

    expect(
      screen.getByText(/Global library search active across all documents/i)
    ).toBeInTheDocument();
  });

  it('triggers onClose when close button is clicked', () => {
    const handleClose = vi.fn();
    render(
      <SystemSpecsModal
        isOpen={true}
        onClose={handleClose}
        selectedDocCount={1}
      />
    );

    const closeBtn = screen.getByRole('button', { name: /close system specs/i });
    fireEvent.click(closeBtn);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('triggers onClose on Escape key press', () => {
    const handleClose = vi.fn();
    render(
      <SystemSpecsModal
        isOpen={true}
        onClose={handleClose}
        selectedDocCount={1}
      />
    );

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('triggers onClose when backdrop overlay is clicked', () => {
    const handleClose = vi.fn();
    render(
      <SystemSpecsModal
        isOpen={true}
        onClose={handleClose}
        selectedDocCount={1}
      />
    );

    const backdrop = screen.getByTestId('system-specs-backdrop');
    fireEvent.click(backdrop);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });
});
