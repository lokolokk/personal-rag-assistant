import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CitationDrawer } from './CitationDrawer';
import { CitationDetail } from '../../types';

describe('CitationDrawer Component', () => {
  const sampleCitation1: CitationDetail = {
    citationIndex: 1,
    chunkId: 'chk_att_01',
    documentId: 'doc_att_123',
    filename: 'attention_is_all_you_need.pdf',
    pageNumber: 4,
    sectionTitle: 'Section 3.2: Multi-Head Attention',
    sourceText: 'Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.',
  };

  const sampleCitation2: CitationDetail = {
    citationIndex: 2,
    chunkId: 'chk_ref_02',
    documentId: 'doc_ref_456',
    filename: 'reformer_efficient_transformer.pdf',
    pageNumber: 7,
    sectionTitle: 'Section 2.1: Locality-Sensitive Hashing',
    sourceText: 'We use locality-sensitive hashing to reduce the complexity of attention from O(L^2) to O(L log L).',
  };

  const sampleCitations = [sampleCitation1, sampleCitation2];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders nothing or is hidden when isOpen is false', () => {
    const { container } = render(
      <CitationDrawer
        isOpen={false}
        citation={sampleCitation1}
        citations={sampleCitations}
        onClose={vi.fn()}
      />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when citation is null even if isOpen is true', () => {
    const { container } = render(
      <CitationDrawer
        isOpen={true}
        citation={null}
        citations={[]}
        onClose={vi.fn()}
      />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders dialog with accessibility attributes when open with active citation', () => {
    render(
      <CitationDrawer
        isOpen={true}
        citation={sampleCitation1}
        citations={sampleCitations}
        onClose={vi.fn()}
      />
    );

    const dialog = screen.getByRole('dialog', { name: /citation inspector/i });
    expect(dialog).toBeInTheDocument();
    expect(dialog).toHaveAttribute('aria-modal', 'true');
  });

  it('displays attribution metadata: citation badge, filename, page, section, and excerpt', () => {
    render(
      <CitationDrawer
        isOpen={true}
        citation={sampleCitation1}
        citations={sampleCitations}
        onClose={vi.fn()}
      />
    );

    // Filename
    expect(screen.getByText('attention_is_all_you_need.pdf')).toBeInTheDocument();
    // Page number indicator
    expect(screen.getAllByText(/page 4/i).length).toBeGreaterThan(0);
    // Section title
    expect(screen.getByText(/section 3\.2: multi-head attention/i)).toBeInTheDocument();
    // Source excerpt text
    expect(
      screen.getByText(/Multi-Head Attention allows the model to jointly attend/i)
    ).toBeInTheDocument();
  });

  it('allows switching between citations via tabs and prev/next buttons', () => {
    const onSelectCitationMock = vi.fn();

    render(
      <CitationDrawer
        isOpen={true}
        citation={sampleCitation1}
        citations={sampleCitations}
        onClose={vi.fn()}
        onSelectCitation={onSelectCitationMock}
      />
    );

    // Both citation tabs should exist as accessible buttons
    const tab1 = screen.getByRole('button', { name: /citation 1/i });
    const tab2 = screen.getByRole('button', { name: /citation 2/i });
    expect(tab1).toBeInTheDocument();
    expect(tab2).toBeInTheDocument();

    // Click tab 2
    fireEvent.click(tab2);
    expect(onSelectCitationMock).toHaveBeenCalledWith(sampleCitation2);

    // Click Next button
    const nextBtn = screen.getByRole('button', { name: /next citation/i });
    fireEvent.click(nextBtn);
    expect(onSelectCitationMock).toHaveBeenCalledWith(sampleCitation2);
  });

  it('copies excerpt text to clipboard with visual feedback', async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    render(
      <CitationDrawer
        isOpen={true}
        citation={sampleCitation1}
        citations={sampleCitations}
        onClose={vi.fn()}
      />
    );

    const copyBtn = screen.getByRole('button', { name: /copy excerpt/i });
    fireEvent.click(copyBtn);

    expect(writeTextMock).toHaveBeenCalledWith(sampleCitation1.sourceText);
    expect(await screen.findByText(/copied/i)).toBeInTheDocument();
  });

  it('calls onClose when close button, backdrop, or Escape key is pressed', () => {
    const onCloseMock = vi.fn();

    render(
      <CitationDrawer
        isOpen={true}
        citation={sampleCitation1}
        citations={sampleCitations}
        onClose={onCloseMock}
      />
    );

    // 1. Close button click
    const closeBtn = screen.getByRole('button', { name: /close inspector/i });
    fireEvent.click(closeBtn);
    expect(onCloseMock).toHaveBeenCalledTimes(1);

    // 2. Backdrop click
    const backdrop = screen.getByTestId('citation-drawer-backdrop');
    fireEvent.click(backdrop);
    expect(onCloseMock).toHaveBeenCalledTimes(2);

    // 3. Escape key press
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onCloseMock).toHaveBeenCalledTimes(3);
  });

  it('supports desktop side-by-side evidence docking mode with evidence-grounding-pane testid', () => {
    render(
      <CitationDrawer
        isOpen={true}
        citation={sampleCitation1}
        citations={sampleCitations}
        onClose={vi.fn()}
      />
    );

    const pane = screen.getByTestId('evidence-grounding-pane');
    expect(pane).toBeInTheDocument();
    expect(pane).toHaveClass('xl:static');
    const backdrop = screen.getByTestId('citation-drawer-backdrop');
    expect(backdrop).toHaveClass('xl:hidden');
  });
});
