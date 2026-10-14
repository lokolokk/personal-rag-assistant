import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { CitationBadge } from './CitationBadge';
import { CitationDetail } from '../../types';

describe('CitationBadge component', () => {
  const sampleCitation: CitationDetail = {
    citationIndex: 1,
    chunkId: 'doc_att_p4_c1',
    documentId: 'doc_att',
    filename: 'attention_is_all_you_need.pdf',
    pageNumber: 4,
    sectionTitle: '3.2 Multi-Head Attention',
    sourceText: 'Multi-head attention allows the model to jointly attend to information...',
  };

  it('renders compact pill with index, truncated filename, and page number', () => {
    render(<CitationBadge citation={sampleCitation} onSelect={vi.fn()} />);

    const badge = screen.getByRole('button', { name: /citation 1/i });
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent('1');
    expect(badge).toHaveTextContent('p.4');
  });

  it('renders fallback index format when filename is missing', () => {
    const minimalCitation: CitationDetail = {
      citationIndex: 3,
      chunkId: 'chunk_3',
      documentId: 'doc_3',
      filename: '',
      pageNumber: 1,
      sectionTitle: '',
      sourceText: 'Some excerpt',
    };

    render(<CitationBadge citation={minimalCitation} onSelect={vi.fn()} />);

    const badge = screen.getByRole('button', { name: /citation 3/i });
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent('[3]');
  });

  it('calls onSelect callback with citation detail when clicked', () => {
    const handleSelect = vi.fn();
    render(<CitationBadge citation={sampleCitation} onSelect={handleSelect} />);

    const badge = screen.getByRole('button', { name: /citation 1/i });
    fireEvent.click(badge);

    expect(handleSelect).toHaveBeenCalledTimes(1);
    expect(handleSelect).toHaveBeenCalledWith(sampleCitation);
  });

  it('provides accessible tooltip title with section and document info', () => {
    render(<CitationBadge citation={sampleCitation} onSelect={vi.fn()} />);

    const badge = screen.getByRole('button', { name: /citation 1/i });
    expect(badge).toHaveAttribute(
      'title',
      expect.stringContaining('attention_is_all_you_need.pdf')
    );
  });
});
