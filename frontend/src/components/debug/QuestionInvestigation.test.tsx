import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import QuestionInvestigation from './QuestionInvestigation';
import { RunView, Snapshot } from '../../services/costDebug';

const snapshot: Snapshot = {
  question: 'For ChatGPT, English and Chinese Rej/Rej*?', standalone_query: 'What are the English Rej/Rej',
  raw_grade: '', route: 'insufficient', response_text: 'Cannot find answer',
  declared_scope: ['rgb'], effective_scope: ['rgb'], history_sha256: 'history', request_sha256: { grading: 'request' },
  retrieved_chunks: [{ chunk_id: 'row', content: 'ChatGPT 24.67 45.00 5.33 43.33', score: 0.03,
    metadata: { document_id: 'rgb', filename: 'rgb.pdf', page_number: 6, section_title: 'Table 3', is_table: true } }],
  supplied_chunks: [], rendered_context: 'Exact context', citations: [{ chunk_id: 'row', document_id: 'rgb', filename: 'rgb.pdf', page_number: 6, quote: 'ChatGPT', section_title: 'Table 3' }],
};
const run = {
  questions: ['rgb-06-table3-row-extraction-gap', 'rag-08-history-followup-wq'].map(case_id => ({ case_id, completed: true, tokens: 2038, seconds: 34,
    calls: [], usage: [], stages: ['initial', 'reformulate_query', 'grade_retrieval', 'inability_fallback'].map(node => ({ node, provenance: `inputs/${node}/${case_id}`, snapshot, seconds: 1 })) })),
  review: null, automatic_cases: [], review_provenance: 'human-review.json',
} satisfies Pick<RunView, 'questions' | 'review' | 'automatic_cases' | 'review_provenance'>;

describe('Question investigation', () => {
  it('starts with RGB and reveals exact ordered evidence on demand', () => {
    render(<QuestionInvestigation run={run} />);
    expect(screen.getByRole('combobox')).toHaveValue('rgb-06-table3-row-extraction-gap');
    expect(screen.getByText(snapshot.question)).toBeInTheDocument();
    expect(screen.getByText(snapshot.standalone_query!)).toBeInTheDocument();
    expect(screen.getByText(/Grade: Empty retained output/)).toBeInTheDocument();
    expect(screen.getAllByText(/\[1\] rgb.pdf · p.6/)).toHaveLength(4);
    expect(screen.getAllByText(/ChatGPT 24.67 45.00 5.33 43.33/)).toHaveLength(4);
    expect(screen.getByText(/Review unavailable; quality unknown/)).toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'rag-08-history-followup-wq' } });
    expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent('rag-08-history-followup-wq');
  });
  it('labels a missing snapshot without reconstructing it', () => {
    const missing = { ...run, questions: [{ ...run.questions[0], tokens: null, stages: [{ node: 'initial', provenance: 'missing', seconds: null, snapshot: null }] }] };
    render(<QuestionInvestigation run={missing} />);
    expect(screen.getByText(/Stage snapshot unavailable; no evidence reconstructed/)).toBeInTheDocument();
    expect(screen.getByText(/Question expenditure: Unavailable/)).toBeInTheDocument();
  });
});

