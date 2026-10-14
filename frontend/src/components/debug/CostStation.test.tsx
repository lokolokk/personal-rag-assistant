import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import CostStation from './CostStation';
import { costDebugEnabled, RunView } from '../../services/costDebug';

const evidence: RunView = {
  questions: [],
  run_id: 'quality-five-20261008-04', complete: true, review_status: 'bound',
  review_provenance: 'backend/.eval-reports/task_6_1/full_pipeline_20261008/online-run-review.json',
  verified_snapshots: 25, missing: [], limitations: ['Tokens are not charges.'],
  summary: { terminal_state: 'completed', completed_cases: 5, total_planned_cases: 5,
    unreviewed_cases: 5, calls: { potentially_dispatched_calls: 27, exact_provider_tokens: 69774,
      known_provider_tokens: 69774, unknown_usage_calls: 0, synthetic_tokens: 0, reserved_tokens: 137330 },
    current: { local_estimate_tokens: 0, tokens_per_successful_answer: null },
    historical: { exact_provider_tokens: 0, unknown_provider_attempts: 0 } },
  configuration: { chat_model: 'qwen3.8-flash', embedding_model: 'qwen3.7-text-embedding',
    output_caps: { reformulation: 256, grading: 2048, generation: 4096 }, retries: 0,
    maximum_calls: 30, allowance_tokens: 195254, execution_mode: 'synchronous',
    configuration_sha256: 'a'.repeat(64), prompt_sha256: {}, corpus_sha256: {}, dataset_sha256: 'b'.repeat(64) },
  ingestion_tokens: 47409, question_tokens: 22365,
  timings: { run_seconds: 94.997, ingestion_seconds: 6.882, ttft_seconds: null,
    cases: { rgb: { total_seconds: 20.165, stages: { grade_retrieval: 3.53 } } } },
  stages: [{ stage: 'ingestion_embedding', calls: 10, exact_provider_tokens: 47409,
    known_provider_tokens: 47409, input_tokens: 47409, output_tokens: 0,
    unknown_usage_calls: 0, synthetic_tokens: 0, input_estimate_tokens: 100, reserved_tokens: 84914, seconds: 6.882 },
    { stage: 'grading', calls: 5, exact_provider_tokens: 14551, known_provider_tokens: 14551,
      input_tokens: 11573, output_tokens: 2978, unknown_usage_calls: 0, synthetic_tokens: 0,
      input_estimate_tokens: 100, reserved_tokens: 35000, seconds: 51.107 },
    ...[['query_embedding', 5, 226], ['reformulation', 5, 2374], ['generation', 2, 5214]].map(([stage, calls, tokens]) => ({
      stage: String(stage), calls: Number(calls), exact_provider_tokens: Number(tokens), known_provider_tokens: Number(tokens),
      input_tokens: Number(tokens), output_tokens: 0, unknown_usage_calls: 0, synthetic_tokens: 0,
      input_estimate_tokens: 100, reserved_tokens: 100, seconds: null,
    }))],
  automatic_cases: [{ case_id: 'rgb', result_kind: 'answer', failure_codes: ['extraction_gap', 'citation_support_failure'], review_complete: false }],
  review: { answerable_successes: '0/4 strict complete; two partial factually supported generated answers',
    absence_successes: '1/1', automatic_labels_limitation: 'RGB extraction/citation labels contradicted by human review.',
    reasoning_finish_reason: 'Precise cause unproved',
    case_review: [{ case: 'rgb', verdict: 'partial answer; strict fail', earliest_issue: 'Rewrite lost Chinese constraints',
      citation: 'Page 6 citation supported', correctness: 'English correct; Chinese omitted', qualification: null, review: null }] },
};

describe('CostStation', () => {
  it('requires explicit local development opt-in', () => {
    vi.stubEnv('DEV', true);
    window.history.replaceState(null, '', '/');
    expect(costDebugEnabled()).toBe(false);
    window.history.replaceState(null, '', '/?cost-debug=1');
    expect(costDebugEnabled()).toBe(true);
    vi.stubEnv('DEV', false);
    expect(costDebugEnabled()).toBe(false);
    window.history.replaceState(null, '', '/');
    vi.unstubAllEnvs();
  });
  it('renders reconciled usage and distinct review provenance using only GET', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(evidence)));
    vi.stubGlobal('fetch', fetchMock);
    render(<CostStation />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading retained evidence');
    expect(await screen.findByText('69,774')).toBeInTheDocument();
    expect(screen.getByText('47,409 tokens · 10 calls')).toBeInTheDocument();
    expect(screen.getByText('22,365 tokens · 17 calls')).toBeInTheDocument();
    expect(screen.getByText(/0\/4 strict complete/)).toBeInTheDocument();
    expect(screen.getByText(/Correct genuine absence: 1\/1/)).toBeInTheDocument();
    expect(screen.getByText('Rewrite lost Chinese constraints')).toBeInTheDocument();
    expect(screen.getByText(/extraction_gap, citation_support_failure/)).toBeInTheDocument();
    expect(screen.getByText(/Tokens are not charges/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/cost-debug/quality-five-20261010-06');
    expect(fetchMock.mock.calls[0][1].method).toBe('GET');
  });

  it('does not turn missing usage or review into zero/success', async () => {
    const missing = { ...evidence, complete: false, review: null, review_status: 'missing',
      question_tokens: null, missing: ['execution.json unavailable'], timings: null,
      summary: { ...evidence.summary, calls: { ...evidence.summary.calls, exact_provider_tokens: null, unknown_usage_calls: 1 } } };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(missing))));
    render(<CostStation />);
    expect(await screen.findByText(/Incomplete evidence/)).toBeInTheDocument();
    expect(screen.getByText('execution.json unavailable')).toBeInTheDocument();
    expect(screen.getByText(/Source review unavailable/)).toBeInTheDocument();
    expect(screen.getAllByText(/Unavailable/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/0\/4/)).not.toBeInTheDocument();
  });

  it('shows API failure without presenting historical quality', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('disabled', { status: 404 })));
    render(<CostStation />);
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('disabled'));
    expect(screen.queryByText('69,774')).not.toBeInTheDocument();
  });

  it('compares interrupted history without treating unknown usage or reviews as success', async () => {
    const interrupted: RunView = { ...evidence, run_id: 'quality-five-20261008-02', complete: false,
      review: null, review_status: 'missing',
      configuration: { ...evidence.configuration, output_caps: { grading: 32 }, configuration_sha256: 'c'.repeat(64) },
      summary: { ...evidence.summary, completed_cases: 0, terminal_state: 'incomplete' },
      questions: [{ case_id: 'interrupted-case', completed: false, tokens: null, seconds: null, stages: [], calls: [], usage: [] }],
    };
    const currentRun: RunView = { ...evidence, run_id: 'quality-five-20261010-06' };
    const fetchMock = vi.fn().mockImplementation((url: string) => Promise.resolve(new Response(JSON.stringify(url.endsWith('02') ? interrupted : currentRun))));
    vi.stubGlobal('fetch', fetchMock);
    render(<CostStation />);
    await screen.findByText('69,774');
    fireEvent.click(screen.getByRole('button', { name: 'Compare runs (02, 03, 04, 05, 06)' }));
    await screen.findByRole('button', { name: 'Investigate quality-five-20261008-02' });
    expect(screen.getByText(/Terminal status: incomplete · completed 0\/5/)).toBeInTheDocument();
    expect(screen.getByText('Not executed')).toBeInTheDocument();
    expect(screen.getByText(/Changed historical configurations do not establish controlled savings/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Investigate quality-five-20261008-02' }));
    expect(screen.getByText(/Incomplete evidence/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(5);
    expect(fetchMock.mock.calls.every(c => c[1].method === 'GET')).toBe(true);
  });
});
