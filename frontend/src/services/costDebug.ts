import { request } from './api';

export interface StageUsage {
  stage: string;
  calls: number;
  known_provider_tokens: number;
  exact_provider_tokens: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  unknown_usage_calls: number;
  synthetic_tokens: number;
  input_estimate_tokens: number;
  reserved_tokens: number;
  seconds: number | null;
}

export interface CaseReview {
  case: string;
  verdict: string;
  earliest_issue: string | null;
  correctness: string | null;
  qualification: string | null;
  citation: string | null;
  review: string | null;
}

export interface RunView {
  questions: QuestionView[];
  run_id: string;
  complete: boolean;
  summary: {
    terminal_state: string;
    completed_cases: number;
    total_planned_cases: number;
    unreviewed_cases: number;
    calls: {
      potentially_dispatched_calls: number;
      known_provider_tokens: number;
      exact_provider_tokens: number | null;
      unknown_usage_calls: number;
      synthetic_tokens: number;
      reserved_tokens: number;
    };
    current: { local_estimate_tokens: number; tokens_per_successful_answer: number | null };
    historical: { exact_provider_tokens: number | null; unknown_provider_attempts: number };
  };
  configuration: {
    chat_model: string;
    embedding_model: string;
    output_caps: Record<string, number>;
    retries: number;
    maximum_calls: number;
    allowance_tokens: number;
    execution_mode: string;
    configuration_sha256: string;
    dataset_sha256: string;
    prompt_sha256: Record<string, string>;
    corpus_sha256: Record<string, string>;
  };
  stages: StageUsage[];
  ingestion_tokens: number | null;
  question_tokens: number | null;
  timings: {
    run_seconds: number;
    ingestion_seconds: number;
    ttft_seconds: number | null;
    cases: Record<string, { total_seconds: number; stages: Record<string, number> }>;
  } | null;
  automatic_cases: { case_id: string; result_kind: string; failure_codes: string[]; review_complete: boolean }[];
  review: {
    answerable_successes: string;
    absence_successes: string;
    case_review: CaseReview[];
    automatic_labels_limitation: string;
    reasoning_finish_reason: string;
  } | null;
  review_status: 'bound' | 'missing' | 'incomplete';
  review_provenance: string;
  verified_snapshots: number;
  missing: string[];
  limitations: string[];
}

export interface RetainedChunk {
  chunk_id: string;
  content: string;
  score: number | null;
  metadata: { document_id: string; filename: string; page_number: number; section_title: string; is_table: boolean };
}

export interface Snapshot {
  question: string;
  standalone_query: string | null;
  raw_grade: string | null;
  route: string;
  response_text: string | null;
  declared_scope: string[];
  effective_scope: string[];
  history_sha256: string;
  retrieved_chunks: RetainedChunk[] | null;
  supplied_chunks: RetainedChunk[] | null;
  rendered_context: string | null;
  citations: { chunk_id: string; document_id: string; filename: string; page_number: number; quote: string; section_title: string }[] | null;
  request_sha256: Record<string, string>;
}

export interface QuestionView {
  case_id: string;
  completed: boolean;
  tokens: number | null;
  seconds: number | null;
  stages: { node: string; provenance: string; snapshot: Snapshot | null; seconds: number | null }[];
  usage: StageUsage[];
  calls: { state: string; reservation: { stage: string; call_id: number; attempt: number; payload_sha256: string; reserved_tokens: number }; outcome: { source: string; input_tokens: number | null; output_tokens: number | null; total_tokens: number | null; provider_request_sha256: string | null } | null }[];
}

export const DEFAULT_RUN_ID = 'quality-five-20261010-06';
export const BASELINE_RUN_ID = 'quality-five-20261008-04';

export function fetchCostRun(signal?: AbortSignal, runId = DEFAULT_RUN_ID): Promise<RunView> {
  return request<RunView>(`/cost-debug/${encodeURIComponent(runId)}`, {
    method: 'GET', signal, cache: 'no-store',
  });
}

export function costDebugEnabled(): boolean {
  return import.meta.env.DEV && ['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname)
    && new URLSearchParams(window.location.search).get('cost-debug') === '1';
}
