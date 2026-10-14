import { useEffect, useState } from 'react';
import { DEFAULT_RUN_ID, fetchCostRun, RunView } from '../../services/costDebug';
import QuestionInvestigation from './QuestionInvestigation';
import RunHistory from './RunHistory';
import BaselineSpecification from './BaselineSpecification';

const number = (value: number | null | undefined): string =>
  value == null ? 'Unavailable' : value.toLocaleString('en-US');
const seconds = (value: number | null | undefined): string =>
  value == null ? 'Unavailable' : `${value.toFixed(3)} s`;
const labels: Record<string, string> = {
  ingestion_embedding: 'Shared ingestion', query_embedding: 'Question embedding',
  reformulation: 'Reformulation', grading: 'Grading', generation: 'Generation',
};

export function RunEvidence({ run }: { run: RunView }) {
  const calls = run.summary.calls;
  const ingestionCalls = run.stages.find(s => s.stage === 'ingestion_embedding')?.calls ?? 0;
  const questionCalls = run.stages.filter(s => s.stage !== 'ingestion_embedding').reduce((n, s) => n + s.calls, 0);
  return (
    <>
      <QuestionInvestigation key={run.run_id} run={run} />
      <section className="border-y border-zinc-800 py-5" aria-label="Run overview">
        <p className="text-sm text-zinc-300">{run.complete ? 'Completed evidence' : 'Incomplete evidence'} · {run.summary.completed_cases}/{run.summary.total_planned_cases} cases · Source review: {run.review_status}</p>
        <div className="flex flex-wrap gap-x-12 gap-y-4 mt-4">
          <div><p className="text-xs text-zinc-400">Provider-reported tokens</p><p className="text-3xl font-mono mt-1">{number(calls.exact_provider_tokens)}</p></div>
          <div><p className="text-xs text-zinc-400">Recorded dispatches</p><p className="text-3xl font-mono mt-1">{calls.potentially_dispatched_calls}</p></div>
          <div><p className="text-xs text-zinc-400">Run time · excludes setup</p><p className="text-3xl font-mono mt-1">{seconds(run.timings?.run_seconds)}</p></div>
        </div>
        <p className="text-xs text-zinc-400 mt-4">Known provider subtotal: {number(calls.known_provider_tokens)} · Unknown usage calls: {calls.unknown_usage_calls} · Configured retries: {run.configuration.retries}</p>
      </section>

      <section className="py-5" aria-labelledby="expenditure">
        <h2 id="expenditure" className="text-lg font-semibold">Where tokens went</h2>
        <div className="grid md:grid-cols-2 gap-4 my-4">
          <div className="border-l-2 border-zinc-600 pl-4"><h3 className="text-sm text-zinc-300">Shared ingestion · once per evaluation run</h3><p className="font-mono mt-1">{number(run.ingestion_tokens)} tokens · {ingestionCalls} calls</p></div>
          <div className="border-l-2 border-amber-600 pl-4"><h3 className="text-sm text-zinc-300">Recurring question stages</h3><p className="font-mono mt-1">{number(run.question_tokens)} tokens · {questionCalls} calls</p></div>
        </div>
        <div className="overflow-x-auto border border-zinc-800">
          <table className="w-full text-sm text-left whitespace-nowrap">
            <caption className="sr-only">Provider usage and observed wall time by stage</caption>
            <thead className="bg-zinc-900 text-zinc-400"><tr>{['Stage', 'Calls', 'Input', 'Output', 'Total tokens', 'Unknown calls', 'Observed time'].map(h => <th key={h} scope="col" className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
            <tbody className="divide-y divide-zinc-800">{run.stages.map(s => <tr key={s.stage}>
              <th scope="row" className="px-4 py-3 font-medium">{labels[s.stage] ?? s.stage}</th>
              {[number(s.calls), number(s.input_tokens), number(s.output_tokens), number(s.exact_provider_tokens), number(s.unknown_usage_calls), seconds(s.seconds)].map((v, i) => <td key={i} className="px-4 py-3 font-mono text-zinc-300">{v}</td>)}
            </tr>)}</tbody>
          </table>
        </div>
        <p className="text-xs text-zinc-400 mt-3">Stage times are graph wall times. Question embedding time is included in retrieval; a separate duration is unavailable.</p>
        <details className="mt-3 text-sm"><summary className="cursor-pointer text-zinc-300 focus-visible:outline focus-visible:outline-amber-500">Reservations, estimates and synthetic evidence</summary>
          <p className="text-zinc-400 mt-2">Reserved allowance used: {number(calls.reserved_tokens)} / {number(run.configuration.allowance_tokens)}. Synthetic usage: {number(calls.synthetic_tokens)}. Local usage estimates: {number(run.summary.current.local_estimate_tokens)}. Imported historical exact tokens: {number(run.summary.historical.exact_provider_tokens)}; unknown attempts: {run.summary.historical.unknown_provider_attempts}.</p>
          <ul className="text-zinc-400 mt-2 space-y-1">{run.stages.map(s => <li key={s.stage}>{labels[s.stage]}: input reservation estimate {number(s.input_estimate_tokens)}; reservation {number(s.reserved_tokens)}; synthetic tokens {number(s.synthetic_tokens)}</li>)}</ul>
          <p className="text-zinc-400 mt-2">Estimates and reservations are separate from provider usage. Missing usage remains unavailable.</p>
        </details>
      </section>

      <section className="border-t border-zinc-800 py-5" aria-labelledby="quality">
        <h2 id="quality" className="text-lg font-semibold">Quality · source-reviewed closeout</h2>
        {run.review ? <>
          <p className="mt-3 text-amber-300">Answerable cases: {run.review.answerable_successes}</p>
          <p className="mt-2 text-zinc-200">Correct genuine absence: {run.review.absence_successes}</p>
          <p className="text-xs text-zinc-400 mt-2">Tokens per fully correct answerable answer: unavailable with zero strict successes.</p>
          <p className="text-sm text-zinc-300 border-l-2 border-amber-600 pl-3 my-4">{run.review.automatic_labels_limitation}</p>
          <div className="divide-y divide-zinc-800">{run.review.case_review.map(c => <article key={c.case} className="py-3">
            <h3 className="text-sm font-mono break-words">{c.case}</h3><p className="text-sm text-amber-300 mt-1">{c.verdict}</p>
            <p className="text-sm text-zinc-300 mt-1">{c.earliest_issue ?? 'No supported failure identified'}</p>
            <details className="mt-2 text-xs text-zinc-400"><summary className="cursor-pointer focus-visible:outline focus-visible:outline-amber-500">Source review notes</summary><ul className="mt-2 space-y-1">{[c.correctness, c.qualification, c.citation, c.review].filter(Boolean).map((note, i) => <li key={i}>{note}</li>)}</ul></details>
          </article>)}</div>
          <p className="text-xs text-zinc-400 mt-3">{run.review.reasoning_finish_reason}</p>
        </> : <p className="mt-3 text-amber-300">Source review unavailable or incompletely bound. Quality is unknown.</p>}
        <p className="text-xs text-zinc-400 mt-4 break-all">Review provenance: {run.review_provenance}. Bound file hashes verify association with retained artifacts; this closeout is separate from the journal semantic-review contract.</p>
        <details className="mt-4 text-sm"><summary className="cursor-pointer text-zinc-300 focus-visible:outline focus-visible:outline-amber-500">Original automatic journal · {run.summary.unreviewed_cases} unreviewed cases</summary>
          <ul className="mt-3 space-y-3 text-xs text-zinc-400">{run.automatic_cases.map(c => <li key={c.case_id} className="break-words"><span className="font-mono text-zinc-300">{c.case_id}</span>: {c.result_kind} · {c.failure_codes.join(', ') || 'No automatic failure code'} · Semantic review {c.review_complete ? 'complete' : 'incomplete'}</li>)}</ul>
          <p className="mt-2 text-xs text-zinc-400">Provenance: original cases.jsonl and recovered summary. Automatic labels are preserved and may conflict with source review.</p>
        </details>
      </section>

      <section className="border-t border-zinc-800 py-5" aria-labelledby="config">
        <h2 id="config" className="text-lg font-semibold">Retained configuration and capture</h2>
        <p className="text-sm text-zinc-300 mt-3 break-words">Chat: {run.configuration.chat_model} · Embeddings: {run.configuration.embedding_model} · {run.configuration.execution_mode} · {run.configuration.maximum_calls}-call ceiling</p>
        <p className="text-sm text-zinc-400 mt-2">Hard output caps: {Object.entries(run.configuration.output_caps).map(([stage, cap]) => `${labels[stage] ?? stage} ${cap}`).join(' · ')}</p>
        <p className="text-sm text-zinc-400 mt-2">Verified snapshots: {run.verified_snapshots} · Streamed TTFT: {seconds(run.timings?.ttft_seconds)} · Monetary charges: unavailable</p>
        <details className="mt-3 text-sm"><summary className="cursor-pointer text-zinc-300 focus-visible:outline focus-visible:outline-amber-500">Configuration, dataset, prompt and corpus hashes</summary>
          <dl className="text-xs text-zinc-400 mt-3 space-y-2">{Object.entries({ configuration: run.configuration.configuration_sha256, dataset: run.configuration.dataset_sha256, ...run.configuration.prompt_sha256, ...run.configuration.corpus_sha256 }).map(([name, hash]) => <div key={name}><dt>{name}</dt><dd className="font-mono break-all">{hash}</dd></div>)}</dl>
        </details>
        <details className="mt-3 text-sm"><summary className="cursor-pointer text-zinc-300 focus-visible:outline focus-visible:outline-amber-500">Per-case observed timings</summary>
          {run.timings ? <ul className="text-xs text-zinc-400 mt-3 space-y-3">{Object.entries(run.timings.cases).map(([id, t]) => <li key={id} className="break-words"><span className="font-mono text-zinc-300">{id}</span> · {seconds(t.total_seconds)}<p className="mt-1">{Object.entries(t.stages).map(([s, v]) => `${s}: ${seconds(v)}`).join(' · ')}</p></li>)}</ul> : <p className="mt-2 text-zinc-400">Timings unavailable</p>}
        </details>
        {run.missing.length > 0 && <div role="alert" className="mt-4 text-sm text-amber-300"><p>Missing evidence</p><ul className="mt-2 space-y-1 break-all">{run.missing.map(m => <li key={m}>{m}</li>)}</ul></div>}
        <ul className="mt-4 text-xs text-zinc-400 space-y-2">{run.limitations.map(l => <li key={l}>{l}</li>)}</ul>
      </section>
    </>
  );
}

export default function CostStation() {
  const [run, setRun] = useState<RunView | null>(null);
  const [latest, setLatest] = useState<RunView | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void fetchCostRun(controller.signal).then(data => {
      if (!controller.signal.aborted) { setRun(data); setLatest(data); }
    }).catch((err: unknown) => {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Evidence request failed');
    });
    return () => controller.abort();
  }, []);
  return (
    <main className="h-full overflow-y-auto bg-zinc-950 text-zinc-100 font-sans">
      <div className="max-w-6xl mx-auto px-4 sm:px-8 py-6">
        <header className="mb-6"><p className="text-xs text-zinc-400 uppercase tracking-wider">Local evidence · read only</p><h1 className="text-2xl font-semibold mt-2">RAG Cost Debugging Station</h1><p className="text-sm font-mono text-zinc-400 mt-2 break-all">{run?.run_id ?? DEFAULT_RUN_ID}</p><p className="text-xs text-zinc-400 mt-3">Retained evaluation expenditure. Viewing sends no provider requests.</p></header>
        {error ? <div role="alert" className="border border-amber-800 p-4 text-amber-300"><p>{error}</p><p className="text-sm mt-2">Start the backend locally with RAG_COST_DEBUG=1. Evidence binding failures require inspection of retained files.</p></div> : run ? <RunEvidence run={run} /> : <p role="status" className="text-zinc-400">Loading retained evidence…</p>}
        {latest && <RunHistory latest={latest} onSelect={setRun} />}
        <BaselineSpecification />
      </div>
    </main>
  );
}
