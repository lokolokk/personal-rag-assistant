import { useEffect, useState } from 'react';
import { fetchCostRun, RunView } from '../../services/costDebug';

const number = (v: number | null | undefined) => v == null ? 'Unavailable' : v.toLocaleString('en-US');
export default function RunHistory({ latest, onSelect }: { latest: RunView; onSelect: (run: RunView) => void }) {
  const [opened, setOpened] = useState(false);
  const [runs, setRuns] = useState<RunView[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!opened) return;
    const controller = new AbortController();
    const historicalIds = [
      'quality-five-20261008-02',
      'quality-five-20261008-03',
      'quality-five-20261008-04',
      'quality-five-20261009-05',
    ];
    void Promise.all(historicalIds.filter(id => id !== latest.run_id).map(id => fetchCostRun(controller.signal, id)))
      .then(data => {
        if (!controller.signal.aborted) {
          const all = [...data, latest];
          const deduplicated = Array.from(new Map(all.map(r => [r.run_id, r])).values());
          deduplicated.sort((a, b) => a.run_id.localeCompare(b.run_id));
          setRuns(deduplicated);
        }
      })
      .catch((err: unknown) => { if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'History unavailable'); });
    return () => controller.abort();
  }, [opened, latest]);
  return <section aria-labelledby="history" className="border-t border-zinc-800 py-5"><h2 id="history" className="text-lg font-semibold">Preserved run history</h2>
    <button className="mt-3 px-3 py-2 border border-zinc-700 focus-visible:outline focus-visible:outline-amber-500" onClick={() => setOpened(true)} disabled={opened}>Compare runs (02, 03, 04, 05, 06)</button>
    <p className="mt-3 text-sm text-amber-300">Changed historical configurations do not establish controlled savings. Shared ingestion is counted once per run; per-case figures exclude it.</p>
    {error && <p role="alert" className="mt-3">{error}</p>}
    {opened && !error && runs.length === 0 && <p role="status">Loading preserved history…</p>}
    {runs.map(r => <article key={r.run_id} className="mt-4 border border-zinc-800 p-4 text-sm">
      <button className="font-mono underline break-all text-left focus-visible:outline focus-visible:outline-amber-500" onClick={() => onSelect(r)}>Investigate {r.run_id}</button>
      <p className="mt-2">Terminal status: {r.summary.terminal_state} · completed {r.summary.completed_cases}/{r.summary.total_planned_cases} planned · source review {r.review_status}</p>
      <p className="mt-2">Provider tokens {number(r.summary.calls.exact_provider_tokens)} · known subtotal {number(r.summary.calls.known_provider_tokens)} · unknown calls {r.summary.calls.unknown_usage_calls} · shared ingestion {number(r.ingestion_tokens)} · questions {number(r.question_tokens)}</p>
      <p className="mt-2">Run latency {r.timings ? `${r.timings.run_seconds.toFixed(3)} s` : 'Unavailable'} · answerable quality {r.review?.answerable_successes ?? 'Unavailable'} · genuine absence {r.review?.absence_successes ?? 'Unavailable'} · tokens per strict answerable success: null (zero verified successes or review unavailable)</p>
      <details className="mt-3"><summary className="cursor-pointer focus-visible:outline focus-visible:outline-amber-500">Configuration differences and bindings · {r.run_id}</summary><pre className="whitespace-pre-wrap break-all text-xs mt-2">{JSON.stringify(r.configuration, null, 2)}</pre></details>
      <p className="mt-2 text-xs text-zinc-400">Chat {r.configuration.chat_model} · caps {JSON.stringify(r.configuration.output_caps)} · configuration differs from primary: {r.configuration.configuration_sha256 === latest.configuration.configuration_sha256 ? 'no' : 'yes'} · review provenance: {r.review_provenance}</p>
      <div className="overflow-x-auto mt-3"><table className="w-full text-left text-xs"><caption className="text-left mb-2">Per-case expenditure, latency and reviewed quality</caption><thead><tr>{['Case', 'Completion', 'Question tokens', 'Latency', 'Human case review'].map(h => <th scope="col" key={h} className="p-2">{h}</th>)}</tr></thead><tbody>{r.questions.map(q => <tr key={q.case_id} className="border-t border-zinc-800"><th scope="row" className="p-2 font-mono break-words">{q.case_id}</th><td className="p-2">{q.completed ? 'Completed' : q.calls.length ? 'Interrupted' : 'Not executed'}</td><td className="p-2">{number(q.tokens)}</td><td className="p-2">{q.seconds == null ? 'Unavailable' : `${q.seconds.toFixed(3)} s`}</td><td className="p-2">{r.review?.case_review.find(c => c.case === q.case_id)?.verdict ?? 'Unavailable; cohort review does not establish per-case review'}</td></tr>)}</tbody></table></div>
      {r.missing.length > 0 && <p className="text-amber-300 mt-3">{r.missing.join(' · ')}</p>}
    </article>)}
  </section>;
}
