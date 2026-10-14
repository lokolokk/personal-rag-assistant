import { useState } from 'react';
import { QuestionView, RetainedChunk, RunView } from '../../services/costDebug';

type InvestigationRun = Pick<RunView, 'questions' | 'review' | 'automatic_cases' | 'review_provenance'>;

const value = (v: number | null | undefined) => v == null ? 'Unavailable' : v.toLocaleString('en-US');
const names: Record<string, string> = { initial: 'Original question', reformulate_query: 'Actual rewrite', bypass_reformulation: 'Bypassed rewrite (Turn 1)', scoped_hybrid_retriever: 'Retrieval', grade_retrieval: 'Grading', generate_grounded_response: 'Generation', inability_fallback: 'Refusal' };
function TextEvidence({ label, text }: { label: string; text: string | null }) {
  return <details className="mt-3"><summary className="cursor-pointer text-zinc-300 focus-visible:outline focus-visible:outline-amber-500">{label}</summary><pre className="whitespace-pre-wrap break-words text-xs text-zinc-300 mt-2 border-l border-zinc-700 pl-3">{text === null ? 'Unavailable' : text === '' ? 'Empty retained output (0 characters)' : text}</pre></details>;
}
function Chunks({ label, chunks }: { label: string; chunks: RetainedChunk[] | null }) {
  return <details className="mt-3"><summary className="cursor-pointer focus-visible:outline focus-visible:outline-amber-500">{label} · {chunks === null ? 'Unavailable' : chunks.length}</summary>
    {chunks?.map((c, i) => <details key={c.chunk_id} className="mt-3 pl-3 border-l border-zinc-700"><summary className="cursor-pointer break-words focus-visible:outline focus-visible:outline-amber-500">[{i + 1}] {c.metadata.filename} · p.{c.metadata.page_number} · {c.metadata.section_title} · score {value(c.score)}</summary><p className="text-xs font-mono break-all mt-2">{c.chunk_id} · scope {c.metadata.document_id}</p><pre className="whitespace-pre-wrap break-words text-xs mt-2">{c.content}</pre></details>)}
    {chunks?.length === 0 && <p className="mt-2">Retained empty list</p>}
  </details>;
}
function Timeline({ question, run }: { question: QuestionView; run: InvestigationRun }) {
  const review = run.review?.case_review.find(c => c.case === question.case_id);
  const automatic = run.automatic_cases.find(c => c.case_id === question.case_id);
  return <article className="mt-4 text-sm">
    <h3 className="font-mono break-words">{question.case_id}</h3>
    <p className="mt-2">Question expenditure: {value(question.tokens)} tokens · {question.seconds == null ? 'Latency unavailable' : `${question.seconds.toFixed(3)} s`} · {question.completed ? 'Completed case' : 'Interrupted / not completed'}</p>
    <div className="border-l-2 border-amber-600 pl-3 mt-4"><p className="text-amber-300">Human-reviewed diagnosis</p><p className="mt-1">{review?.earliest_issue ?? (review ? 'No supported issue' : 'Review unavailable; quality unknown')}</p><p className="mt-1">{review?.verdict}</p><TextEvidence label="Human source review details and provenance" text={review ? `${run.review_provenance}\n${[review.correctness, review.qualification, review.citation, review.review].filter(Boolean).join('\n')}` : null} /></div>
    <p className="mt-3 text-zinc-400">Automatic journal findings: {automatic?.failure_codes.join(', ') || 'Unavailable / no automatic code'}. These are separate from source review.</p>
    <p className="mt-3 text-zinc-400">Hypothesis: output truncation or hidden reasoning may contribute to incomplete outputs; precise cause is unproved. Finish reasons and reasoning-token splits are unavailable.</p>
    <h4 className="mt-5 font-semibold">Observed evidence · retained stage order</h4>
    <ol className="mt-3 space-y-4">{question.stages.map((s, i) => <li key={s.node} className="border border-zinc-800 p-4">
      <h5 className="font-semibold">{i + 1}. {names[s.node] ?? s.node}</h5><p className="text-xs text-zinc-400 mt-1 break-all">{s.provenance} · {s.seconds == null ? 'Time unavailable' : `${s.seconds.toFixed(3)} s`}</p>
      {s.snapshot ? <>
        {s.node === 'initial' ? <p className="mt-3 whitespace-pre-wrap">{s.snapshot.question}</p> : (s.node === 'reformulate_query' || s.node === 'bypass_reformulation') ? <p className="mt-3 whitespace-pre-wrap">{s.snapshot.standalone_query ?? 'Unavailable'}</p> : s.node === 'grade_retrieval' ? <p className="mt-3">Grade: {s.snapshot.raw_grade === '' ? 'Empty retained output (0 characters)' : s.snapshot.raw_grade ?? 'Unavailable'} · snapshot route {s.snapshot.route}</p> : (s.node === 'generate_grounded_response' || s.node === 'inability_fallback') ? <TextEvidence label={s.node === 'generate_grounded_response' ? 'Actual generated answer' : 'Actual refusal'} text={s.snapshot.response_text} /> : <p className="mt-3">Ordered retrieval retained below</p>}
        <TextEvidence label="Declared / effective scope and history binding" text={`Declared: ${s.snapshot.declared_scope.join(', ')}\nEffective: ${s.snapshot.effective_scope.join(', ')}\nHistory SHA256: ${s.snapshot.history_sha256} (raw history unavailable)`} />
        <Chunks label="Ordered retrieved chunks" chunks={s.snapshot.retrieved_chunks} />
        <Chunks label="Ordered supplied chunks" chunks={s.snapshot.supplied_chunks} />
        <TextEvidence label="Exact rendered context" text={s.snapshot.rendered_context} />
        <TextEvidence label="Retained citations → chunk identity" text={s.snapshot.citations === null ? null : JSON.stringify(s.snapshot.citations, null, 2)} />
        <TextEvidence label="Usage request references (SHA256)" text={JSON.stringify(s.snapshot.request_sha256, null, 2)} />
      </> : <p className="mt-3 text-amber-300">Stage snapshot unavailable; no evidence reconstructed.</p>}
    </li>)}</ol>
    <details className="mt-4"><summary className="cursor-pointer focus-visible:outline focus-visible:outline-amber-500">Question usage ledger · calls.jsonl</summary><ul className="mt-3 space-y-2 text-xs">{question.calls.map(c => <li key={`${c.reservation.stage}-${c.reservation.call_id}`} className="break-all">{c.reservation.stage} #{c.reservation.call_id} attempt {c.reservation.attempt} · {c.state} · {c.outcome?.source ?? 'usage unavailable'} · input {value(c.outcome?.input_tokens)} / output {value(c.outcome?.output_tokens)} / total {value(c.outcome?.total_tokens)}<p>Request {c.reservation.payload_sha256} · provider request {c.outcome?.provider_request_sha256 ?? 'Unavailable'} · reservation {value(c.reservation.reserved_tokens)} (not consumption)</p></li>)}</ul></details>
    <p className="mt-4 text-xs text-zinc-400">Shared ingestion excluded from question expenditure. Exact wire captures, wider retrieval candidates and streamed TTFT unavailable.</p>
  </article>;
}
export default function QuestionInvestigation({ run }: { run: InvestigationRun }) {
  const [selected, setSelected] = useState('rgb-06-table3-row-extraction-gap');
  const question = run.questions.find(q => q.case_id === selected) ?? run.questions[0];
  return <section aria-labelledby="investigation" className="py-5 border-y border-zinc-800"><h2 id="investigation" className="text-lg font-semibold">Investigate a question</h2><label className="block mt-3">Question<select className="block mt-2 bg-zinc-900 border border-zinc-700 p-2 w-full text-sm" value={question?.case_id ?? ''} onChange={e => setSelected(e.target.value)}>{run.questions.map(q => <option key={q.case_id} value={q.case_id}>{q.case_id}</option>)}</select></label>{question ? <Timeline key={question.case_id} question={question} run={run} /> : <p className="mt-3">No question snapshots available</p>}</section>;
}


