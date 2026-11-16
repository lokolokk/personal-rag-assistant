import { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

/** Only KaTeX writes markup here; message HTML is never accepted. */
export function MathExpression({ source, display = false }: {
  source: string;
  display?: boolean;
}): React.JSX.Element {
  const target = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!target.current) return;
    if (source.length > 10000) {
      target.current.textContent = source;
      return;
    }
    try {
      katex.render(source, target.current, {
        displayMode: display,
        output: 'htmlAndMathml',
        trust: false,
        strict: 'error',
        throwOnError: true,
        maxSize: 10,
        maxExpand: 1000,
        macros: {},
      });
    } catch {
      // Incomplete/unsupported expressions remain readable during streaming.
      target.current.textContent = source;
    }
  }, [source, display]);
  return <span ref={target} className={display ? 'block max-w-full overflow-x-auto my-2' : ''} />;
}
