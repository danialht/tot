import React from 'react';
import katex from 'katex';

type MathTextProps = {
  text: string;
  inline?: boolean;
  errorColor?: string;
};

// Very small parser: splits into text and math runs for $...$ (inline) and $$...$$ (block)
function splitIntoRuns(input: string): Array<{ type: 'text' | 'math'; content: string; displayMode: boolean }>{
  const runs: Array<{ type: 'text' | 'math'; content: string; displayMode: boolean }> = [];
  let i = 0;
  let buffer = '';
  while (i < input.length) {
    if (input[i] === '$') {
      const isBlock = i + 1 < input.length && input[i + 1] === '$';
      const delimiter = isBlock ? '$$' : '$';
      const start = i + delimiter.length;
      const end = input.indexOf(delimiter, start);
      if (end !== -1) {
        if (buffer) {
          runs.push({ type: 'text', content: buffer, displayMode: false });
          buffer = '';
        }
        const mathContent = input.slice(start, end);
        runs.push({ type: 'math', content: mathContent, displayMode: isBlock });
        i = end + delimiter.length;
        continue;
      }
    }
    buffer += input[i];
    i += 1;
  }
  if (buffer) runs.push({ type: 'text', content: buffer, displayMode: false });
  return runs;
}

const MathText: React.FC<MathTextProps> = ({ text, inline = true, errorColor = '#e57373' }) => {
  const runs = splitIntoRuns(text ?? '');
  return (
    <>
      {runs.map((run, idx) => {
        if (run.type === 'text') {
          return <React.Fragment key={idx}>{run.content}</React.Fragment>;
        }
        try {
          const html = katex.renderToString(run.content, {
            displayMode: run.displayMode && !inline,
            throwOnError: false,
            errorColor,
            trust: false,
            output: 'html',
          });
          return (
            <span
              key={idx}
              aria-label="math"
              dangerouslySetInnerHTML={{ __html: html }}
              style={run.displayMode && !inline ? { display: 'block', margin: '8px 0' } : undefined}
            />
          );
        } catch (e) {
          return (
            <code key={idx} style={{ color: errorColor }}>
              {run.content}
            </code>
          );
        }
      })}
    </>
  );
};

export default MathText;


