import { memo, useEffect, useRef } from 'react';
import { contentHeight } from '@/layout/metrics';

/**
 * 日志视图。真实实现是按字节 offset 增量拉 output.log（SSE），
 * 这里只渲染给定的行数组，滚动与着色逻辑先定下来。
 *
 * 着色改成取 --state-* 和 --chart-*：原来四个色值是写死的十六进制，
 * 深色模式下 ERROR 那个红会发暗、行号那个 #c9d1d9 会白得刺眼。
 * 走变量之后深浅两套自动成立。
 */
export function LogViewer({
  lines,
  follow = true,
  /* 192px 是详情页自己的页头和 Tab 条；底部 tips 条开关时高度自动跟着变 */
  height = contentHeight(192),
}: {
  lines: string[];
  follow?: boolean;
  height?: string | number;
}) {
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (follow && boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight;
    }
  }, [lines, follow]);

  return (
    <div
      ref={boxRef}
      className="bg-muted border-border overflow-auto border py-3 font-mono text-[12.5px] leading-5"
      style={{ height }}
    >
      {lines.map((line, i) => <LogLine key={i} line={line} number={i + 1} />)}
    </div>
  );
}

// 追加日志不重新着色历史行；主题仍由 CSS 变量实时切换。
const LogLine = memo(function LogLine({ line, number }: { line: string; number: number }) {
  return (
    <div className="flex px-3.5 whitespace-pre-wrap break-all">
      <span className="text-muted-foreground/60 w-10 flex-none pr-3.5 text-right select-none tabular">{number}</span>
      <span style={{ color: colorOf(line) }}>{line}</span>
    </div>
  );
});

/** ERROR 先于 WARN，时间戳最后判断；workflow 节段仍优先突出。 */
function colorOf(line: string): string {
  if (line.startsWith('[workflow]')) return 'var(--chart-2)';
  if (/\b(ERROR|Traceback|CUDA out of memory|FAILED)\b/.test(line) || line.includes('运行失败')) return 'var(--state-failed)';
  if (/\bWARN(ING)?\b/.test(line)) return 'var(--state-paused)';
  if (line.startsWith('{')) return 'var(--chart-3)';
  if (/^\[\d{4}-/.test(line) || line.includes('run.sh')) return 'var(--muted-foreground)';
  return 'var(--foreground)';
}
