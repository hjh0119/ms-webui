import { useLayoutEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Titlebar } from './Titlebar';
import { TipsBar } from './TipsBar';
import { CONTENT_H_VAR, contentHeightValue } from './metrics';
import { useSettings } from '@/settings/SettingsContext';
import { DECORATIVE_TOKENS, resolvePalette, type Palette } from '@/theme/palettes';

/**
 * 应用外壳：标题栏、侧栏、内容区与底部 tips 条共用当前模式的装饰配色。
 * workflow 深色画布保持独立，不随外壳配色改变。
 *
 * 原来外面还套了一层深色桌面、窗口带圆角和大阴影，现在去掉了，
 * 理由写在 metrics.ts 的 WINDOW_PAD 上。
 */
export function AppLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const { settings, tipsDismissed } = useSettings();

  /* 明暗类与装饰变量在绘制前一起应用；system 监听随最新配色更新。 */
  useLayoutEffect(() => {
    const root = document.documentElement;
    const originalDark = root.classList.contains('dark');
    const original = DECORATIVE_TOKENS.map((token) => ({
      token,
      value: root.style.getPropertyValue(`--${token}`),
      priority: root.style.getPropertyPriority(`--${token}`),
    }));
    let appliedDark = originalDark;
    let applied: Palette;
    const apply = (dark: boolean) => {
      appliedDark = dark;
      root.classList.toggle('dark', dark);
      const mode = dark ? 'dark' : 'light';
      applied = resolvePalette(mode, settings.colorSchemes[mode]);
      for (const token of DECORATIVE_TOKENS) root.style.setProperty(`--${token}`, applied[token]);
    };
    const mq = settings.theme === 'system' ? window.matchMedia('(prefers-color-scheme: dark)') : null;
    apply(mq ? mq.matches : settings.theme === 'dark');
    const onChange = (e: MediaQueryListEvent) => apply(e.matches);
    mq?.addEventListener('change', onChange);
    return () => {
      mq?.removeEventListener('change', onChange);
      /* 只恢复仍属于本功能的写入，保留其他代码后来修改的属性。 */
      for (const { token, value, priority } of original) {
        if (root.style.getPropertyValue(`--${token}`) !== applied[token]
          || root.style.getPropertyPriority(`--${token}`) !== '') continue;
        if (value) root.style.setProperty(`--${token}`, value, priority);
        else root.style.removeProperty(`--${token}`);
      }
      if (root.classList.contains('dark') === appliedDark) root.classList.toggle('dark', originalDark);
    };
  }, [settings.theme, settings.colorSchemes]);

  const tipsVisible = settings.tipsEnabled && !tipsDismissed;

  return (
    <div
      className="flex h-screen flex-col overflow-hidden bg-background"
      style={
        {
          /*
           * 整屏页面（对话、编排画布）用这个变量拿可用高度。
           * tips 条能被叉掉，高度是运行时变的，只有这里知道到底减不减那 26px。
           */
          [CONTENT_H_VAR]: contentHeightValue(tipsVisible),
        } as React.CSSProperties
      }
    >
      <Titlebar collapsed={collapsed} onToggleSidebar={() => setCollapsed((v) => !v)} />
      <div className="flex min-h-0 flex-1">
        <Sidebar collapsed={collapsed} />
        <main className="min-w-0 flex-1 overflow-auto bg-background">
          <Outlet />
        </main>
      </div>
      {tipsVisible && <TipsBar />}
    </div>
  );
}
