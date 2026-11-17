import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, ArrowRight, BookOpen, Monitor, Moon, PanelLeftClose, PanelLeftOpen, Sun } from 'lucide-react';
import { ChromeIconButton } from './ChromeIconButton';
import { TITLEBAR_H } from './metrics';
import { moduleOf } from '@/theme/modules';
import { useSettings } from '@/settings/SettingsContext';
import { useT, type MessageKey } from '@/i18n';

/**
 * 窗口标题栏：侧栏开合 + 前进后退在左，当前位置居中，全局链接贴右。
 *
 * 原来左边有一组 macOS 交通灯，配合「浮在深色桌面上的圆角窗口」才成立。
 * 现在界面铺满视口，那三个点点了也不会关窗口，纯装饰且跟工具型界面的克制冲突，去掉了。
 *
 * 刻意不做面包屑，也不放居中的大搜索条——搜索入口在侧栏。
 */

export function Titlebar({
  collapsed,
  onToggleSidebar,
}: {
  collapsed: boolean;
  onToggleSidebar: () => void;
}) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const current = moduleOf(pathname);
  const { t } = useT();
  const { settings, update } = useSettings();
  /*
   * 独立于模块的路由（如 /settings）不隶属任何模块：moduleOf 会回落到 chat，
   * 直接拿 chat 的前缀去切 pathname 会截出「ings」这种碎片。没真正命中模块时
   * 就不切前缀、也不显示模块名，只把路径段交给 tail 翻译。
   */
  const inModule = pathname === current.path || pathname.startsWith(current.path + '/');
  const rest = (inModule ? pathname.slice(current.path.length) : pathname).split('/').filter(Boolean);

  const tail: Record<string, MessageKey> = {
    new: 'titlebar.pageNew',
    log: 'titlebar.pageLog',
    metrics: 'titlebar.pageMetrics',
    result: 'titlebar.pageResult',
    edit: 'titlebar.pageEdit',
    settings: 'titlebar.pageSettings',
  };
  const sub = rest.map((s) => (tail[s] ? t(tail[s]) : s)).join(' · ');

  /* 主题三态循环：浅色 → 深色 → 跟随系统（图标即当前档，tooltip 说明） */
  const nextTheme = settings.theme === 'light' ? 'dark' : settings.theme === 'dark' ? 'system' : 'light';
  const ThemeIcon = settings.theme === 'light' ? Sun : settings.theme === 'dark' ? Moon : Monitor;

  return (
    <div
      className="bg-sidebar border-sidebar-border flex shrink-0 items-center gap-0.5 border-b px-2.5 select-none"
      style={{ height: TITLEBAR_H }}
    >
      <ChromeIconButton
        icon={collapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
        label={collapsed ? t('titlebar.expandSidebar') : t('titlebar.collapseSidebar')}
        onClick={onToggleSidebar}
      />
      <ChromeIconButton icon={<ArrowLeft size={15} />} label={t('titlebar.back')} onClick={() => navigate(-1)} />
      <ChromeIconButton icon={<ArrowRight size={15} />} label={t('titlebar.forward')} onClick={() => navigate(1)} />

      <div className="flex min-w-0 flex-1 items-center justify-center gap-2">
        {inModule && <span className="text-sidebar-foreground/60 text-xs whitespace-nowrap">{t(current.label)}</span>}
        {sub && (
          <span className="text-sidebar-foreground/40 text-xs whitespace-nowrap">{sub}</span>
        )}
      </div>

      <ChromeIconButton
        icon={<ThemeIcon size={14} />}
        label={t(`theme.${settings.theme}` as MessageKey)}
        onClick={() => update('theme', nextTheme)}
      />
      <ChromeIconButton
        icon={<BookOpen size={14} />}
        label={t('common.docs')}
        href="https://swift.readthedocs.io"
      />
    </div>
  );
}
