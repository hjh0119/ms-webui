/** 装饰配色；状态、图表、端口和深色画布不进入此映射。 */
export const PALETTE_IDS = ['ice', 'warm', 'teal', 'violet', 'rose'] as const;
export type PaletteId = (typeof PALETTE_IDS)[number];
export type ColorMode = 'light' | 'dark';
export interface ColorScheme {
  preset: PaletteId;
  primary: string | null;
  background: string | null;
}
export type ColorSchemes = Record<ColorMode, ColorScheme>;

export const DECORATIVE_TOKENS = [
  'background', 'foreground', 'card', 'card-foreground', 'popover', 'popover-foreground',
  'primary', 'primary-foreground', 'secondary', 'secondary-foreground',
  'muted', 'muted-foreground', 'accent', 'accent-foreground', 'border', 'input', 'ring',
  'sidebar', 'sidebar-foreground', 'sidebar-primary', 'sidebar-primary-foreground',
  'sidebar-accent', 'sidebar-accent-foreground', 'sidebar-border', 'sidebar-ring',
] as const;
export type Palette = Record<(typeof DECORATIVE_TOKENS)[number], string>;

/** 默认值每次独立创建，浅深两组不会共享可变对象。 */
export function defaultColorSchemes(): ColorSchemes {
  return {
    light: { preset: 'ice', primary: null, background: null },
    dark: { preset: 'ice', primary: null, background: null },
  };
}

/** 只接受不透明 HEX；短格式展开，非法输入不能进入 CSS。 */
export function normalizeHex(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const hex = value.trim().toLowerCase();
  if (!/^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/.test(hex)) return null;
  return hex.length === 4 ? `#${[...hex.slice(1)].map((c) => c + c).join('')}` : hex;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

/** 旧配置和手写配置逐字段回退，保留另一模式及同组的有效字段。 */
export function normalizeColorSchemes(value: unknown): ColorSchemes {
  const saved = record(value);
  const read = (mode: ColorMode): ColorScheme => {
    const scheme = record(saved[mode]);
    return {
      preset: PALETTE_IDS.find((id) => id === scheme.preset) ?? 'ice',
      primary: normalizeHex(scheme.primary),
      background: normalizeHex(scheme.background),
    };
  };
  return { light: read('light'), dark: read('dark') };
}

type Rgb = [number, number, number];
const BLACK = '#000000';
const WHITE = '#ffffff';

function rgb(hex: string): Rgb {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as Rgb;
}

function mix(from: string, to: string, amount: number): string {
  const start = rgb(from);
  return `#${rgb(to).map((c, i) => Math.round(start[i] + (c - start[i]) * amount)
    .toString(16).padStart(2, '0')).join('')}`;
}

/** WCAG 的 sRGB 相对亮度，不能直接对 8 位 RGB 做加权。 */
function luminance(hex: string): number {
  const [r, g, b] = rgb(hex).map((c) => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const x = luminance(a);
  const y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

function contrastEndpoint(backgrounds: string[]): string {
  const minimum = (color: string) => Math.min(...backgrounds.map((bg) => contrast(color, bg)));
  return minimum(BLACK) >= minimum(WHITE) ? BLACK : WHITE;
}

/** 最大程度柔化文字，但每次按最终取整后的 HEX 检查 4.5:1。 */
function readableText(background: string): string {
  const endpoint = contrastEndpoint([background]);
  let passing = endpoint;
  let low = 0;
  let high = 1;
  for (let i = 0; i < 24; i++) {
    const amount = (low + high) / 2;
    const candidate = mix(endpoint, background, amount);
    if (contrast(candidate, background) >= 4.5) {
      passing = candidate;
      low = amount;
    } else high = amount;
  }
  return passing;
}

/** 焦点色只向黑或白调整，针对实际相邻表面而非主色按钮内底。 */
function focusColor(primary: string, backgrounds: string[]): string {
  const passes = (color: string) => backgrounds.every((bg) => contrast(color, bg) >= 3);
  if (passes(primary)) return primary;
  const endpoint = contrastEndpoint(backgrounds);
  let passing = endpoint;
  let low = 0;
  let high = 1;
  for (let i = 0; i < 24; i++) {
    const amount = (low + high) / 2;
    const candidate = mix(primary, endpoint, amount);
    if (passes(candidate)) {
      passing = candidate;
      high = amount;
    } else low = amount;
  }
  return passing;
}

/** 冰蓝与 index.css 的启动回退逐项一致；不要用生成器近似当前外观。 */
const ICE: Record<ColorMode, Palette> = {
  light: {
    background: '#f6f7f9', foreground: '#252a36', card: '#ffffff', 'card-foreground': '#252a36',
    popover: '#ffffff', 'popover-foreground': '#252a36', primary: '#3e4773', 'primary-foreground': '#f8f5f0',
    secondary: '#edf0f8', 'secondary-foreground': '#353e5b', muted: '#eef1f5', 'muted-foreground': '#657084',
    accent: '#e3e8f3', 'accent-foreground': '#353e5b', border: '#dde2ea', input: '#cbd3df', ring: '#4063a1',
    sidebar: '#eef1f5', 'sidebar-foreground': '#252a36', 'sidebar-primary': '#3e4773',
    'sidebar-primary-foreground': '#f8f5f0', 'sidebar-accent': '#e3e8f3', 'sidebar-accent-foreground': '#353e5b',
    'sidebar-border': '#dde2ea', 'sidebar-ring': '#4063a1',
  },
  dark: {
    background: '#141720', foreground: '#e8edf6', card: '#1b202c', 'card-foreground': '#e8edf6',
    popover: '#222938', 'popover-foreground': '#e8edf6', primary: '#9aa5db', 'primary-foreground': '#16141f',
    secondary: '#252e42', 'secondary-foreground': '#e8edf6', muted: '#202735', 'muted-foreground': '#a5afc1',
    accent: '#2c364a', 'accent-foreground': '#f0f4fa', border: '#30394a', input: '#414d63', ring: '#6b78ad',
    sidebar: '#181d28', 'sidebar-foreground': '#e8edf6', 'sidebar-primary': '#b7c0e6',
    'sidebar-primary-foreground': '#1e1b2e', 'sidebar-accent': '#263146', 'sidebar-accent-foreground': '#f0f4fa',
    'sidebar-border': '#30394a', 'sidebar-ring': '#6b78ad',
  },
};

/** 暖黄保留此前纸色和蓝紫强调色，不恢复旧布局或旧阴影。 */
const WARM: Record<ColorMode, Palette> = {
  light: {
    ...ICE.light,
    background: '#f8f5f0', foreground: '#2b2438', card: '#fdfbf7', 'card-foreground': '#2b2438',
    popover: '#fffdfa', 'popover-foreground': '#2b2438',
    secondary: '#ebe7f1', 'secondary-foreground': '#35305c', muted: '#f0ebe4', 'muted-foreground': '#655f78',
    accent: '#ddd6ea', 'accent-foreground': '#35305c', border: '#e0d6c9', input: '#e0d6c9',
    sidebar: '#f0ebe4', 'sidebar-foreground': '#2b2438', 'sidebar-accent': '#ddd6ea',
    'sidebar-accent-foreground': '#35305c', 'sidebar-border': '#e0d6c9',
  },
  dark: {
    ...ICE.dark,
    background: '#16141f', foreground: '#ece9f2', card: '#1d1a29', 'card-foreground': '#ece9f2',
    popover: '#221f31', 'popover-foreground': '#ece9f2', secondary: '#2a2740', 'secondary-foreground': '#ece9f2',
    muted: '#221f2e', 'muted-foreground': '#a49eb8', accent: '#322d47', 'accent-foreground': '#f4f2f8',
    border: '#302b42', input: '#3b3554', sidebar: '#1a1728', 'sidebar-foreground': '#ece9f2',
    'sidebar-accent': '#2b2740', 'sidebar-accent-foreground': '#f4f2f8', 'sidebar-border': '#322d47',
  },
};

interface PresetSurfaces {
  background: string;
  card: string;
  popover: string;
  sidebar: string;
  primary: string;
  foreground: string;
  mutedForeground: string;
}

/** 新主题分别指定浅深表面，选择态与焦点色复用同一生成逻辑。 */
function presetPalette(mode: ColorMode, surfaces: PresetSurfaces): Palette {
  const { mutedForeground, ...tokens } = surfaces;
  const p: Palette = {
    ...ICE[mode], ...tokens,
    'card-foreground': surfaces.foreground,
    'popover-foreground': surfaces.foreground,
    'sidebar-foreground': surfaces.foreground,
    muted: mix(surfaces.background, surfaces.foreground, 0.06),
    'muted-foreground': mutedForeground,
    'sidebar-primary': surfaces.primary,
    border: mix(surfaces.background, surfaces.foreground, 0.16),
    input: mix(surfaces.card, surfaces.foreground, 0.28),
    'sidebar-border': mix(surfaces.sidebar, surfaces.foreground, 0.16),
  };
  applyEmphasis(p);
  return p;
}

export const PALETTES: Readonly<Record<PaletteId, Readonly<Record<ColorMode, Readonly<Palette>>>>> = {
  ice: ICE,
  warm: WARM,
  teal: {
    light: presetPalette('light', {
      background: '#f1f7f5', card: '#ffffff', popover: '#ffffff', sidebar: '#e8f1ed',
      primary: '#23695f', foreground: '#243c37', mutedForeground: '#5d756e',
    }),
    dark: presetPalette('dark', {
      background: '#111d1a', card: '#182823', popover: '#21352d', sidebar: '#15231e',
      primary: '#8bc9b6', foreground: '#e5f1eb', mutedForeground: '#a2baaf',
    }),
  },
  violet: {
    light: presetPalette('light', {
      background: '#f5f3fa', card: '#ffffff', popover: '#ffffff', sidebar: '#eeebf5',
      primary: '#6d5193', foreground: '#302b3e', mutedForeground: '#736984',
    }),
    dark: presetPalette('dark', {
      background: '#1b1724', card: '#252030', popover: '#30293c', sidebar: '#211c2c',
      primary: '#bea6de', foreground: '#efe9f8', mutedForeground: '#b6a7c8',
    }),
  },
  rose: {
    light: presetPalette('light', {
      background: '#fbf3f5', card: '#ffffff', popover: '#ffffff', sidebar: '#f4e9ed',
      primary: '#af556c', foreground: '#432c35', mutedForeground: '#876a76',
    }),
    dark: presetPalette('dark', {
      background: '#21171d', card: '#2d2028', popover: '#392932', sidebar: '#271c23',
      primary: '#e8a2b8', foreground: '#f6e8ee', mutedForeground: '#c4a9b5',
    }),
  },
};

function applyEmphasis(p: Palette, primaryChanged = true): void {
  if (primaryChanged) {
    p['primary-foreground'] = readableText(p.primary);
    p['sidebar-primary-foreground'] = readableText(p['sidebar-primary']);
  }
  p.secondary = mix(p.background, p.primary, 0.12);
  p['secondary-foreground'] = readableText(p.secondary);
  p.accent = mix(p.background, p.primary, 0.18);
  p['accent-foreground'] = readableText(p.accent);
  p['sidebar-accent'] = mix(p.sidebar, p['sidebar-primary'], 0.18);
  p['sidebar-accent-foreground'] = readableText(p['sidebar-accent']);
  p.ring = focusColor(p.primary, [p.background, p.card, p.popover, p.muted, p.secondary, p.accent]);
  p['sidebar-ring'] = focusColor(p['sidebar-primary'], [p.sidebar, p['sidebar-accent']]);
}

/** 返回完整新映射；修改、恢复和切换都不会改变预设或留下旧变量。 */
export function resolvePalette(mode: ColorMode, scheme: ColorScheme): Palette {
  const p = { ...PALETTES[scheme.preset][mode] };
  if (scheme.background !== null) {
    p.background = scheme.background;
    const direction = contrastEndpoint([p.background]);
    p.foreground = readableText(p.background);
    p.card = mix(p.background, direction, 0.04);
    p['card-foreground'] = readableText(p.card);
    p.popover = mix(p.background, direction, 0.08);
    p['popover-foreground'] = readableText(p.popover);
    p.muted = mix(p.background, direction, 0.06);
    p['muted-foreground'] = readableText(p.muted);
    p.sidebar = p.muted;
    p['sidebar-foreground'] = readableText(p.sidebar);
    p.border = mix(p.background, p.foreground, 0.16);
    p.input = mix(p.card, p['card-foreground'], 0.28);
    p['sidebar-border'] = mix(p.sidebar, p['sidebar-foreground'], 0.16);
  }
  if (scheme.primary !== null) {
    p.primary = scheme.primary;
    p['sidebar-primary'] = scheme.primary;
  }
  if (scheme.primary !== null || scheme.background !== null) applyEmphasis(p, scheme.primary !== null);
  return p;
}
