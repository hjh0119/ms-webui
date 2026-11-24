import { useEffect, useId, useRef, useState } from 'react';
import { Undo2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useT } from '@/i18n';
import { useSettings } from '@/settings/SettingsContext';
import { PALETTE_IDS, PALETTES, normalizeHex, type ColorMode, type ColorScheme } from '@/theme/palettes';

/** 两种模式复用同一编辑器；编辑不改变当前显示模式。 */
export function ColorSchemeEditor({ mode }: { mode: ColorMode }) {
  const { settings, update, loaded } = useSettings();
  const { t } = useT();
  const scheme = settings.colorSchemes[mode];
  const preset = PALETTES[scheme.preset][mode];
  const title = t(mode === 'light' ? 'settings.lightColors' : 'settings.darkColors');
  const patch = (change: Partial<ColorScheme>) => update('colorSchemes', (previous) => ({
    ...previous,
    [mode]: { ...previous[mode], ...change },
  }));

  return (
    <fieldset disabled={!loaded} className="min-w-0 border-border/60 border-b px-0.5 py-3.5">
      <legend className="sr-only">{title}</legend>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="text-foreground text-[13.5px]">{title}</span>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={() => patch({ primary: null, background: null })}
        >
          <Undo2 /> {t('settings.restoreColorScheme')}
        </Button>
      </div>
      <div className="flex flex-wrap gap-2" role="group" aria-label={t('settings.colorPresets')}>
        {PALETTE_IDS.map((id) => {
          const colors = PALETTES[id][mode];
          const selected = id === scheme.preset;
          return (
            <Button
              key={id}
              type="button"
              variant="outline"
              size="sm"
              className={selected ? 'border-ring' : ''}
              aria-pressed={selected}
              onClick={() => patch({ preset: id, primary: null, background: null })}
            >
              <span
                aria-hidden="true"
                className="flex h-4 w-7 overflow-hidden rounded-sm border border-border"
              >
                <span className="w-1/2" style={{ backgroundColor: colors.background }} />
                <span className="w-1/2" style={{ backgroundColor: colors.primary }} />
              </span>
              {t(`settings.palette.${id}`)}
            </Button>
          );
        })}
      </div>
      <p className="text-muted-foreground mt-2 text-xs leading-[17px]">
        {t('settings.colorSchemeHint')}
      </p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {(['primary', 'background'] as const).map((field) => (
          <ColorField
            key={field}
            label={t(field === 'primary' ? 'settings.primaryColor' : 'settings.backgroundColor')}
            modeLabel={title}
            value={scheme[field] ?? preset[field]}
            revision={scheme}
            onChange={(value) => patch({ [field]: value })}
            onReset={() => patch({ [field]: null })}
          />
        ))}
      </div>
    </fieldset>
  );
}

/** HEX 草稿不进入全局设置；提交、取消和外部恢复分别同步有效值。 */
function ColorField({ label, modeLabel, value, revision, onChange, onReset }: {
  label: string;
  modeLabel: string;
  value: string;
  revision: ColorScheme;
  onChange: (value: string) => void;
  onReset: () => void;
}) {
  const { t } = useT();
  const id = useId();
  const [draft, setDraft] = useState(value);
  const [invalid, setInvalid] = useState(false);
  const draftRef = useRef(value);
  const descriptionId = `${id}-error`;
  const fullLabel = `${modeLabel} · ${label}`;
  const syncDraft = (next: string) => {
    draftRef.current = next;
    setDraft(next);
    setInvalid(false);
  };

  useEffect(() => {
    draftRef.current = value;
    setDraft(value);
    setInvalid(false);
  }, [value, revision]);

  const commit = () => {
    const next = normalizeHex(draftRef.current);
    if (next === null) {
      setInvalid(true);
      return;
    }
    syncDraft(next);
    if (next !== value) onChange(next);
  };

  return (
    <div className="min-w-0">
      <label htmlFor={id} className="text-foreground mb-1 block text-xs">{label}</label>
      <div className="flex items-center gap-1.5">
        <Input
          type="color"
          value={value}
          aria-label={t('settings.chooseColor', { label: fullLabel })}
          className="h-8 w-10 shrink-0 cursor-pointer p-0.5 dark:bg-card [&::-webkit-color-swatch-wrapper]:p-0 [&::-webkit-color-swatch]:rounded-sm [&::-webkit-color-swatch]:border-0 [&::-moz-color-swatch]:border-0"
          onChange={(e) => {
            const next = normalizeHex(e.target.value);
            if (next !== null) {
              syncDraft(next);
              if (next !== value) onChange(next);
            }
          }}
        />
        <Input
          id={id}
          type="text"
          value={draft}
          aria-label={t('settings.colorHex', { label: fullLabel })}
          aria-invalid={invalid}
          aria-describedby={invalid ? descriptionId : undefined}
          spellCheck={false}
          autoComplete="off"
          className="h-8 min-w-0 flex-1 font-mono text-xs"
          onChange={(e) => syncDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.nativeEvent.isComposing) return;
            if (e.key === 'Enter') {
              e.preventDefault();
              commit();
            } else if (e.key === 'Escape') {
              e.preventDefault();
              syncDraft(value);
            }
          }}
        />
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label={t('settings.restoreColor', { label: fullLabel })}
          title={t('settings.restoreColor', { label: fullLabel })}
          onClick={onReset}
        >
          <Undo2 />
        </Button>
      </div>
      {invalid && (
        <p id={descriptionId} role="alert" className="text-state-failed mt-1 text-xs">
          {t('settings.invalidColor')}
        </p>
      )}
    </div>
  );
}
