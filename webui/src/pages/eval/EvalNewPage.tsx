import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { FlaskConical } from 'lucide-react';
import { toast } from 'sonner';
import { ConfigFormShell } from '@/components/ConfigFormShell';
import type { ConfigField } from '@/components/PageAssistant';
import { Field, FieldStack, NumberInput, SegmentedControl, SelectInput } from '@/components/FormField';
import { SearchSelect, TagInput, type SearchOption } from '@/components/SearchSelect';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { MODULES } from '@/theme/modules';
import { listModels, type ModelConfig } from '@/api/models';
import { listCatalog, modelOption, listCkptCatalog, ckptOption, ckptPick, type CatalogEntry, type CkptEntry } from '@/api/catalog';
import { listPlugins } from '@/api/plugins';
import {
  listEvalDatasets,
  listEvals,
  startEval,
  type EvalDatasetInfo,
  type EvalItem,
  type EvalLocalTarget,
} from '@/api/eval';
import { pushRecent } from '@/store/recent';
import { prefillIntro, usePrefill } from '@/store/prefill';
import { useT } from '@/i18n';

/** 已选评测集在侧栏的摘要：带 limit 的标条数，否则标全量 */
function itemSummary(items: EvalItem[]): string {
  return items
    .map((it) => (it.limit ? `${it.dataset}(${it.limit})` : it.dataset))
    .join(' ');
}

/* 存档任务（datasetArgs 形态）→ 表单项（subsets 形态）的回显归一 */
function fromStoredItem(it: EvalItem): EvalItem {
  const inner = (it.datasetArgs as Record<string, { subsets?: unknown }> | undefined)?.[it.dataset];
  const subs = inner && Array.isArray(inner.subsets) ? inner.subsets.map(String) : undefined;
  return {
    dataset: it.dataset,
    ...(it.limit ? { limit: it.limit } : {}),
    ...(subs?.length ? { subsets: subs } : {}),
  };
}

/**
 * 新建评测。被测模型只能选模型配置（models.json）里的条目——评测走 swift eval
 * 的 api 模式（--eval_url 指向该条目的端点），本地不加载任何模型。评测集来自
 * EvalScope 的数据集注册表（可搜索）；选中即展开首段描述与子集勾选（部分勾选
 * 经 --eval_dataset_args 下传），每集可独立设「只评前 N 条」。
 */
export function EvalNewPage() {
  const { t } = useT();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const prefill = usePrefill('eval');
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [datasets, setDatasets] = useState<EvalDatasetInfo[]>([]);
  const [modelId, setModelId] = useState('');
  /** 搜索关键词过滤 290 个基准 */
  const [query, setQuery] = useState('');
  /** 选中项的 limit（key = dataset name；undefined = 全量） */
  const [items, setItems] = useState<EvalItem[]>([]);
  const [plugins, setPlugins] = useState<string[]>([]);
  const [extraArgs, setExtraArgs] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [pluginOptions, setPluginOptions] = useState<SearchOption[]>([]);
  /*
   * 本地直评（autoresearch 的训后 checkpoint 评测主用；这里是最小手填入口）：
   * 底座 + adapters 直接评，不连端点。与 api 模式互斥，评测集选择共用。
   */
  const [localMode, setLocalMode] = useState(false);
  const [baseModel, setBaseModel] = useState('');
  const [adapters, setAdapters] = useState('');
  const [gpus, setGpus] = useState('');
  /* 本地直评底座候选：注册表模型 + 本机训练 ckpt；ckpt 另存 map 供选中联动 */
  const [baseOptions, setBaseOptions] = useState<SearchOption[]>([]);
  const [ckptMap, setCkptMap] = useState<Record<string, CkptEntry>>({});
  /* 选中 ckpt 的回显名（「底座:断点名」，与部署页同病同修）：底座字段落到的是
   * 底座名，回显 value 分不出选的是哪个断点——SearchSelect 的 display 覆盖 */
  const [pickedCkpt, setPickedCkpt] = useState('');

  useEffect(() => {
    listModels()
      .then((configs) => {
        setModels(configs);
        setModelId((prev) => prev || configs[0]?.id || '');
      })
      .catch((e) => toast.error(String(e)));
    /*
     * 本地直评底座候选一次装配：注册表模型与本机 ckpt 两路到齐拼接，ckpt 排前
     * （空关键词截前 50）；先行各自 set 会互相覆盖。失败退空数组不拦另一路。
     */
    void Promise.all([
      listCatalog().then((c) => c.models).catch(() => [] as CatalogEntry[]),
      listCkptCatalog(),
    ]).then(([models, ckpts]) => {
      setBaseOptions([...ckpts.map(ckptOption), ...models.map(modelOption)]);
      setCkptMap(Object.fromEntries(ckpts.map((c) => [c.path, c])));
    });
    listEvalDatasets()
      .then(setDatasets)
      .catch((e) => toast.error(String(e)));
    listPlugins()
      .then((ps) => setPluginOptions(ps.map((p) => ({ value: p.source, label: p.id, hint: p.note }))))
      .catch(() => {
        /* 清单不可达：TagInput 仍接受自由输入 */
      });
  }, []);

  /*
   * 底座选中入口：命中本机训练 ckpt 时联动填表——full 断点 baseModel 直接填
   * 路径；LoRA 断点 baseModel 填底座、adapters 填路径。后改不自动清 adapters。
   */
  const pickBaseModel = (v: string) => {
    const hit = ckptPick(ckptMap, v);
    if (hit) {
      setBaseModel(hit.model);
      setPickedCkpt(ckptOption(ckptMap[v]).label ?? '');
      if (hit.adapters) setAdapters(hit.adapters);
      return;
    }
    setPickedCkpt('');
    setBaseModel(v);
  };

  /*
   * 「以此为模板新建」入口：?from=<任务 id> 预填被测模型与评测集（含每集的
   * limit）。任务不存在（webui 重启后内存表清空）时静默退化为普通新建；
   * listModels 的默认选中不会覆盖这里设的 modelId（它的 setModelId 只在
   * prev 为空时生效）。
   */
  useEffect(() => {
    const from = searchParams.get('from');
    /* AI 预填与模板克隆是两个互斥入口：带了预填就不回填模板任务 */
    if (!from || prefill) return;
    listEvals()
      .then((tasks) => {
        const src = tasks.find((x) => x.id === from);
        if (!src) return;
        setModelId(src.modelId);
        /* 兼容旧任务的 datasetArgs 形态：反解出 subsets 供表单回显 */
        setItems(src.items.map((it) => ({ ...fromStoredItem(it) })));
        setPlugins(src.plugins ?? []);
        setExtraArgs(src.extraArgs ?? '');
      })
      .catch(() => {
        /* 后端暂不可达：保持空表单 */
      });
  }, [searchParams]);

  /*
   * 首页 AI 创建工具带来的预填：mount 时读取（读取即清），字段缺失静默忽略。
   * 评测集项带可选 limit；数据集名不在注册表里的照收——页面提交前自会校验。
   */
  useEffect(() => {
    if (!prefill) return;
    const f = prefill.fields;
    if (typeof f.modelId === 'string' && f.modelId.trim()) setModelId(f.modelId.trim());
    if (Array.isArray(f.items)) {
      setItems(
        f.items
          .map((it) => {
            const d = String((it as Record<string, unknown>).dataset ?? '').trim();
            const limit = Number((it as Record<string, unknown>).limit);
            const subs = (it as Record<string, unknown>).subsets;
            return d
              ? {
                  dataset: d,
                  ...(Number.isInteger(limit) && limit > 0 ? { limit } : {}),
                  ...(Array.isArray(subs) && subs.length ? { subsets: subs.map(String) } : {}),
                }
              : null;
          })
          .filter((it): it is EvalItem => !!it),
      );
    }
    if (Array.isArray(f.plugins)) setPlugins(f.plugins.map(String).filter(Boolean));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return datasets;
    return datasets.filter(
      (d) =>
        d.name.toLowerCase().includes(q) ||
        d.prettyName.toLowerCase().includes(q) ||
        d.tags.some((x) => x.toLowerCase().includes(q)),
    );
  }, [datasets, query]);

  const toggle = (name: string) =>
    setItems((prev) =>
      prev.some((it) => it.dataset === name)
        ? prev.filter((it) => it.dataset !== name)
        : [...prev, { dataset: name }],
    );

  const setLimit = (name: string, limit: number | undefined) =>
    setItems((prev) =>
      prev.map((it) => (it.dataset === name ? { ...it, limit: limit && limit > 0 ? limit : undefined } : it)),
    );

  /*
   * 子集勾选：item.subsets 未设 = 全选（整集评测）；点选后存选中清单，回到全集
   * 时归一化成未设（不传 datasetArgs）。至少保留一个——取消最后一个被忽略。
   */
  const toggleSubset = (name: string, sub: string) => {
    const all = datasets.find((d) => d.name === name)?.subsets ?? [];
    setItems((prev) =>
      prev.map((it) => {
        if (it.dataset !== name) return it;
        const current = it.subsets ?? all;
        const next = current.includes(sub) ? current.filter((s) => s !== sub) : [...current, sub];
        if (!next.length) return it;
        return { ...it, subsets: next.length === all.length ? undefined : next };
      }),
    );
  };

  const selectAllSubsets = (name: string) =>
    setItems((prev) => prev.map((it) => (it.dataset === name ? { ...it, subsets: undefined } : it)));

  /*
   * 部分勾选子集时的 dataset_args 形态（EvalScope 的 {dataset: {subsets: [...]}}）。
   * 全选不传——整集评测与显式列全部子集在 EvalScope 语义上等价，省一个参数。
   */
  const argsOf = (it: EvalItem): Record<string, unknown> | undefined => {
    const all = datasets.find((d) => d.name === it.dataset)?.subsets;
    if (it.subsets?.length && all && it.subsets.length < all.length) {
      return { [it.dataset]: { subsets: it.subsets } };
    }
    return undefined;
  };

  /* 提交/预览共用的逐集参数尾（limit + 子集选择） */
  const itemTail = (it: EvalItem): string => {
    const args = argsOf(it);
    return (
      (it.limit ? ` \\\n    --eval_limit ${it.limit}` : '') +
      (args ? ` \\\n    --eval_dataset_args ${JSON.stringify(args)}` : '')
    );
  };

  const model = models.find((m) => m.id === modelId);

  /* 本地直评目标：adapters/gpus 都是空白或逗号分隔的多值 */
  const localTarget: EvalLocalTarget | undefined = localMode
    ? {
        model: baseModel.trim(),
        ...(adapters.trim()
          ? { adapters: adapters.split(/[\s,]+/).map((a) => a.trim()).filter(Boolean) }
          : {}),
        ...(gpus.trim()
          ? {
              gpus: gpus
                .split(/[\s,]+/)
                .map(Number)
                .filter((n) => Number.isInteger(n) && n >= 0),
            }
          : {}),
      }
    : undefined;

  const preview = useMemo(() => {
    /* 本地直评：--model 底座 + --adapters checkpoint，无 eval_url */
    if (localMode) {
      if (!baseModel.trim() || !items.length) {
        return t('ev.cmdHintLocal');
      }
      const tail =
        (adapters.trim() ? ` \\\n    --adapters ${adapters.split(/[\s,]+/).filter(Boolean).join(' ')}` : '') +
        (gpus.trim() ? `  # CUDA_VISIBLE_DEVICES=${gpus}` : '');
      return [
        t('ev.cmdLocalNote'),
        ...items.map(
          (it) =>
            `swift eval \\\n    --model ${baseModel.trim()}${tail} \\\n    --eval_dataset ${it.dataset}` + itemTail(it),
        ),
      ].join('\n\n');
    }
    if (!model || !items.length) {
      return t('ev.cmdHintApi');
    }
    /* 插件与附加参数对每个评测集都生效 */
    const tail =
      (plugins.length ? ` \\\n    --external_plugins ${plugins.join(' ')}` : '') +
      (extraArgs.trim() ? ` \\\n    ${extraArgs.trim()}` : '');
    return [
      t('ev.cmdApiNote'),
      ...items.map(
        (it) =>
          `swift eval \\` +
          `\n    --model ${model.model} \\` +
          `\n    --eval_url ${model.baseUrl} \\` +
          `\n    --eval_dataset ${it.dataset}` + itemTail(it) + tail,
      ),
    ].join('\n\n');
  }, [localMode, baseModel, adapters, gpus, model, items, datasets, plugins, extraArgs, t]);

  const submit = async () => {
    if (localMode && !baseModel.trim()) {
      toast.error(t('ev.errBase'));
      return;
    }
    if (!localMode && (!modelId || !model)) {
      toast.error(t('ev.errModel'));
      return;
    }
    if (!items.length) {
      toast.error(t('ev.errDatasets'));
      return;
    }
    setSubmitting(true);
    try {
      const id = await startEval({
        ...(localMode ? { local: localTarget } : { modelId }),
        /* subsets 是表单态；后端只认 datasetArgs（部分勾选时才有） */
        items: items.map((it) => {
          const args = argsOf(it);
          return {
            dataset: it.dataset,
            ...(it.limit ? { limit: it.limit } : {}),
            ...(args ? { datasetArgs: args } : {}),
          };
        }),
        ...(plugins.length ? { plugins } : {}),
        ...(extraArgs.trim() ? { extraArgs: extraArgs.trim() } : {}),
      });
      pushRecent({
        id: `eval:${id}`,
        label: t('ev.taskLabel', { name: localMode ? baseModel.trim() : modelId }),
        path: `/eval/${id}/overview`,
      });
      toast.success(t('ev.started'));
      navigate('/eval');
    } catch (e) {
      toast.error(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  /* 交给 AI 助手的字段 */
  const fields: ConfigField[] = [
    {
      key: 'model',
      label: t('ev.modelUnderTest'),
      value: localMode ? baseModel : modelId,
      apply: localMode ? pickBaseModel : setModelId,
    },
    {
      key: 'eval_dataset',
      label: t('ev.datasets'),
      value: itemSummary(items),
      /* AI 能引用所选集，但增删以用户在列表里的勾选为准 */
      apply: (v) =>
        setItems(
          v
            .split(/[\s,]+/)
            .filter(Boolean)
            .map((token) => {
              const m = token.match(/^(.+)\((\d+)\)$/);
              return m
                ? { dataset: m[1], limit: Number(m[2]) }
                : { dataset: token };
            })
            .filter((it) => datasets.some((d) => d.name === it.dataset)),
        ),
    },
  ];

  return (
    <ConfigFormShell
      module={MODULES.eval}
      title={t('ev.newEval')}
      desc={t('ev.desc')}
      preview={preview}
      fields={fields}
      onSubmit={() => void submit()}
      extraArgs={extraArgs}
      onExtraArgsChange={setExtraArgs}
      warning={t('ev.warn')}
      actions={
        <Button onClick={() => void submit()} disabled={submitting}>
          <FlaskConical /> {submitting ? t('tn.launching') : t('ev.launchBtn')}
        </Button>
      }
      sections={[
        {
          title: t('ev.modelUnderTest'),
          content: (
            <FieldStack>
              <Field
                label={t('ev.method')}
                hint={t('ev.methodHint')}
              >
                <SegmentedControl
                  value={localMode ? 'local' : 'api'}
                  onChange={(v) => setLocalMode(v === 'local')}
                  options={[
                    { value: 'api', label: t('ev.optApi') },
                    { value: 'local', label: t('ev.optLocal') },
                  ]}
                />
              </Field>
              {localMode ? (
                <>
                  <Field label={t('exp.baseModel')} hint={t('ev.baseHint')}>
                    <SearchSelect
                      value={baseModel}
                      onChange={pickBaseModel}
                      options={baseOptions}
                      placeholder="Qwen/Qwen2.5-0.5B-Instruct"
                      display={pickedCkpt || undefined}
                    />
                  </Field>
                  <Field label={t('ev.adaptersLabel')} hint={t('ev.adaptersHint')}>
                    <Input
                      value={adapters}
                      onChange={(e) => setAdapters(e.target.value)}
                      placeholder="/root/.../output/v1-20250101-120000"
                      className="h-8 font-mono text-[13px]"
                    />
                  </Field>
                  <Field label={t('ev.gpusLabel')} hint={t('ev.gpusHint')}>
                    <Input
                      value={gpus}
                      onChange={(e) => setGpus(e.target.value)}
                      placeholder="0,1"
                      className="h-8 font-mono text-[13px]"
                    />
                  </Field>
                </>
              ) : (
                <Field label={t('tn.modelLabel')} hint={t('ev.modelHint')}>
                  {models.length ? (
                    <SelectInput
                      value={modelId}
                      onChange={setModelId}
                      options={models.map((m) => ({
                        value: m.id,
                        label: m.note === 'swift 部署' ? `${m.id}${t('ev.deployedSuffix')}` : m.id,
                      }))}
                    />
                  ) : (
                    <div className="text-muted-foreground text-[13px]">
                      {t('ev.noModels')}
                    </div>
                  )}
                </Field>
              )}
              <Field
                label={t('tn.pluginsLabel')}
                hint={t('tn.pluginsHint')}
              >
                <TagInput
                  values={plugins}
                  onChange={setPlugins}
                  options={pluginOptions}
                  placeholder="/path/to/plugin.py"
                />
              </Field>
            </FieldStack>
          ),
        },
        {
          title: t('ev.datasets'),
          content: (
            <FieldStack>
              <Field
                label={t('ev.pickLabel', { n: items.length })}
                hint={t('ev.pickHint')}
              >
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={t('ev.searchPh')}
                  className="h-8 text-[13px]"
                />
              </Field>
              <div className="border-border/60 max-h-96 overflow-auto rounded-lg border">
                {filtered.map((d) => {
                  const picked = items.find((it) => it.dataset === d.name);
                  /* 子集选择只在多子集时有意义（单名义子集如 default/main 不展示） */
                  const subsets = d.subsets && d.subsets.length > 1 ? d.subsets : undefined;
                  const pickedSubs = picked?.subsets ?? subsets ?? [];
                  return (
                    <div key={d.name} className="border-border/60 border-b last:border-b-0">
                      <div className="hover:bg-muted/50 flex items-center gap-3 px-3 py-2">
                        <button
                          type="button"
                          onClick={() => toggle(d.name)}
                          className="flex min-w-0 flex-1 items-center gap-2.5 text-left"
                        >
                          <span
                            className={
                              picked
                                ? 'bg-primary text-primary-foreground flex size-4 flex-none items-center justify-center rounded text-[10px]'
                                : 'border-border size-4 flex-none rounded border'
                            }
                          >
                            {picked ? '✓' : ''}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="text-foreground block truncate text-[13px]">
                              {d.name}
                              {d.prettyName !== d.name && (
                                <span className="text-muted-foreground"> · {d.prettyName}</span>
                              )}
                            </span>
                            {(d.tags.length > 0 || d.metrics.length > 0) && (
                              <span className="text-muted-foreground/70 block truncate text-[11px]">
                                {[...d.tags, ...d.metrics].slice(0, 6).join(' / ')}
                              </span>
                            )}
                          </span>
                        </button>
                        {subsets && picked && (
                          <span className="text-muted-foreground flex-none text-[11px]">
                            {t('ev.subsetsPicked', { n: picked.subsets?.length ?? subsets.length, m: subsets.length })}
                          </span>
                        )}
                      </div>
                      {picked && (
                        /* 选中即展开：描述 + 子集勾选 + 采样数量 */
                        <div className="bg-muted/30 space-y-2.5 border-t px-3 py-2.5">
                          {d.description && (
                            <p className="text-muted-foreground line-clamp-3 text-[12px]" title={d.description}>
                              {d.description}
                            </p>
                          )}
                          {subsets && (
                            <div>
                              <div className="mb-1.5 flex items-center justify-between">
                                <span className="text-muted-foreground text-[11px]">
                                  {t('ev.subsetsLabel', { n: subsets.length })}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => selectAllSubsets(d.name)}
                                  className="text-primary hover:underline text-[11px]"
                                >
                                  {t('ev.subsetsAll')}
                                </button>
                              </div>
                              <div className="flex max-h-32 flex-wrap gap-1.5 overflow-auto">
                                {subsets.map((s) => {
                                  const on = pickedSubs.includes(s);
                                  return (
                                    <button
                                      key={s}
                                      type="button"
                                      onClick={() => toggleSubset(d.name, s)}
                                      className={
                                        on
                                          ? 'bg-primary/10 border-primary/40 text-foreground rounded-full border px-2 py-0.5 text-[11px]'
                                          : 'border-border/70 text-muted-foreground rounded-full border px-2 py-0.5 text-[11px] opacity-60'
                                      }
                                    >
                                      {s}
                                    </button>
                                  );
                                })}
                              </div>
                              <p className="text-muted-foreground/70 mt-1 text-[11px]">
                                {t('ev.subsetsHint')}
                              </p>
                            </div>
                          )}
                          <div className="flex items-center gap-1.5">
                            <span className="text-muted-foreground text-[11px]">{t('ev.limitBefore')}</span>
                            <NumberInput
                              className="h-7 w-20 text-[13px]"
                              value={picked.limit}
                              onChange={(n) => setLimit(d.name, n)}
                              min={1}
                            />
                            <span className="text-muted-foreground text-[11px]">{t('ev.limitAfter')}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
                {filtered.length === 0 && (
                  <div className="text-muted-foreground px-3 py-4 text-[13px]">{t('ev.noMatch')}</div>
                )}
              </div>
            </FieldStack>
          ),
        },
      ]}
      assistantAutoOpen={Boolean(prefill?.context)}
      assistantIntro={prefill?.context ? prefillIntro('评测', prefill) : undefined}
      assistantConv={prefill?.context?.conv}
      assistantModel={prefill?.context?.model}
    />
  );
}
