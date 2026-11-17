import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Info, Rocket } from 'lucide-react';
import { toast } from 'sonner';
import { ConfigFormShell } from '@/components/ConfigFormShell';
import { EMPTY_GPU_SELECTION, GpuPicker, type GpuSelection } from '@/components/GpuPicker';
import { SearchSelect, TagInput, type SearchOption } from '@/components/SearchSelect';
import type { ConfigField } from '@/components/PageAssistant';
import { Field, FieldStack, NumberInput, SegmentedControl } from '@/components/FormField';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { MODULES } from '@/theme/modules';
import { listCatalog, modelOption, listCkptCatalog, ckptOption, ckptPick, type CatalogEntry, type CkptEntry } from '@/api/catalog';
import { listPlugins } from '@/api/plugins';
import { listDeploys, startDeploy } from '@/api/deploy';
import { pushRecent } from '@/store/recent';
import { prefillIntro, usePrefill } from '@/store/prefill';
import { useT } from '@/i18n';

/**
 * 新建部署：表单直连 vite 任务执行器（POST /api/deploy）。
 * attached 模式（无 --detach）：任务由 webui 持有，停止=SIGTERM 优雅关停；
 * 多部署各占一个端口、路径都是标准 /v1。就绪后 endpoint 自动进模型配置页。
 * 显卡分配两模式：本机选卡走 CUDA_VISIBLE_DEVICES；集群卡数只声明张数
 * （远端卡由集群调度，本机不设 CUDA 限制）。卡数统一进 --nproc_per_node——
 * Ray 据此给 sampler 分 N 张卡，张量并行由引擎从可见卡自动推出。
 */
export function DeployNewPage() {
  const { t } = useT();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const prefill = usePrefill('deploy');
  const [model, setModel] = useState('');
  const [servedName, setServedName] = useState('');
  const [engine, setEngine] = useState('vllm');
  const [port, setPort] = useState<number | undefined>(0);
  const [gpuSel, setGpuSel] = useState<GpuSelection>(EMPTY_GPU_SELECTION);
  const [plugins, setPlugins] = useState<string[]>([]);
  const [adapters, setAdapters] = useState<string[]>([]);
  /* 合并单个 LoRA 进权重（--merge_lora）：与 adapters 直挂二选一——勾了它启动慢但请求免 adapter 路由 */
  const [mergeLora, setMergeLora] = useState(false);
  const [extraArgs, setExtraArgs] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [modelOptions, setModelOptions] = useState<SearchOption[]>([]);
  /* 本机训练 ckpt：path -> 条目（选中时联动填 model/adapters） */
  const [ckptMap, setCkptMap] = useState<Record<string, CkptEntry>>({});
  /*
   * 选中 ckpt 的回显名（「底座:断点名」）：model 字段落到的是底座名，多个断点
   * 同底座时回显 value 分不出选的是哪个——SearchSelect 的 display 覆盖文本
   * 由此供给；未选 ckpt 时空串 = 回显 value 原文。
   */
  const [pickedCkpt, setPickedCkpt] = useState('');
  const [pluginOptions, setPluginOptions] = useState<SearchOption[]>([]);

  /*
   * 模型选中入口（表单与「将要执行」共用）：命中本机训练 ckpt 时联动填表——
   * full 断点 model 直接填路径；LoRA 断点 model 填底座、adapters 填路径。
   * 后改 model 不自动清 adapters（显式字段，用户可见可改）。
   */
  const pickModel = (v: string) => {
    const hit = ckptPick(ckptMap, v);
    if (hit) {
      setModel(hit.model);
      setPickedCkpt(ckptOption(ckptMap[v]).label ?? '');
      if (hit.adapters) setAdapters([hit.adapters]);
      return;
    }
    setPickedCkpt('');
    setModel(v);
  };

  useEffect(() => {
    /*
     * 模型候选一次装配：注册表清单（python 探针，秒级）与本机 ckpt（实时扫描，
     * 毫秒级）各自先行 set 会互相覆盖；两路到齐一次拼接，ckpt 排前——空关键
     * 词下拉截前 50，排在千余条注册表后面永远轮不到显示。
     */
    void Promise.all([
      listCatalog().then((c) => c.models).catch(() => [] as CatalogEntry[]),
      listCkptCatalog(),
    ]).then(([models, ckpts]) => {
      setModelOptions([...ckpts.map(ckptOption), ...models.map(modelOption)]);
      setCkptMap(Object.fromEntries(ckpts.map((c) => [c.path, c])));
    });
    listPlugins()
      .then((ps) => setPluginOptions(ps.map((p) => ({ value: p.source, label: p.id, hint: p.note }))))
      .catch(() => {
        /* 同上 */
      });
  }, []);

  /*
   * 两个带参入口：?from=<部署任务 id>「以此为模板新建」预填该部署的参数
   * （端口留 0 让新任务自选，避免与模板本体抢同一端口）；?model=<路径> 是
   * 导出详情页「去部署」带来的产物目录，只预填模型字段。任务不存在（webui
   * 重启后内存表清空）时静默退化为普通新建。
   */
  useEffect(() => {
    const from = searchParams.get('from');
    const presetModel = searchParams.get('model');
    /* AI 预填与带参入口互斥：带了预填就不回填 URL 预设 */
    if (prefill) return;
    if (presetModel) setModel(presetModel);
    if (!from) return;
    listDeploys()
      .then((tasks) => {
        const src = tasks.find((x) => x.id === from);
        if (!src) return;
        setModel(src.model);
        setServedName(src.servedName);
        setEngine(src.engine);
        setPlugins(src.plugins ?? []);
        setAdapters(src.adapters ?? []);
        setGpuSel(
          src.gpus?.length
            ? { mode: 'local', cards: src.gpus, count: undefined }
            : src.gpuCount
              ? { mode: 'cluster', cards: [], count: src.gpuCount }
              : EMPTY_GPU_SELECTION,
        );
      })
      .catch(() => {
        /* 后端暂不可达：保持空表单 */
      });
  }, [searchParams]);

  /*
   * 首页 AI 创建工具带来的预填：mount 时读取（读取即清），字段缺失静默忽略。
   * 选卡两形态互斥：卡号列表（本机）或卡数（集群），优先卡号列表。
   */
  useEffect(() => {
    if (!prefill) return;
    const f = prefill.fields;
    if (typeof f.model === 'string' && f.model.trim()) setModel(f.model.trim());
    if (typeof f.servedName === 'string' && f.servedName.trim()) setServedName(f.servedName.trim());
    if (f.engine === 'vllm' || f.engine === 'sglang') setEngine(f.engine);
    const port = Number(f.port);
    if (Number.isInteger(port)) setPort(port);
    if (Array.isArray(f.plugins)) setPlugins(f.plugins.map(String).filter(Boolean));
    if (Array.isArray(f.adapters)) setAdapters(f.adapters.map(String).filter(Boolean));
    if (Array.isArray(f.gpus) && f.gpus.length) {
      setGpuSel({
        mode: 'local',
        cards: f.gpus.map(Number).filter((n) => Number.isInteger(n) && n >= 0),
        count: undefined,
      });
    } else if (Number.isInteger(f.gpuCount) && (f.gpuCount as number) > 0) {
      setGpuSel({ mode: 'cluster', cards: [], count: f.gpuCount as number });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* 卡数：本机模式取勾选数，集群模式取声明的卡数；进 --nproc_per_node */
  const nproc = gpuSel.mode === 'local' ? gpuSel.cards.length : (gpuSel.count ?? 0);

  const preview = useMemo(() => {
    const lines = [
      t('dp.cmdComment'),
      ...(gpuSel.mode === 'local' && gpuSel.cards.length
        ? [`CUDA_VISIBLE_DEVICES=${gpuSel.cards.join(',')} \\`]
        : []),
      'swift deploy \\',
      `    --model ${model || t('dp.cmdModel')} \\`,
      `    --infer_backend ${engine} \\`,
      `    --served_model_name ${servedName || t('dp.cmdServedName')} \\`,
      ...(adapters.length ? [`    --adapters ${adapters.join(' ')} \\`] : []),
      ...(mergeLora ? ['    --merge_lora true \\'] : []),
      ...(plugins.length ? [`    --external_plugins ${plugins.join(' ')} \\`] : []),
      ...(nproc > 1 ? [`    --nproc_per_node ${nproc} \\`] : []),
      `    --port ${port || t('dp.cmdPort')}`,
      ...(extraArgs.trim() ? [`    ${extraArgs.trim()}`] : []),
    ];
    return lines.join('\n');
  }, [model, engine, servedName, port, gpuSel, nproc, plugins, adapters, mergeLora, extraArgs, t]);

  const submit = async () => {
    if (!model.trim()) {
      toast.error(t('dp.errModel'));
      return;
    }
    setSubmitting(true);
    try {
      const id = await startDeploy({
        model: model.trim(),
        servedName: servedName.trim() || undefined,
        engine,
        port: port || undefined,
        ...(plugins.length ? { plugins } : {}),
        ...(adapters.length ? { adapters } : {}),
        ...(mergeLora ? { mergeLora: true } : {}),
        ...(extraArgs.trim() ? { extraArgs: extraArgs.trim() } : {}),
        /* 两模式互斥：本机给卡号列表，集群给卡数 */
        ...(gpuSel.mode === 'local'
          ? gpuSel.cards.length
            ? { gpus: gpuSel.cards }
            : {}
          : gpuSel.count
            ? { gpuCount: gpuSel.count }
            : {}),
      });
      /* 启动即记入侧栏「最近」；任务 id 作去重键，重复启动别的部署不挤掉它 */
      pushRecent({
        id: `deploy:${id}`,
        label: t('dp.taskLabel', { name: servedName.trim() || model.trim().split('/').pop() || model.trim() }),
        path: `/deploy/${id}/overview`,
      });
      toast.success(t('dp.started'));
      navigate('/deploy');
    } catch (e) {
      toast.error(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  /* 交给 AI 助手的字段 */
  const fields: ConfigField[] = [
    { key: 'model', label: t('tn.modelLabel'), value: model, apply: pickModel },
    { key: 'served_model_name', label: t('dp.servedName'), value: servedName, apply: setServedName },
    { key: 'infer_backend', label: t('dp.engine'), value: engine, apply: setEngine },
    {
      key: 'adapters',
      label: 'LoRA adapter',
      value: adapters.join(' '),
      apply: (v) => setAdapters(v.split(/\s+/).filter(Boolean)),
    },
    { key: 'merge_lora', label: t('dp.mergeLora'), value: mergeLora ? 'true' : 'false', apply: (v) => setMergeLora(v === 'true') },
    { key: 'port', label: t('dp.port'), value: String(port ?? ''), apply: (v) => setPort(Number(v)) },
    {
      key: 'gpu',
      label: 'GPU',
      value:
        gpuSel.mode === 'local'
          ? gpuSel.cards.join(',')
          : gpuSel.count
            ? t('dp.clusterCards', { n: gpuSel.count })
            : '',
      apply: () => {},
    },
  ];

  return (
    <ConfigFormShell
      module={MODULES.deploy}
      title={t('dp.newDeploy')}
      desc={t('dp.desc')}
      preview={preview}
      fields={fields}
      onSubmit={() => void submit()}
      extraArgs={extraArgs}
      onExtraArgsChange={setExtraArgs}
      actions={
        <Button onClick={() => void submit()} disabled={submitting}>
          <Rocket /> {submitting ? t('tn.launching') : t('dp.launchBtn')}
        </Button>
      }
      sections={[
        {
          title: t('dp.secServer'),
          content: (
            <FieldStack>
              <Field label={t('tn.modelLabel')} hint={t('tn.modelHint')}>
                <SearchSelect
                  value={model}
                  onChange={pickModel}
                  options={modelOptions}
                  placeholder="Qwen/Qwen3-8B"
                  display={pickedCkpt || undefined}
                />
              </Field>
              <Field label={t('dp.servedName')} hint={t('dp.servedNameHint')}>
                <Input
                  value={servedName}
                  onChange={(e) => setServedName(e.target.value)}
                  placeholder={t('dp.servedNamePh')}
                  className="h-8 text-[13px]"
                />
              </Field>
              <Field label={t('dp.engine')}>
                <SegmentedControl
                  value={engine}
                  onChange={setEngine}
                  options={[
                    { value: 'vllm', label: 'vLLM' },
                    { value: 'sglang', label: 'SGLang' },
                  ]}
                />
              </Field>
              <Field label={t('dp.port')} hint={t('dp.portHint')}>
                <NumberInput
                  className="w-50"
                  value={port}
                  onChange={setPort}
                  min={0}
                  max={65535}
                />
              </Field>
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
              <Field
                label="LoRA adapter"
                hint={t('dp.adaptersHint')}
              >
                <TagInput
                  values={adapters}
                  onChange={setAdapters}
                  options={[]}
                  placeholder={t('dp.adaptersPh')}
                />
              </Field>
              <Field label={t('dp.mergeLora')} hint={t('dp.mergeLoraHint')}>
                {/* 开关带状态字（一格网格里只放一枚 Switch 会显得空） */}
                <div className="flex h-8 items-center gap-2.5">
                  <Switch checked={mergeLora} onCheckedChange={setMergeLora} />
                  <span className="text-muted-foreground text-[12px]">{mergeLora ? t('tn.switchOn') : t('tn.switchOff')}</span>
                </div>
              </Field>
            </FieldStack>
          ),
        },
        {
          title: t('tn.secGpu'),
          content: (
            <FieldStack>
              <Field
                label={t('tn.gpuLabel')}
                hint={
                  gpuSel.mode === 'local'
                    ? t('dp.gpuLocalHint')
                    : t('dp.gpuClusterHint')
                }
              >
                <GpuPicker value={gpuSel} onChange={setGpuSel} />
              </Field>
            </FieldStack>
          ),
        },
        {
          title: t('dp.secNote'),
          content: (
            <div className="text-muted-foreground flex items-start gap-2 text-[13px] leading-5">
              <Info size={14} className="mt-0.5 flex-none" />
              {t('dp.note')}
            </div>
          ),
        },
      ]}
      assistantAutoOpen={Boolean(prefill?.context)}
      assistantIntro={prefill?.context ? prefillIntro('部署', prefill) : undefined}
      assistantConv={prefill?.context?.conv}
      assistantModel={prefill?.context?.model}
    />
  );
}
