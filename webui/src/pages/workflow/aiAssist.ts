import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import type { AiMessage, ActionState } from '@/components/AiChatPanel';
import { messageId, reconcileEngineMessages, toEngineDisplay, userMessage } from '@/components/AiChatPanel';
import { pushChat } from '@/store/chats';
import {
  submitTurn,
  openTurnLive,
  abortTurn,
  fetchConvHistory,
  fetchPendingPageTools,
  resolvePageTool,
  type PageToolSpec,
} from '@/api/turns';
import type { ChatMessage } from '@/api/models';
import type { GraphEdge, GraphFrame, GraphNode, NodeParam } from './nodeTypes';
import { NODE_TYPES, findPortOf, nodeLabel, paramsOf } from './nodeTypes';
import type { TFn } from './nodeTypes';
import registryData from './registry.json';
import type { RegistryComponent, RegistryParam, RegistryPort } from './registryNodes';

/* 参数生效视图的真相源在 nodeTypes（画布/面板/提交共用）；这里转发给
 * 既有消费方（NodeInspector 等），语义见 nodeTypes.paramsOf 的注释。 */
export { paramsOf } from './nodeTypes';

/**
 * 编排页的 AI 协助层。
 *
 * 这里放三样东西：AI 能提出的动作类型（AiAction）、离线剧本（无可用模型
 * 时的关键词匹配兜底，消息带 offline 标记）、引擎式会话 hook（useFlowAiChat
 * ——回合由 server 引擎执行，动作经页面工具挂起协议回传）。
 *
 * 最重要的一条设计不变：AI 不直接改图。它只能"提议"——每个提议是一张
 * 动作卡片，得点了应用才会落到图上。理由是这份图形是流程的唯一事实
 * 来源，如果 AI 能静默改它，「谁把 num_generations 从 8 调成 32 的」
 * 就会变成查不出来的事。提议的上抛通道从旧版 ```action 文本块换成
 * tool-calling（set_node_param / add_node / add_edge / flow_control /
 * describe_component 页面工具，引擎挂起上抛、页面出卡、用户点应用）。
 */

/** AI 能提议的动作。故意做得很窄——只有这几种能被安全地一键应用 */
export type AiAction =
  | {
      kind: 'param';
      /** 目标节点 id */
      nodeId: string;
      /** 参数名，对应 NodeParam.label */
      label: string;
      /** 改成什么 */
      to: string;
    }
  | {
      kind: 'addNode';
      typeKey: string;
      note: string;
      /** 随节点落图的参数（label→字符串值）；不给则用类型默认 */
      params?: Record<string, string>;
      /** 随节点落图的连线：新节点一端写 'new'，应用时换成真实 id */
      connections?: { from: string; fromPort: string; to: string; toPort: string }[];
      /** 放入哪个循环框（memberIds 追加）；不给则落在框外 */
      frameId?: string;
    }
  | {
      kind: 'addEdge';
      /** 两端都是图上已有节点（新节点随线落图走 addNode 的 connections） */
      from: string;
      fromPort: string;
      to: string;
      toPort: string;
      note: string;
    }
  | { kind: 'control'; op: 'start' | 'pause' | 'resume' | 'stop' };

/** 编排页的一条消息。消息的形状是通用的，只有提议的类型是这一页特有的 */
export type FlowMessage = AiMessage<AiAction>;

/** 问 AI 时带上的上下文。有 node 就是在问单个节点，没有就是在问整张图 */
export interface AiContext {
  node?: GraphNode;
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** 循环框（带成员的才是真归属）：助手要知道环在哪，才能把新阶段放进环里 */
  frames?: GraphFrame[];
  presetLabel: string;
  running: boolean;
}

function paramValue(node: GraphNode, label: string): string | undefined {
  return paramsOf(node).find((p) => p.label === label)?.value;
}

function reply(text: string, actions?: AiAction[]): FlowMessage {
  return {
    id: messageId(),
    role: 'assistant',
    text,
    actions,
    states: actions?.map(() => 'pending' as ActionState),
    /* 剧本产出：面板据此给这条消息标「本地示意」，与 LLM 回复区分 */
    offline: true,
  };
}

/** 单个节点视角的开场白：直接把这个节点在干什么、接了什么说清楚 */
export function nodeIntro(node: GraphNode, ctx: AiContext, t: TFn): FlowMessage {
  const def = NODE_TYPES[node.type];
  const upstream = ctx.edges
    .filter((e) => e.to === node.id)
    .map((e) => nodeLabel(NODE_TYPES[ctx.nodes.find((n) => n.id === e.from)?.type ?? ''], t))
    .filter(Boolean);
  const downstream = ctx.edges
    .filter((e) => e.from === node.id)
    .map((e) => nodeLabel(NODE_TYPES[ctx.nodes.find((n) => n.id === e.to)?.type ?? ''], t))
    .filter(Boolean);

  const lines = [t('wf.aiNodeIntroTitle', { label: nodeLabel(def, t), hint: def.hint ? t(def.hint) : '' })];
  lines.push(
    upstream.length
      ? t('wf.aiUpstream', { list: [...new Set(upstream)].join(t('wf.listSep')) })
      : t('wf.aiNoUpstream'),
  );
  lines.push(
    downstream.length
      ? t('wf.aiDownstream', { list: [...new Set(downstream)].join(t('wf.listSep')) })
      : t('wf.aiNoDownstream'),
  );
  if (def.lockedPorts) {
    lines.push(t('wf.aiLockedNote'));
  }
  lines.push(t('wf.aiAskAnything'));
  return reply(lines.join('\n'));
}

/** 整图视角的开场白。跟整图兜底是同一套话，统一从这里出 */
export function flowIntro(ctx: AiContext, t: TFn): FlowMessage {
  return askAi('', ctx, t);
}

/**
 * 假 AI。按关键词匹配剧本。
 *
 * 剧本不是随便编的：每条都对应一个真实会被问到的问题，
 * 回答里引用的是图上当前的真实取值，所以看着才像真在读这张图。
 * 无可用模型（models.json 一个都没配）时整页回落到它，页面不空转。
 */
export function askAi(question: string, ctx: AiContext, t: TFn): FlowMessage {
  const q = question.toLowerCase();
  const node = ctx.node;

  /* ---- 控制类：启动 / 暂停 ---- */
  if (/暂停|停一下|停下|pause/.test(q)) {
    return reply(
      ctx.running ? t('wf.aiPauseRunning') : t('wf.aiPauseIdle'),
      ctx.running ? [{ kind: 'control', op: 'pause' }] : [{ kind: 'control', op: 'start' }],
    );
  }
  if (/继续|恢复|resume/.test(q)) {
    return reply(t('wf.aiResume'), [{ kind: 'control', op: 'resume' }]);
  }
  if (/停止|终止|别跑了|stop|kill/.test(q)) {
    return reply(t('wf.aiStop'), [{ kind: 'control', op: 'stop' }]);
  }
  if (/跑起来|启动|开始训|运行|开跑|start|run/.test(q)) {
    return reply(
      t('wf.aiStart', { n: ctx.nodes.length, m: ctx.edges.length }),
      [{ kind: 'control', op: 'start' }],
    );
  }

  /* ---- 显存 ---- */
  if (/oom|显存|内存不够|out of memory|爆了/.test(q)) {
    const trainer = ctx.nodes.find((n) => n.type === 'train_step');
    const cur = trainer ? paramValue(trainer, 'micro_batch_size') : undefined;
    const actions: AiAction[] =
      trainer && cur && Number(cur) > 1
        ? [{ kind: 'param', nodeId: trainer.id, label: 'micro_batch_size', to: '1' }]
        : [];
    return reply(
      t('wf.aiOom', { cur: cur ? t('wf.aiOomCur', { cur }) : '' }),
      actions,
    );
  }

  /* ---- 优势为 0 ---- */
  if (/优势.*0|advantage.*0|全对|全错|方差/.test(q)) {
    const adv = ctx.nodes.find((n) => n.type === 'advantage');
    const numGen = adv ? paramValue(adv, 'num_generations') : undefined;
    return reply(
      t('wf.aiAdvZero', { ng: numGen ? t('wf.aiNg', { n: numGen }) : '' }),
    );
  }

  /* ---- 速度 ---- */
  if (/慢|快一点|速度|太久|加速|slow|faster|speed|too long|accelerate/.test(q)) {
    const rollout = ctx.nodes.find(
      (n) => n.type === 'rollout' || n.type === 'generate',
    );
    const gen = rollout ? paramValue(rollout, 'num_generations') : undefined;
    const actions: AiAction[] =
      rollout && gen && Number(gen) > 4
        ? [{ kind: 'param', nodeId: rollout.id, label: 'num_generations', to: '4' }]
        : [];
    return reply(
      t('wf.aiSpeed', { gen: gen ? t('wf.aiGenNow', { n: gen }) : '' }),
      actions,
    );
  }

  /* ---- 轨迹对错位 ---- */
  if (/错位|对不上|对齐|过滤|丢样本|filter|misalign|out of sync/.test(q)) {
    return reply(t('wf.aiMisalign'));
  }

  /* ---- 学习率 ---- */
  if (/学习率|lr|learning rate/.test(q)) {
    return reply(t('wf.aiLr'));
  }

  /* ---- 节点上下文相关的兜底 ---- */
  if (node) {
    const def = NODE_TYPES[node.type];
    const ps = paramsOf(node);
    return reply(
      t('wf.aiNodeFallback', {
        label: nodeLabel(def, t),
        params: ps.map((p) => `· ${p.label} = ${p.value}`).join('\n'),
        hint: def.hint ? t(def.hint) : '—',
      }),
    );
  }

  /* ---- 整图兜底 ---- */
  return reply(
    t('wf.aiGraphFallback', {
      preset: ctx.presetLabel,
      n: ctx.nodes.length,
      m: ctx.edges.length,
      state: ctx.running ? t('wf.aiStateRunning') : t('wf.aiStateIdle'),
    }),
  );
}

/**
 * 把一条提议翻成卡片上的样子。
 *
 * 翻译放在这里而不是面板里：面板是通用的，它不知道 nodeId 对应哪个节点，
 * 也不该知道——新建页那边的提议改的是表单字段，压根没有节点这回事。
 */
export function describeAction(a: AiAction, nodes: GraphNode[], t: TFn): { label: string; heavy?: boolean } {
  /* 控制类（启停）比改个参数重得多，给它醒目色 */
  return { label: actionLabel(a, nodes, t), heavy: a.kind === 'control' };
}

function actionLabel(a: AiAction, nodes: GraphNode[], t: TFn): string {
  if (a.kind === 'param') {
    const n = nodes.find((x) => x.id === a.nodeId);
    const label = n ? nodeLabel(NODE_TYPES[n.type], t) : a.nodeId;
    const cur = n ? paramValue(n, a.label) : undefined;
    return t('wf.aiActParam', {
      label,
      param: a.label,
      from: cur ? `${cur} → ` : t('wf.aiActSetTo'),
      to: a.to,
    });
  }
  if (a.kind === 'addNode') {
    let label = t('wf.aiAddNode', { label: nodeLabel(NODE_TYPES[a.typeKey as GraphNode['type']], t) || a.typeKey });
    const p = Object.keys(a.params ?? {}).length;
    const e = a.connections?.length ?? 0;
    if (p || e) label += t('wf.aiAddNodeWiring', { p, e });
    if (a.frameId) label += t('wf.aiAddNodeFrame', { frame: a.frameId });
    return label;
  }
  if (a.kind === 'addEdge') {
    return t('wf.aiAddEdge', { from: `${a.from}:${a.fromPort}`, to: `${a.to}:${a.toPort}` });
  }
  const ops = { start: t('wf.aiOpStart'), pause: t('task.pause'), resume: t('task.resume'), stop: t('task.stop') };
  return ops[a.op];
}

/** 注册表原文（端口/参数的唯一真相源）：describe_component 直接读它，不经
 * NODE_TYPES 的展示层投影（那里丢了 options/doc，助手设 dict 参数会无从查键名）。 */
const REGISTRY_COMPONENTS = registryData.components as Record<string, RegistryComponent>;

/** 默认值转文本：对象/数组走 JSON，标量走 String（与面板默认值同口径） */
function paramDefaultText(value: unknown): string {
  if (value === null || value === undefined) return '';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

/** 小模型常把嵌套的对象/数组参数二次 JSON 编码成字符串（实测 Qwen3.5-4B 的
 * add_node：params/connections 传成 "{\"...}" / "[...]"）。先解一层 JSON 还原；
 * 解不出时返回原值，由调用方按形状拒绝并点名原因。 */
function unwrapJsonArg(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  try { return JSON.parse(value) as unknown; } catch { return value; }
}

/** 与画布手拉连线同一套校验（端口存在/类型兼容含 acceptedKinds/锁定组件），
 * add_node 的随线校验与 add_edge 共用。返回错误文案，'' 为通过。 */
function checkWiring(src: GraphNode, fromPort: string, dst: GraphNode, toPort: string): string {
  const pa = findPortOf(src, fromPort, 'out');
  const pb = findPortOf(dst, toPort, 'in');
  if (!pa || !pb) {
    return `端口不存在：${src.id}:${fromPort} -> ${dst.id}:${toPort}；输出口查源节点、输入口查目标节点的注册表声明。`;
  }
  if (pa.type !== pb.type && !(pb.acceptedKinds ?? []).includes(pa.type)) {
    return `端口类型不匹配：${src.id}:${fromPort}（${pa.type}）-> ${dst.id}:${toPort}（${pb.type}）。`;
  }
  const locked = NODE_TYPES[src.type]?.lockedPorts ? src.type : NODE_TYPES[dst.type]?.lockedPorts ? dst.type : '';
  if (locked) return `连线被拒：${locked} 是连线锁定组件，只能改参数不能接线。`;
  return '';
}

function portLine(p: RegistryPort): string {
  const flags = `${p.many ? '多值 ' : ''}${p.optional ? '可选 ' : ''}`.trim();
  const accepts = p.accepted_kinds?.length ? `（另接受 ${p.accepted_kinds.join('/')}）` : '';
  return `- ${p.key} [${p.kind}]${flags ? ` ${flags}` : ''}：${p.role}${accepts}`;
}

function paramLine(p: RegistryParam): string {
  const head = `- ${p.name}（${p.type ?? 'str'}${p.required ? '，必填' : ''}，默认 ${paramDefaultText(p.default) || '空'}）`;
  const opts = p.options?.length ? `；枚举：${p.options.map((o) => String(o)).join(' / ')}` : '';
  const doc = p.doc ? `；${p.doc}` : '';
  return `${head}${opts}${doc}`;
}

/**
 * 组件的注册表声明全文（端口表 + 参数值域）：助手接线/设参前查它拿真实
 * 端口 key 与参数名，杜绝按惯例猜 ref_model/kl_coeff 这类不存在的名字。
 * 类型不在注册表时回错误并点名可用全集，让模型自行改正。
 */
export function describeComponentText(type: string): string {
  const c = REGISTRY_COMPONENTS[type];
  if (!c) {
    return `没有类型为 ${type} 的组件；可用类型：${Object.keys(REGISTRY_COMPONENTS).join(' / ')}。`;
  }
  const lines = [`组件 ${c.type}（${c.label}，${c.lifecycle}）：${c.math_doc}`];
  lines.push('输入口：', ...(c.ports_in.length ? c.ports_in.map(portLine) : ['（无）']));
  lines.push('输出口：', ...(c.ports_out.length ? c.ports_out.map(portLine) : ['（无）']));
  lines.push('参数：', ...(c.params.length ? c.params.map(paramLine) : ['（无）']));
  return lines.join('\n');
}

/** 图的文本摘要：system prompt 的主体。只放事实（类型/参数/连线），不放解读。 */
export function graphContextText(ctx: AiContext, t?: TFn): string {
  const lines: string[] = [];
  lines.push(`流程图「${ctx.presetLabel}」，${ctx.nodes.length} 个节点、${ctx.edges.length} 根连线，运行状态 ${ctx.running ? 'running' : 'idle'}。`);
  /* 可用类型全集来自注册表而非当前图：只列图上已有类型会让助手把
   * 「图上没有」误判成「平台没有」（实测教训：code_component / model_forward
   * 不在图上时被断言不存在）。Object.keys 不经 Proxy 的 get 陷阱，枚举到的
   * 全是注册表内真类型，不含 unknown 占位。 */
  lines.push(`可用组件类型（add_node 可加其中任何一个，不限于图上已出现的）：${Object.keys(NODE_TYPES).join(' / ')}。`);
  lines.push('载荷事实：每条轨迹（traj）自带 rollout_logps（采样器逐 token logprob）与 response_token_ids；model_forward 节点可输出训练侧 logps（outputs=["logps"]），接 advantage 的 old_out 口并设 kwargs={"log_rollout_offpolicy_metrics": true} 即在 metric 口产出训推一致性诊断（rollout_correction/*）。');
  lines.push('add_node 可一次带 params（参数名→字符串值）、connections（新节点一端写 "new"，另一端用图上节点 id 与端口 key）与 frameId（放入已有循环框）——加节点、设参数、接线、进环一张卡完成，不要只加一个孤节点。');
  lines.push('两个已有节点之间补线用 add_edge（两端都用图摘要里的节点 id 与注册表端口 key）；单输入口重连即替换旧线。');
  lines.push('端口 key 与参数键名以注册表为准：接线或设参（尤其 loss_kwargs/kwargs 这类 dict）前先调 describe_component(type) 查该组件的端口表与参数值域，不要按惯例猜名字（如参考前向接 advantage.ref_out 或 train_step.reference，而非自造的 ref_model）。');
  /* 循环框是迭代作用域：新阶段要每步重跑就必须进框。只列带成员的真框，
   * 纯视觉分区没有 memberIds，不参与运行语义。 */
  for (const f of ctx.frames ?? []) {
    if (!(f.memberIds ?? []).length) continue;
    const label = t ? t(f.label) : f.label;
    lines.push(`- 循环框 ${f.id}（${label}）：成员 ${f.memberIds!.join(', ')}`);
  }
  for (const n of ctx.nodes) {
    const def = NODE_TYPES[n.type];
    const params = paramsOf(n)
      .map((p) => `${p.label}=${p.value}`)
      .join(', ');
    lines.push(`- 节点 ${n.id}（${def?.label ?? n.type}）${params ? `：${params}` : ''}`);
  }
  for (const e of ctx.edges) {
    lines.push(`- 连线 ${e.from}:${e.fromPort} -> ${e.to}:${e.toPort}`);
  }
  return lines.join('\n');
}

export const ASSISTANT_SYSTEM_PROMPT = `你是 ms-swift 可视化编排页的流程助手。用户正在一张拖拽组装的训练流程图上工作，你回答关于这张图的问题。
回答规则：
- 简洁直接，先给结论再给理由；引用图上的真实取值时用节点 id 或参数名。
- 接线或设参前先调 describe_component(type) 查注册表的真实端口 key 与参数键名/值域，不要按惯例猜（参考前向是 advantage.ref_out / train_step.reference，GRPO 的 KL 系数是 loss_kwargs.beta、熵奖励是 entropy_coef）。提议被拒（端口/参数不存在）时先查注册表再改正重发，不要据此断言「平台做不到」；只有查过注册表确无对应能力时才明说做不到和为什么。
- 想替用户改参数、加节点或接线，调用 set_node_param / add_node / add_edge：修改会以卡片列出，用户点了「应用」才会落图——正文里说明为什么提议即可，不要催用户应用。add_node 支持同时带参数、连线与循环框归属：替用户接一个组件时把参数和接线一次提全，不要只加一个没接线的孤节点；两个已有节点之间补线用 add_edge，不要承诺「稍后补线」却不调工具。
- 想替用户启停流程，调用 flow_control（start/pause/resume/stop；执行前会弹审批卡，用户确认后才执行）。
- 不确定就说不确定，不要编造图上不存在的参数或节点。`;

/**
 * 会话桶的 ChatMessage → 面板展示消息：共享基础版（编排页用户消息不带图片）。
 * 思维链/工具调用/工具结果以结构化字段交给面板渲染折叠卡（与对话页同一套）。
 */
const toDisplay = (m: ChatMessage): FlowMessage => toEngineDisplay<AiAction>(m);

/** 拖拽页的五个页面工具（schema 随回合提交；执行体在页面，经挂起协议回传） */
const FLOW_PAGE_TOOLS: PageToolSpec[] = [
  {
    name: 'set_node_param',
    description: '把某个节点的某个参数改成新值。修改以卡片列出，用户逐条点「应用」才落图。',
    parameters: {
      type: 'object',
      properties: {
        nodeId: { type: 'string', description: '目标节点 id（图摘要里列出的）' },
        label: { type: 'string', description: '参数名（对应节点参数 label）' },
        to: { type: 'string', description: '新值（一律字符串）' },
      },
      required: ['nodeId', 'label', 'to'],
    },
  },
  {
    name: 'add_node',
    description:
      '在图上新增一个组件节点，可同时设参数、接连线、放入循环框——一张卡原子落图。用户点「应用」才生效。替用户接组件时务必带全 connections（新节点一端写 "new"），只加孤节点等于没接。',
    parameters: {
      type: 'object',
      properties: {
        typeKey: { type: 'string', description: '组件类型 key（注册表内任何类型，全集见图摘要的可用组件类型清单）' },
        note: { type: 'string', description: '为什么加这个节点（卡片上展示）' },
        params: {
          type: 'object',
          description: '参数名→新值（一律字符串；参数名必须是该类型声明的 label，如 model_forward 的 tokenizer/outputs）',
          additionalProperties: { type: 'string' },
        },
        connections: {
          type: 'array',
          description: '随节点落图的连线。每根恰有一端是 "new"（新节点），另一端用图摘要里的节点 id；端口用注册表 key（如 out/traj/model/old_out）',
          items: {
            type: 'object',
            properties: {
              from: { type: 'string', description: '源节点 id 或 "new"' },
              fromPort: { type: 'string', description: '源输出口 key' },
              to: { type: 'string', description: '目标节点 id 或 "new"' },
              toPort: { type: 'string', description: '目标输入口 key' },
            },
            required: ['from', 'fromPort', 'to', 'toPort'],
          },
        },
        frameId: { type: 'string', description: '要把新节点放进的循环框 id（图摘要列出）；需要每步重跑的阶段必须进环' },
      },
      required: ['typeKey', 'note'],
    },
  },
  {
    name: 'add_edge',
    description:
      '在两个已有节点之间接一根连线（两端都用图摘要里的节点 id）。出卡前校验端口存在与类型兼容；用户点「应用」才落图，单输入口重连即替换旧线。加新节点并随线落图用 add_node，不用本工具。',
    parameters: {
      type: 'object',
      properties: {
        from: { type: 'string', description: '源节点 id（图上已有）' },
        fromPort: { type: 'string', description: '源输出口 key（注册表声明，如 model/out/traj）' },
        to: { type: 'string', description: '目标节点 id（图上已有）' },
        toPort: { type: 'string', description: '目标输入口 key（注册表声明，如 model/ref_out/sync 的 model）' },
        note: { type: 'string', description: '为什么接这根线（卡片上展示）' },
      },
      required: ['from', 'fromPort', 'to', 'toPort'],
    },
  },
  {
    name: 'flow_control',
    description: '启动/暂停/恢复/停止整张流程图的运行（start/pause/resume/stop）。执行前需用户审批。',
    parameters: {
      type: 'object',
      properties: { op: { type: 'string', enum: ['start', 'pause', 'resume', 'stop'], description: '要执行的操作' } },
      required: ['op'],
    },
    approval: { title: '控制流程运行', reason: '对流程图的常驻会话执行启停控制（AI 提议，需用户确认）。' },
  },
  {
    name: 'describe_component',
    description:
      '查某个组件类型的注册表声明：输入/输出端口（key、通道、角色）与全部参数（名、类型、默认值、枚举、说明）。接线或设参（尤其 loss_kwargs/kwargs 这类 dict）前先调它拿真实端口 key 与参数键名，不要按惯例猜。只读，不需审批。',
    parameters: {
      type: 'object',
      properties: {
        type: { type: 'string', description: '组件类型 key（图摘要的可用组件类型清单里的，如 advantage/train_step/loss/model_forward）' },
      },
      required: ['type'],
    },
  },
];

/**
 * 页面工具调用的全局去重表：整图抽屉与节点面板两个 hook 实例可能同时订阅
 * 同一会话（两面板同时开着），callId 首见才执行——不共享会双出卡+双回传。
 * callId 是 UUID，跨会话不碰撞。
 */
const seenPageCallIds = new Set<string>();

/**
 * 挂起调用的真实执行模型（callId → executor）：单流会话切模型后，卡片
 * 决定/工具回传仍按挂起时的执行者路由（与 seenPageCallIds 同生命周期）。
 */
const pageToolExecutors = new Map<string, string>();

/**
 * 编排页 AI 会话 hook（整图抽屉 FlowAssistant 与节点面板 NodeInspector 共用）：
 * 回合由 server 引擎执行（单写者架构——压缩历史、模型流、落盘全在引擎），
 * 动作提议经页面工具挂起协议（引擎上抛 → 页面出卡 → 用户点应用 → 回传续轮）。
 * 无可用模型时整条路回落离线剧本 askAi（页面不空转、不提交引擎）。
 */
export function useFlowAiChat({
  baseConv,
  model,
  apiKey,
  ctx,
  t,
  onControl,
  confirmBeforeApply,
  onAction,
}: {
  /** 会话锚点：已保存实验传 `wf-<flowId>`（重开画布续上历史图对话），未保存传 ''（首发建临时会话） */
  baseConv: string;
  /** 当前模型 id（'' = 无可用模型，发送回落离线剧本）；只是执行者，历史与模型无关 */
  model: string;
  /** MS 默认模型的会话令牌：随回合提交鉴权（models.json 条目不传） */
  apiKey?: string;
  ctx: () => AiContext;
  t: TFn;
  onControl: (op: 'start' | 'pause' | 'resume' | 'stop') => void;
  /** 偏好「应用前确认」：关闭时提议出卡即自动落图（与 PageAssistant 免确认同语义）；缺省按要确认处理 */
  confirmBeforeApply?: boolean;
  /** 非控制类提议的应用回调（改参/加节点/接线）；免确认自动落图用，与卡片按钮同一条路 */
  onAction?: (a: AiAction) => void;
}) {
  const [messages, setMessages] = useState<FlowMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  /* 会话 id：'' = 未起（首发时建临时）；SSE 订阅与回传都以它为锚 */
  const [convId, setConvId] = useState(baseConv);
  /* 图与控制回调都活在组件 render 里——hook 的 effect 闭包会过期，用 ref 拿现值 */
  const ctxRef = useRef(ctx);
  ctxRef.current = ctx;
  const controlRef = useRef(onControl);
  controlRef.current = onControl;
  const apiKeyRef = useRef(apiKey);
  apiKeyRef.current = apiKey;
  const confirmRef = useRef(confirmBeforeApply);
  confirmRef.current = confirmBeforeApply;
  const actionRef = useRef(onAction);
  actionRef.current = onAction;

  /* 出卡唯一出口：免确认（confirmBeforeApply=false 且页面给了应用回调）时立即
   * 落图并把卡标成已应用；否则留待决等用户点。只在新提议时调用——历史重放
   * 走 reconcile 插回旧卡，不经这里，不会把旧提议再应用一遍。 */
  const propose = (actions: AiAction[]): boolean => {
    const auto = confirmRef.current === false && !!actionRef.current;
    if (auto) actions.forEach((a) => actionRef.current!(a));
    setMessages((ms) => [
      ...ms,
      {
        id: messageId(),
        role: 'assistant',
        text: t('assistant.proposeChanges'),
        actions,
        states: actions.map(() => (auto ? 'applied' : 'pending') as ActionState),
      },
    ]);
    return auto;
  };

  /*
   * 挂载/切会话即从 server 拉回单流历史（引擎单写者落盘，刷新/重开画布
   * 不丢）：历史与模型无关（固定 '*' 桶），切模型不重拉也不搬桶。桶重建，
   * 本地未决提议卡由 reconcile 插回；首发建档引起的 convId 变化桶还空时
   * 不动本地乐观消息。
   */
  useEffect(() => {
    if (!convId) return;
    const conv = convId;
    void fetchConvHistory(conv).then((history) => {
      if (!history.length) return;
      setMessages((ms) => reconcileEngineMessages(ms, history, toDisplay));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [convId]);

  /* 切会话（历史会话下拉）：目标历史由上方 effect 拉回；新会话清空消息，
   * 开场白由面板在空消息时补上 */
  const switchConv = (id: string) => {
    if (streaming) return;
    setConvId(id);
    setMessages([]);
  };

  /*
   * 页面工具的执行面：引擎把挂起调用上抛（SSE page-tool 事件或恢复重放），
   * 这里出卡（param/addNode，等用户点应用）或执行控制（flow_control，审批
   * 已由引擎先行完成），然后回传结果续轮。恢复重放时 flow_control 不重执行
   * ——防止刷新造成误启停，改回执请用户用画布上的按钮。
   */
  const runPageTool = async (
    call: { id: string; name: string; arguments: string },
    replay: boolean,
    executor?: string,
  ) => {
    if (seenPageCallIds.has(call.id)) return;
    seenPageCallIds.add(call.id);
    /* 记住真实执行模型：用户之后切了模型，回传仍按它路由 */
    const exec = executor ?? model;
    pageToolExecutors.set(call.id, exec);
    let args: Record<string, unknown> = {};
    try {
      args = JSON.parse(call.arguments || '{}') as Record<string, unknown>;
    } catch {
      await resolvePageTool(convId, exec, call.id, '参数不是合法 JSON；请重发完整参数。');
      return;
    }
    let result = '';
    if (call.name === 'set_node_param') {
      const nodeId = String(args.nodeId ?? '');
      const label = String(args.label ?? '');
      const to = String(args.to ?? '');
      const node = ctxRef.current().nodes.find((n) => n.id === nodeId);
      if (node && label) {
        const auto = propose([{ kind: 'param', nodeId, label, to }]);
        result = auto
          ? '免确认已开启：参数修改已自动落图。'
          : '已列出参数修改卡片，用户点「应用」才会改图；应用结果不回显，可在图上核对。';
      } else {
        result = `图上没有节点 ${nodeId} 或参数名为空；请用图摘要里的节点 id 与参数名重发。`;
      }
    } else if (call.name === 'add_node') {
      const typeKey = String(args.typeKey ?? '');
      const note = String(args.note ?? '');
      /* in 不经 Proxy 的 get 陷阱：NODE_TYPES[typeKey] 缺键兑占位对象恒真，
       * 拿它当守卫会让任意字符串都落一张灰卡 */
      if (!(typeKey && typeKey in NODE_TYPES)) {
        result = `没有类型为 ${typeKey} 的组件；请用图摘要的可用组件类型清单里的 key 重发。`;
      } else {
        /* 参数/连线/进环在出卡前逐条校验：不合法的提议直接拒回并点名
         * 原因，让模型能自行改正重发，而不是落一张应用时才报错的卡。 */
        const def = NODE_TYPES[typeKey];
        const ctxNow = ctxRef.current();
        const params: Record<string, string> = {};
        let invalid = '';
        /* params/connections 可能被小模型二次 JSON 编码成字符串：先解一层还原，
         * 形状仍不对就点名拒绝（Object.entries 落到字符串会枚举字符下标，产出
         * 「没有参数 0」这类模型无从改正的回执）。 */
        const paramsRaw = unwrapJsonArg(args.params);
        const paramsObj = paramsRaw != null && typeof paramsRaw === 'object' && !Array.isArray(paramsRaw)
          ? (paramsRaw as Record<string, unknown>) : null;
        if (args.params != null && paramsObj === null) {
          invalid = 'params 必须是「参数名→值」的对象（不要二次 JSON 编码成字符串）；请重发。';
        }
        for (const [label, value] of Object.entries(paramsObj ?? {})) {
          if (!def.params.some((p) => p.label === label)) {
            invalid = `类型 ${typeKey} 没有参数 ${label}；可用参数：${def.params.map((p) => p.label).join(', ') || '（无）'}。`;
            break;
          }
          params[label] = String(value);
        }
        const connections: { from: string; fromPort: string; to: string; toPort: string }[] = [];
        const newNode: GraphNode = { id: 'new', type: typeKey, x: 0, y: 0, status: 'idle' };
        const connRaw = unwrapJsonArg(args.connections);
        if (!invalid && args.connections != null && !Array.isArray(connRaw)) {
          invalid = 'connections 必须是连线对象数组（不要二次 JSON 编码成字符串）；请重发。';
        }
        if (!invalid) {
          for (const raw of Array.isArray(connRaw) ? connRaw : []) {
            const c = (raw ?? {}) as Record<string, unknown>;
            const from = String(c.from ?? '');
            const fromPort = String(c.fromPort ?? '');
            const to = String(c.to ?? '');
            const toPort = String(c.toPort ?? '');
            if ((from === 'new') === (to === 'new')) {
              invalid = `连线 ${from}:${fromPort} -> ${to}:${toPort} 必须恰有一端是 "new"（新节点）。`;
              break;
            }
            const other = ctxNow.nodes.find((n) => n.id === (from === 'new' ? to : from));
            if (!other) {
              invalid = `图上没有节点 ${from === 'new' ? to : from}；请用图摘要里的节点 id。`;
              break;
            }
            const pa = findPortOf(from === 'new' ? newNode : other, fromPort, 'out');
            const pb = findPortOf(to === 'new' ? newNode : other, toPort, 'in');
            if (!pa || !pb) {
              invalid = `端口不存在：${from}:${fromPort} -> ${to}:${toPort}；输出口查源节点、输入口查目标节点的注册表声明。`;
              break;
            }
            /* 与画布手拉连线同一套兼容规则（含 acceptedKinds 异构通道） */
            const wiringErr = checkWiring(from === 'new' ? newNode : other, fromPort, to === 'new' ? newNode : other, toPort);
            if (wiringErr) {
              invalid = wiringErr;
              break;
            }
            connections.push({ from, fromPort, to, toPort });
          }
        }
        const frameId = args.frameId ? String(args.frameId) : undefined;
        if (!invalid && frameId && !(ctxNow.frames ?? []).some((f) => f.id === frameId && (f.memberIds ?? []).length)) {
          const available = (ctxNow.frames ?? []).filter((f) => (f.memberIds ?? []).length).map((f) => f.id);
          invalid = `没有循环框 ${frameId}；可用：${available.join(', ') || '（图上没有循环框）'}。`;
        }
        if (invalid) {
          result = invalid;
        } else {
          const auto = propose([{
            kind: 'addNode', typeKey, note,
            ...(Object.keys(params).length ? { params } : {}),
            ...(connections.length ? { connections } : {}),
            ...(frameId ? { frameId } : {}),
          }]);
          result = auto
            ? '免确认已开启：新节点（含参数与连线）已自动整部落图。'
            : '已列出加节点卡片（含参数与连线），用户点「应用」才会整部落图。';
        }
      }
    } else if (call.name === 'add_edge') {
      const from = String(args.from ?? '');
      const fromPort = String(args.fromPort ?? '');
      const to = String(args.to ?? '');
      const toPort = String(args.toPort ?? '');
      const note = String(args.note ?? '');
      const ctxNow = ctxRef.current();
      const src = ctxNow.nodes.find((n) => n.id === from);
      const dst = ctxNow.nodes.find((n) => n.id === to);
      let invalid = '';
      if (!src || !dst) {
        invalid = `图上没有节点 ${!src ? from : to}；add_edge 只接两个已有节点，加新节点并随线落图请用 add_node（一端写 "new"）。`;
      } else {
        invalid = checkWiring(src, fromPort, dst, toPort);
        if (!invalid && ctxNow.edges.some((e) => e.from === from && e.fromPort === fromPort && e.to === to && e.toPort === toPort)) {
          invalid = `连线 ${from}:${fromPort} -> ${to}:${toPort} 已存在，无需重复接。`;
        }
      }
      if (invalid) {
        result = invalid;
      } else {
        const auto = propose([{ kind: 'addEdge', from, fromPort, to, toPort, note }]);
        result = auto
          ? '免确认已开启：接线已自动落图；单输入口重连即替换旧线。'
          : '已列出接线卡片，用户点「应用」才会落图；单输入口重连即替换旧线。';
      }
    } else if (call.name === 'flow_control') {
      const op = String(args.op ?? '');
      if (['start', 'pause', 'resume', 'stop'].includes(op)) {
        if (replay) {
          result = '页面刷新中断了控制动作；为避免误启停未自动重执行，请用户用画布上的启停按钮自行控制。';
        } else {
          controlRef.current(op as 'start' | 'pause' | 'resume' | 'stop');
          result = `已对流程执行 ${op}；状态以画布上的运行指示为准。`;
        }
      } else {
        result = `未知操作 ${op}（只接受 start/pause/resume/stop）。`;
      }
    } else if (call.name === 'describe_component') {
      result = describeComponentText(String(args.type ?? ''));
    } else {
      result = `未知页面工具 ${call.name}（本面板未注册它的执行体）。`;
    }
    await resolvePageTool(convId, exec, call.id, result);
  };
  
  /*
   * 引擎回合增量订阅（SSE）：与页面助手面板同构。单流会话——只按会话
   * 过滤（模型只是执行者）；事件直接 patch messages，引擎落盘的消息是
   * 唯一真相，turn-done 时拉单流历史对账兜底；挂起中的页面工具随重连重放。
   */
  useEffect(() => {
    if (!convId) return;
    const conv = convId;
    void fetchPendingPageTools(conv).then((calls) => {
      for (const c of calls) void runPageTool(c, true, c.executor);
    });
    const close = openTurnLive(conv, (ev) => {
      if (ev.conv !== conv) return;
      switch (ev.type) {
        case 'sync':
          setStreaming((ev.models ?? []).length > 0);
          return;
        case 'turn-start':
          setStreaming(true);
          return;
        case 'user': {
          const incoming = ev.message;
          if (!incoming) return;
          setMessages((ms) => {
            const last = ms[ms.length - 1];
            if (last?.role === 'user' && last.text === incoming.content) return ms;
            return [...ms, userMessage(incoming.content)];
          });
          return;
        }
        case 'assistant-start':
          setMessages((ms) => [...ms, { id: messageId(), role: 'assistant', text: '', pending: true }]);
          return;
        case 'delta':
          setMessages((ms) => {
            const last = ms[ms.length - 1];
            if (!last || last.role !== 'assistant') return ms; /* 乱序保护 */
            /* 思维链与正文各自累加：面板的折叠卡据此展示「思考中…」进度 */
            return [
              ...ms.slice(0, -1),
              {
                ...last,
                text: last.text + (ev.content ?? ''),
                reasoning: (last.reasoning ?? '') + (ev.reasoning ?? ''),
              },
            ];
          });
          return;
        case 'page-tool': {
          if (!ev.call) return;
          void runPageTool(ev.call, false, ev.executor);
          return;
        }
        case 'assistant':
        case 'tool':
        case 'error': {
          /*
           * 引擎落盘的完整消息：尾部对齐只替换本轮流式残影（pending）。
           * 不能只看 role——页面工具的卡片消息也是 assistant 且被插在尾部
           * （page-tool 出卡），引擎续轮前的落盘事件若把它当残影替换掉，
           * 卡片就永远不见了（时序竞态时隐时现）。
           */
          const incoming = ev.message;
          if (!incoming) return;
          setMessages((ms) => {
            const disp = toDisplay(incoming);
            const last = ms[ms.length - 1];
            if (last?.role === 'assistant' && last.pending && incoming.role === 'assistant') {
              return [...ms.slice(0, -1), disp];
            }
            return [...ms, disp];
          });
          return;
        }
        case 'turn-done':
          setStreaming(false);
          /* 对账兜底：单流历史拉回重建，本地未决卡片插回（与页面助手同一
           * 语义——引擎桶里只有回执文本，整列重置会抹掉用户还没点「应用」的提议卡） */
          void fetchConvHistory(conv).then((history) => {
            if (!history.length) return;
            setMessages((prev) => reconcileEngineMessages(prev, history, toDisplay));
          });
          return;
      }
    });
    return close;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [convId]);

  /*
   * 引擎回合挂起的审批面（flow_control 等）：轮询拉回渲染——决定走 POST，
   * 引擎恢复回合。与页面助手面板的审批轮询同一模式。
   */
  const [approvals, setApprovals] = useState<
    { id: string; title: string; reason: string; command?: string }[]
  >([]);
  const [engineAlways, setEngineAlways] = useState<Record<string, boolean>>({});
  useEffect(() => {
    if (!convId) {
      setApprovals([]);
      return;
    }
    const conv = convId;
    const tick = async () => {
      try {
        const res = await fetch(`/api/headless/approvals?conv=${encodeURIComponent(conv)}`);
        if (!res.ok) return;
        const r = (await res.json()) as {
          approvals?: { id: string; title: string; reason: string; command?: string }[];
        };
        setApprovals(r.approvals ?? []);
      } catch {
        /* 审批面暂不可达：下一拍重试 */
      }
    };
    void tick();
    const timer = setInterval(() => void tick(), 5000);
    return () => clearInterval(timer);
  }, [convId]);
  const resolveEngineApproval = (id: string, approve: boolean, allowCommand: boolean) => {
    setApprovals((prev) => prev.filter((a) => a.id !== id));
    void fetch(`/api/headless/approvals/${encodeURIComponent(id)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approve, ...(allowCommand ? { alwaysAllow: true } : {}) }),
    }).catch(() => undefined);
  };

  const send = async (text: string) => {
    if (!text || streaming) return;
    const conv = convId || `wf${Date.now().toString(36)}`;
    if (conv !== convId) setConvId(conv);
    const title = text.length > 16 ? `${text.slice(0, 16)}…` : text;
    pushChat(conv, title);
    setMessages((ms) => [...ms, userMessage(text)]);
    /*
     * 无可用模型：回落离线剧本（不提交引擎、不落盘）——画布不空转是编排页
     * 的既有兜底，引擎化的前提是有模型。
     */
    if (!model) {
      setMessages((ms) => [...ms, askAi(text, ctxRef.current(), t)]);
      return;
    }
    setStreaming(true);
    const ctxNow = ctxRef.current();
    const focus = ctxNow.node
      ? `用户当前选中的节点：${ctxNow.node.id}（${NODE_TYPES[ctxNow.node.type]?.label ?? ctxNow.node.type}）。优先回答与它相关的问题。\n\n`
      : '';
    const isNew = conv !== convId;
    const r = await submitTurn(conv, {
      text,
      models: [model],
      system: [`${ASSISTANT_SYSTEM_PROMPT}\n\n${focus}当前图：\n${graphContextText(ctxNow, t)}`],
      tools: [],
      pageTools: FLOW_PAGE_TOOLS,
      /* MS 默认模型不在 models.json：令牌随回合交给引擎（x-swift-api-key 头） */
      ...(apiKeyRef.current ? { apiKey: apiKeyRef.current } : {}),
      /* 单流：历史与模型无关，模型只是下一回合的执行者 */
      singleStream: true,
      /* 新会话把标题一并交给引擎建档（server 侧 chats 注册表与本机同步，
       * 首页会话列表才能看到这段对话） */
      ...(isNew ? { title } : {}),
    });
    if ('error' in r) {
      setStreaming(false);
      setMessages((ms) => [...ms, { id: messageId(), role: 'assistant', text: t('wf.streamFail') }]);
    }
  };

  const stop = () => {
    if (!convId || !model) return;
    void abortTurn(convId, model);
  };

  return {
    messages,
    setMessages,
    streaming,
    send,
    stop,
    convId,
    switchConv,
    approvals,
    engineAlways,
    setEngineAlways,
    resolveEngineApproval,
  } satisfies {
    messages: FlowMessage[];
    setMessages: Dispatch<SetStateAction<FlowMessage[]>>;
    streaming: boolean;
    send: (text: string) => Promise<void>;
    stop: () => void;
    /** 当前会话 id（'' = 未起）：历史会话下拉的选中值 */
    convId: string;
    switchConv: (id: string) => void;
    approvals: { id: string; title: string; reason: string; command?: string }[];
    engineAlways: Record<string, boolean>;
    setEngineAlways: Dispatch<SetStateAction<Record<string, boolean>>>;
    resolveEngineApproval: (id: string, approve: boolean, allowCommand: boolean) => void;
  };
}
