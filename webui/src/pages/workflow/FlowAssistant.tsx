import { useEffect, useState } from 'react';
import { Bot } from 'lucide-react';
import { SelectInput } from '@/components/FormField';
import { EngineApprovalCard } from '@/components/EngineApprovalCard';
import { AiChatPanel } from '@/components/AiChatPanel';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { useChats } from '@/store/chats';
import type { GraphEdge, GraphFrame, GraphNode } from './nodeTypes';
import type { AiAction, FlowMessage } from './aiAssist';
import { describeAction, flowIntro, useFlowAiChat } from './aiAssist';
import { useT } from '@/i18n';

export type RunState = 'idle' | 'running' | 'paused';

/**
 * 整图 AI 助手抽屉（功能 3）。
 *
 * 跟节点面板的区别不在于界面，而在于上下文：这里的 AiContext 不带 node，
 * 所以 AI 回答的是「这张图」的事——启动、暂停、整体为什么慢。回合由
 * server 引擎执行（useFlowAiChat），动作提议（改参数/加节点/启停）经
 * 页面工具挂起协议上抛成本页卡片，用户点「应用」才落图。
 *
 * 手动启停与运行状态都不在这里重复：控制面唯一入口是画布工具栏（运行/
 * 暂停/恢复/停止），进度看画布节点点亮；AI 提议的启停卡应用后也走同一个 control()。
 */
export function FlowAssistant({
  open,
  onClose,
  nodes,
  edges,
  frames,
  presetLabel,
  runState,
  confirmBeforeApply,
  onControl,
  onAction,
  baseConv,
  model,
  apiKey,
  modelOptions,
  onModelChange,
}: {
  open: boolean;
  onClose: () => void;
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** 循环框：进 AI 上下文，助手才能把新阶段提议进环（add_node 的 frameId） */
  frames?: GraphFrame[];
  presetLabel: string;
  runState: RunState;
  confirmBeforeApply: boolean;
  onControl: (op: 'start' | 'pause' | 'resume' | 'stop') => void;
  /** 非控制类提议（改参数、加节点）交给页面处理 */
  onAction: (a: AiAction) => void;
  /** 会话锚点（wf-<flowId>）：单流历史与模型无关，重开画布续上历史 */
  baseConv: string;
  /** 当前模型 id（'' = 无可用模型，发送回落离线剧本）；只是执行者，切模型历史不动 */
  model: string;
  /** MS 默认模型的会话令牌（随回合提交鉴权）；models.json 条目不传 */
  apiKey?: string;
  modelOptions: { value: string; label: string }[];
  onModelChange: (id: string) => void;
}) {
  const [draft, setDraft] = useState('');
  const running = runState === 'running';
  const { t } = useT();
  const chats = useChats();
  const {
    messages, setMessages, streaming, send, stop, convId, switchConv,
    approvals, engineAlways, setEngineAlways, resolveEngineApproval,
  } =
    useFlowAiChat({
      baseConv,
      model,
      apiKey,
      ctx: () => ({ nodes, edges, frames, presetLabel, running }),
      t,
      onControl,
      confirmBeforeApply,
      onAction,
    });

  // 打开时（或切到新会话后）消息为空就给开场白；之后再开保留上次的对话
  useEffect(() => {
    if (open && messages.length === 0) {
      setMessages([flowIntro({ nodes, edges, presetLabel, running }, t)]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, messages.length]);

  const sendNow = () => {
    const text = draft.trim();
    if (!text) return;
    setDraft('');
    void send(text);
  };

  const handleAction = (msgId: string, index: number, next: 'applied' | 'ignored') => {
    const action = messages.find((m) => m.id === msgId)?.actions?.[index];
    if (next === 'applied' && action) {
      if (action.kind === 'control') onControl(action.op);
      else onAction(action);
    }
    setMessages((ms) =>
      ms.map((m) =>
        m.id === msgId ? { ...m, states: m.states?.map((s, i) => (i === index ? next : s)) } : m,
      ),
    );
  };

  return (
    /*
      不盖遮罩、不夺焦点：开着助手还要能看着画布上的节点一个个点亮。
      onInteractOutside 也要拦下来，不然点一下画布抽屉就自己关了。
    */
    <Sheet modal={false} open={open} onOpenChange={(o) => !o && onClose()}>
      <SheetContent
        showOverlay={false}
        className="w-105 gap-0 sm:max-w-none"
        onInteractOutside={(e) => e.preventDefault()}
      >
        <SheetHeader className="px-4 py-3">
          <SheetTitle className="flex items-center gap-2 text-sm">
            <Bot className="text-primary size-4" />
            {t('wf.assistant')}
          </SheetTitle>
        </SheetHeader>

        {/* 模型与会话两枚下拉（与页面助手同布局）：模型只是下一回合的执行者
            （历史与模型无关）；会话默认锚在本实验，可选任一历史会话续聊 */}
        <div className="border-border/60 flex items-center gap-2 border-y px-4 py-2.5">
          <SelectInput
            value={model}
            onChange={onModelChange}
            options={modelOptions}
            placeholder={t('assistant.noModel')}
            disabled={!modelOptions.length || streaming}
            className="text-muted-foreground h-7 flex-1 gap-1.5 truncate border-0 px-2 text-xs shadow-none"
          />
          <SelectInput
            value={convId}
            onChange={switchConv}
            options={[
              { value: '', label: t('assistant.newSession') },
              ...chats.map((c) => ({ value: c.id, label: c.title })),
            ]}
            disabled={streaming}
            className="text-muted-foreground h-7 flex-1 gap-1.5 truncate border-0 px-2 text-xs shadow-none"
          />
        </div>

        <AiChatPanel
          messages={messages}
          describe={(a) => describeAction(a, nodes, t)}
          draft={draft}
          onDraftChange={setDraft}
          onSend={sendNow}
          onStop={streaming ? stop : undefined}
          onAction={handleAction}
          confirmBeforeApply={confirmBeforeApply}
          suggestions={
            running
              ? [t('wf.sugPause'), t('wf.sugSlow'), t('wf.sugOom')]
              : [t('wf.sugRun'), t('wf.sugWhatGraph'), t('wf.sugAdvZero')]
          }
          placeholder={t('wf.flowPlaceholder')}
        />

        {/* 引擎回合挂起的审批（flow_control 等）：决定后引擎恢复回合 */}
        {approvals.map((a) => (
          <EngineApprovalCard
            key={a.id}
            approval={a}
            checked={engineAlways[a.id] === true}
            onCheck={(v) => setEngineAlways((prev) => ({ ...prev, [a.id]: v }))}
            onResolve={resolveEngineApproval}
          />
        ))}
      </SheetContent>
    </Sheet>
  );
}
