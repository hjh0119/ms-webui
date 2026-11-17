/**
 * 新建页候选清单（模型/数据集）：真相源是 swift 的两个注册表——
 * MODEL_MAPPING 展开的全部 hub id 与 DATASET_MAPPING 的数据集 id（后端
 * spawn python 探测一次并缓存，swift 新增模型后重启 webui 即自动出现），
 * 另并入本机 modelscope 缓存里已下载的模型。清单是「捷径不是白名单」——
 * SearchSelect/TagInput 都接受任意自由输入（本地路径、清单外的 hub id）。
 */

import { apiBase } from './base';
import { tStatic } from '../i18n/static';
import type { SearchOption } from '../components/SearchSelect';

/** 一条候选：id 是落进命令行的值，modelType/tags 仅辅助展示 */
export interface CatalogEntry {
  id: string;
  /** 模型：注册表里的 model_type */
  modelType?: string;
  tags: string[];
  /** 注册表里没有、只在本机缓存里的模型 */
  cached?: boolean;
}

export async function listCatalog(): Promise<{ models: CatalogEntry[]; datasets: CatalogEntry[] }> {
  const res = await fetch(apiBase() + '/api/catalog');
  if (!res.ok) throw new Error(tStatic('api.readCatalogFailed', { status: res.status }));
  return await res.json();
}

/*
 * 本机训练 checkpoint（/api/ckpt-catalog）：server 每次请求实时扫
 * ~/.swift/train，删了即消失——与 /api/catalog 的探测缓存职责分开。
 */
export interface CkptEntry {
  /** 断点目录绝对路径（落进 --model / --adapters 的值） */
  path: string;
  taskId: string;
  name: string;
  mtimeMs: number;
  kind: 'lora' | 'full';
  /** 训练时的底座模型（lineage.json）；LoRA 断点要配底座才能用 */
  baseModel?: string;
  tuner?: string;
}

/** 实时清单：失败返回空数组（ckpt 面不可达不拦模型候选，下拉仍可自由输入） */
export async function listCkptCatalog(): Promise<CkptEntry[]> {
  try {
    const res = await fetch(apiBase() + '/api/ckpt-catalog');
    if (!res.ok) return [];
    return ((await res.json()) as { checkpoints?: CkptEntry[] }).checkpoints ?? [];
  } catch {
    return [];
  }
}

/** ckpt 条目 -> 搜索下拉候选（value 是目录路径，label 短名，hint 带任务与类型） */
export function ckptOption(c: CkptEntry): SearchOption {
  const short = (c.baseModel ?? c.taskId).split('/').pop() || c.taskId;
  return {
    value: c.path,
    label: `${short}:${c.name}`,
    hint: `${c.taskId} · ${tStatic(c.kind === 'lora' ? 'api.ckptLora' : 'api.ckptFull')}`,
  };
}

/*
 * 选中值命中本机 ckpt 时的表单落点（四页共用）：full -> model 填路径；
 * lora -> model 填底座、adapters 填路径。未命中返回 null，页面走原逻辑。
 */
export function ckptPick(map: Record<string, CkptEntry>, value: string): { model: string; adapters?: string } | null {
  const c = map[value];
  if (!c) return null;
  if (c.kind === 'full') return { model: c.path };
  return { model: c.baseModel ?? '', adapters: c.path };
}

/** 模型条目 -> 搜索下拉候选（hint 显示 model_type 与能力标签） */
export function modelOption(m: CatalogEntry): { value: string; hint?: string } {
  if (m.cached) return { value: m.id, hint: tStatic('api.cached') };
  const hint = [m.modelType, ...m.tags.slice(0, 2)].filter(Boolean).join(' · ');
  return { value: m.id, ...(hint ? { hint } : {}) };
}

/** 数据集条目 -> 搜索下拉候选（hint 显示标签，🔥 即 legacy docs 标的重要数据集） */
export function datasetOption(d: CatalogEntry): { value: string; hint?: string } {
  const hint = d.tags.slice(0, 3).join(' · ');
  return { value: d.id, ...(hint ? { hint } : {}) };
}
