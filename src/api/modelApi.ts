import axios, { AxiosError } from 'axios'

export interface DetectedObject {
  label: string
  score: number
  bbox: [number, number, number, number]
  mask: string | number[][]
  mask_rle: string
  mask_size: [number, number]
  keypoints: Keypoint[]
  hand_keypoints: Keypoint[][]
  ocr_text: string
  description: string
  caption: string
}

export interface Keypoint {
  x: number
  y: number
  score: number
}

export type PromptMode = 'universal' | 'text' | 'referring'

export interface AnalyzeMediaResponse {
  objects: DetectedObject[]
  labels: string[]
  feedback: string
  summary: string
  task_id: string
  prompt_mode: PromptMode
  media_type: 'image' | 'video'
  frames: VideoFrameAnnotation[]
  activity: string
}

export interface VideoFrameAnnotation {
  timestamp: number
  objects: DetectedObject[]
}

export type BackendModelStatus = 'ready' | 'loading' | 'model_error' | 'not_loaded'

export interface BackendHealth {
  status: BackendModelStatus
  device: string
  engine: 'openvino' | 'pytorch-cpu'
  grounding_dino_loaded: boolean
  sam2_loaded: boolean
  rtmpose_loaded: boolean
  rtmpose_hand_loaded?: boolean
  model_dir: string
  model_error: string | null
  capabilities?: Record<string, { available: boolean; path: string }>
  model_assets?: Record<string, { available: boolean; checksum: 'verified' | 'unverified' | 'failed' | 'missing'; path: string; source: string; error: string }>
  model_progress?: { checked: number; total: number; available: number; percent: number; phase: 'ready' | 'loading' | 'audit' }
  privacy?: { network_runtime: boolean; cuda: boolean }
  native_chinese_grounding?: boolean
}

interface ErrorResponse {
  detail?: string
}

const modelClient = axios.create({
  // 开发环境直接访问 FastAPI，避免 Vite 代理失败时把后端错误伪装成
  // localhost:5173 的页面请求；生产环境仍可通过 VITE_API_BASE_URL 覆盖。
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  timeout: 10 * 60 * 1000,
  withCredentials: true,
})

export async function getBackendHealth(): Promise<BackendHealth> {
  const response = await modelClient.get<BackendHealth>('/api/health', { timeout: 3000 })
  return response.data
}

/** 健康探针失败分类：只有无法建立连接/超时才提示后端未启动。 */
export function isBackendConnectionError(error: unknown): boolean {
  if (!axios.isAxiosError(error)) return error instanceof TypeError
  const axiosError = error as AxiosError
  return axiosError.code === 'ECONNABORTED'
    || axiosError.code === 'ETIMEDOUT'
    || axiosError.code === 'ERR_NETWORK'
    || !axiosError.response
}

export async function analyzeMedia(
  media: File,
  textPrompt: string,
  promptMode: PromptMode,
  signal?: AbortSignal,
): Promise<AnalyzeMediaResponse> {
  const formData = new FormData()
  formData.append('media', media)
  formData.append('text_prompt', textPrompt)
  formData.append('prompt_mode', promptMode)

  const response = await modelClient.post<AnalyzeMediaResponse>('/api/analyze-media', formData, { signal })
  return response.data
}

export function getApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ErrorResponse>
    if (axiosError.code === 'ERR_CANCELED') {
      return '已取消等待当前推理结果。服务端若已开始计算，会在本次计算结束后释放模型资源。'
    }
    if (axiosError.code === 'ECONNABORTED') {
      return '推理请求超时，请检查后端状态或降低图像分辨率。'
    }
    if (!axiosError.response) {
      return '无法连接本地推理服务，请确认后端已在 8000 端口启动。'
    }
    return axiosError.response.data?.detail || `请求失败（HTTP ${axiosError.response.status}）`
  }
  return error instanceof Error ? error.message : '发生未知错误。'
}
