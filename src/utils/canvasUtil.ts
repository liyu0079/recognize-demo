export type RgbColor = readonly [number, number, number]

export interface Size {
  width: number
  height: number
}

export interface Point {
  x: number
  y: number
}

export function getContainSize(source: Size, viewport: Size, padding = 48): Size {
  const usableWidth = Math.max(1, viewport.width - padding * 2)
  const usableHeight = Math.max(1, viewport.height - padding * 2)
  const scale = Math.min(usableWidth / source.width, usableHeight / source.height, 1)
  return {
    width: Math.max(1, Math.round(source.width * scale)),
    height: Math.max(1, Math.round(source.height * scale)),
  }
}

export function imagePointToCanvas(point: Point, image: Size, canvas: Size): Point {
  return {
    x: point.x * (canvas.width / image.width),
    y: point.y * (canvas.height / image.height),
  }
}

export function canvasPointToImage(point: Point, image: Size, canvas: Size): Point {
  return {
    x: point.x * (image.width / canvas.width),
    y: point.y * (image.height / canvas.height),
  }
}

/** Expands the backend's row-major binary RLE only when a mask layer is needed. */
export function decodeMask(mask: string | number[][], rle: string, size: readonly number[]): number[][] {
  if (Array.isArray(mask) && mask.length) return mask
  if (typeof mask === 'string' && mask) rle = mask
  const [height, width] = size
  if (!rle || !Number.isInteger(height) || !Number.isInteger(width) || height <= 0 || width <= 0) return []
  const total = height * width
  const flat = new Uint8Array(total)
  let offset = 0
  let value = 0
  for (const token of rle.split(',')) {
    const run = Number(token)
    if (!Number.isInteger(run) || run < 0 || offset + run > total) return []
    if (value) flat.fill(1, offset, offset + run)
    offset += run
    value = value ? 0 : 1
  }
  if (offset !== total) return []
  return Array.from({ length: height }, (_, y) => Array.from(flat.subarray(y * width, (y + 1) * width)))
}

/** 将后端二维 0/1 矩阵一次转换为可复用的透明彩色图层。 */
export function createMaskCanvas(mask: number[][], color: RgbColor, alpha = 0.42): HTMLCanvasElement {
  const height = mask.length
  const width = mask[0]?.length ?? 0
  if (!width || !height) {
    throw new Error('后端返回了空的分割掩码。')
  }

  const layer = document.createElement('canvas')
  layer.width = width
  layer.height = height
  const context = layer.getContext('2d')
  if (!context) {
    throw new Error('浏览器无法创建掩码画布。')
  }

  const imageData = context.createImageData(width, height)
  const pixels = imageData.data
  const opacity = Math.round(255 * alpha)
  for (let y = 0; y < height; y += 1) {
    const row = mask[y]
    if (!row || row.length !== width) {
      throw new Error('后端返回的分割掩码尺寸不一致。')
    }
    for (let x = 0; x < width; x += 1) {
      if (!row[x]) continue
      const offset = (y * width + x) * 4
      pixels[offset] = color[0]
      pixels[offset + 1] = color[1]
      pixels[offset + 2] = color[2]
      pixels[offset + 3] = opacity
    }
  }
  context.putImageData(imageData, 0, 0)
  return layer
}

/** 将掩码轮廓预处理为独立图层，避免每次重绘时重复扫描完整掩码。 */
export function createMaskEdgeCanvas(
  mask: number[][],
  color: RgbColor,
  alpha = 0.96,
): HTMLCanvasElement {
  const height = mask.length
  const width = mask[0]?.length ?? 0
  if (!width || !height) throw new Error('后端返回了空的分割掩码。')

  const layer = document.createElement('canvas')
  layer.width = width
  layer.height = height
  const context = layer.getContext('2d')
  if (!context) throw new Error('浏览器无法创建掩码边界图层。')

  const imageData = context.createImageData(width, height)
  const pixels = imageData.data
  const opacity = Math.round(255 * alpha)
  for (let y = 0; y < height; y += 1) {
    const row = mask[y]
    if (!row || row.length !== width) throw new Error('后端返回的分割掩码尺寸不一致。')
    for (let x = 0; x < width; x += 1) {
      if (!row[x]) continue
      const isEdge = x === 0 || y === 0 || x === width - 1 || y === height - 1
        || !mask[y][x - 1] || !mask[y][x + 1] || !mask[y - 1]?.[x] || !mask[y + 1]?.[x]
      if (!isEdge) continue
      const offset = (y * width + x) * 4
      pixels[offset] = color[0]
      pixels[offset + 1] = color[1]
      pixels[offset + 2] = color[2]
      pixels[offset + 3] = opacity
    }
  }
  context.putImageData(imageData, 0, 0)
  return layer
}

export function drawMaskLayer(
  context: CanvasRenderingContext2D,
  layer: HTMLCanvasElement,
  target: Size,
  emphasized = false,
  opacity = 0.82,
): void {
  context.save()
  context.globalAlpha = emphasized ? 1 : opacity
  context.imageSmoothingEnabled = false
  context.drawImage(layer, 0, 0, target.width, target.height)
  context.restore()
}
