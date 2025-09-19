export type DiffToken = { type: "equal" | "ins" | "del"; text: string }

export function diffWords(a: string, b: string): DiffToken[] {
  const aw = a.split(/(\s+)/)
  const bw = b.split(/(\s+)/)
  const n = aw.length
  const m = bw.length
  const dp: number[][] = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0))
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] = aw[i - 1] === bw[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1])
    }
  }
  const out: DiffToken[] = []
  let i = n, j = m
  while (i > 0 && j > 0) {
    if (aw[i - 1] === bw[j - 1]) {
      out.push({ type: "equal", text: aw[i - 1] })
      i--; j--
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      out.push({ type: "del", text: aw[i - 1] })
      i--
    } else {
      out.push({ type: "ins", text: bw[j - 1] })
      j--
    }
  }
  while (i > 0) { out.push({ type: "del", text: aw[--i] }) }
  while (j > 0) { out.push({ type: "ins", text: bw[--j] }) }
  return out.reverse()
}

