"use client"

import * as React from "react"
import Graph from "graphology"
import { SigmaContainer, useLoadGraph, ControlsContainer, ZoomControl } from "@react-sigma/core"
import "@react-sigma/core/lib/react-sigma.min.css"

type TreeNode = {
  id?: string
  value: number
  visits: number
  children?: TreeNode[]
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function heatColor01(x: number) {
  const t = Math.max(0, Math.min(1, x))
  // Blue (#3b82f6) -> Red (#ef4444)
  const b = { r: 59, g: 130, b: 246 }
  const r = { r: 239, g: 68, b: 68 }
  const rr = Math.round(lerp(b.r, r.r, t))
  const gg = Math.round(lerp(b.g, r.g, t))
  const bb = Math.round(lerp(b.b, r.b, t))
  return `rgb(${rr}, ${gg}, ${bb})`
}

function buildGraph(root: TreeNode): Graph {
  const g = new Graph()
  function walk(n: TreeNode, path: string) {
    const id = n.id ?? path
    const size = Math.log((n.visits ?? 0) + 1) * 5 + 2
    const color = heatColor01(n.value ?? 0)
    if (!g.hasNode(id)) {
      g.addNode(id, {
        label: `${id}`,
        size,
        color,
        visits: n.visits ?? 0,
        value: n.value ?? 0,
        x: Math.random() * 100,
        y: Math.random() * 100,
      })
    }
    const children = n.children ?? []
    children.forEach((c, idx) => {
      const cid = c.id ?? `${path}.${idx}`
      if (!g.hasNode(cid)) {
        const csize = Math.log((c.visits ?? 0) + 1) * 5 + 2
        g.addNode(cid, {
          label: `${cid}`,
          size: csize,
          color: heatColor01(c.value ?? 0),
          visits: c.visits ?? 0,
          value: c.value ?? 0,
          x: Math.random() * 100,
          y: Math.random() * 100,
        })
      }
      const eid = `${id}->${cid}`
      if (!g.hasEdge(eid)) g.addEdgeWithKey(eid, id, cid, { size: 1 })
      walk(c, cid)
    })
  }
  walk(root, root.id ?? "root")
  return g
}

function LoadGraph({ data }: { data: { tree: TreeNode } }) {
  const loadGraph = useLoadGraph()
  React.useEffect(() => {
    if (!data?.tree) return
    const gg = buildGraph(data.tree)
    loadGraph(gg)
  }, [data, loadGraph])
  return null
}

export function PuctGraph({ data }: { data: { tree: TreeNode } | null }) {
  if (!data?.tree) return <div className="text-sm text-muted-foreground">Sem dados do PUCT.</div>
  return (
    <SigmaContainer style={{ height: 600, width: "100%" }}>
      <LoadGraph data={data} />
      <ControlsContainer position={"bottom-right"}>
        <ZoomControl />
      </ControlsContainer>
    </SigmaContainer>
  )
}

