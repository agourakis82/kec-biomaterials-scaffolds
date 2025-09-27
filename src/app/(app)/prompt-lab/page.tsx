"use client"

import * as React from "react"
import Image from "next/image"
import { PromptLab } from "@/features/promptgen/PromptLab"

export default function PromptLabPage() {
  return (
    <div className="space-y-6">
      <div className="relative h-48 w-full overflow-hidden rounded-lg border">
        <Image
          src="/assets/charles_darwin.jpg"
          alt="Charles Darwin"
          fill
          className="object-cover"
          priority
        />
        <div className="absolute inset-0 bg-black/40" />
        <div className="absolute bottom-3 left-4 text-white">
          <div className="text-xl font-semibold">Prompt-Lab</div>
          <div className="text-sm opacity-90">Charles Darwin (c. 1868–1871)</div>
        </div>
      </div>

      <PromptLab />
    </div>
  )
}

