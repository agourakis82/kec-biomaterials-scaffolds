"use client"

import React, { useState } from 'react'
import { Zap, Cpu, MemoryStick, Activity, Play, Pause, BarChart3 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface JAXPerformanceDashboardProps {
  onBenchmarkStart: () => void
  onBatchProcess: (count: number) => void
}

export const JAXPerformanceDashboard: React.FC<JAXPerformanceDashboardProps> = ({
  onBenchmarkStart,
  onBatchProcess
}) => {
  const [isRunning, setIsRunning] = useState(false)
  const [batchSize, setBatchSize] = useState(1000)

  const mockMetrics = {
    speedup: '1,247x',
    throughput: '2.3M ops/s',
    gpuUtilization: 89,
    memoryUsage: 67,
    cpuUsage: 34,
    activeJobs: 12
  }

  const handleBenchmarkToggle = () => {
    setIsRunning(!isRunning)
    if (!isRunning) {
      onBenchmarkStart()
    }
  }

  const handleBatchProcess = () => {
    onBatchProcess(batchSize)
  }

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-yellow-500 via-orange-500 to-red-500 bg-clip-text text-transparent">
          ⚡ JAX Ultra-Performance Monitor
        </h1>
        <p className="text-muted-foreground max-w-3xl mx-auto">
          Monitoramento em tempo real de performance JAX com aceleração GPU/TPU para computação científica
        </p>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-6">
        {[
          { label: 'Speedup', value: mockMetrics.speedup, icon: Zap, color: 'text-yellow-500' },
          { label: 'Throughput', value: mockMetrics.throughput, icon: Activity, color: 'text-green-500' },
          { label: 'GPU Usage', value: `${mockMetrics.gpuUtilization}%`, icon: Cpu, color: 'text-blue-500' },
          { label: 'Memory', value: `${mockMetrics.memoryUsage}%`, icon: MemoryStick, color: 'text-purple-500' },
          { label: 'CPU Usage', value: `${mockMetrics.cpuUsage}%`, icon: Cpu, color: 'text-orange-500' },
          { label: 'Active Jobs', value: mockMetrics.activeJobs.toString(), icon: BarChart3, color: 'text-cyan-500' }
        ].map((metric, index) => (
          <div key={index} className="bg-background/50 border border-border/50 rounded-lg p-6 text-center">
            <div className={cn("mb-2", metric.color)}>
              <metric.icon className="w-6 h-6 mx-auto" />
            </div>
            <div className="text-2xl font-bold mb-1">{metric.value}</div>
            <div className="text-sm text-muted-foreground">{metric.label}</div>
          </div>
        ))}
      </div>

      {/* Control Panel */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Benchmark Control */}
        <div className="bg-background/50 border border-border/50 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Benchmark Control
          </h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Status:</span>
              <span className={cn(
                "px-3 py-1 rounded-full text-sm font-medium",
                isRunning ? "bg-green-500/20 text-green-400" : "bg-gray-500/20 text-gray-400"
              )}>
                {isRunning ? 'Executando' : 'Parado'}
              </span>
            </div>

            <button
              onClick={handleBenchmarkToggle}
              className={cn(
                "w-full flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-medium transition-all",
                isRunning 
                  ? "bg-red-500 hover:bg-red-600 text-white"
                  : "bg-green-500 hover:bg-green-600 text-white"
              )}
            >
              {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isRunning ? 'Parar Benchmark' : 'Iniciar Benchmark'}
            </button>

            <div className="text-xs text-muted-foreground text-center">
              Benchmark de performance com operações matriciais intensivas
            </div>
          </div>
        </div>

        {/* Batch Processing */}
        <div className="bg-background/50 border border-border/50 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Batch Processing
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Batch Size:
              </label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                min="100"
                max="10000"
                step="100"
                className="w-full px-3 py-2 bg-background/50 border border-border/50 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <button
              onClick={handleBatchProcess}
              className="w-full py-3 px-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
            >
              Processar Batch
            </button>

            <div className="text-xs text-muted-foreground text-center">
              Processamento em lote otimizado com JAX
            </div>
          </div>
        </div>
      </div>

      {/* Performance Charts Placeholder */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* GPU Utilization Chart */}
        <div className="bg-background/50 border border-border/50 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">GPU Utilization</h3>
          <div className="h-48 flex items-center justify-center bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg">
            <div className="text-center">
              <Activity className="w-12 h-12 text-blue-500 mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">
                Gráfico de utilização GPU em tempo real
              </p>
              <div className="mt-4 text-2xl font-bold text-blue-500">
                {mockMetrics.gpuUtilization}%
              </div>
            </div>
          </div>
        </div>

        {/* Memory Usage Chart */}
        <div className="bg-background/50 border border-border/50 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Memory Usage</h3>
          <div className="h-48 flex items-center justify-center bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-lg">
            <div className="text-center">
              <MemoryStick className="w-12 h-12 text-purple-500 mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">
                Monitoramento de uso de memória
              </p>
              <div className="mt-4 text-2xl font-bold text-purple-500">
                {mockMetrics.memoryUsage}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* System Information */}
      <div className="bg-background/50 border border-border/50 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">System Information</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-medium mb-2">Hardware</h4>
            <div className="space-y-1 text-sm text-muted-foreground">
              <div>GPU: NVIDIA A100 80GB</div>
              <div>CPU: Intel Xeon 32 cores</div>
              <div>RAM: 256GB DDR4</div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">Software</h4>
            <div className="space-y-1 text-sm text-muted-foreground">
              <div>JAX: 0.4.20</div>
              <div>CUDA: 12.2</div>
              <div>Python: 3.11.5</div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium mb-2">Performance</h4>
            <div className="space-y-1 text-sm text-muted-foreground">
              <div>Peak FLOPS: 312 TFLOPS</div>
              <div>Memory Bandwidth: 2TB/s</div>
              <div>Uptime: 99.9%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default JAXPerformanceDashboard
