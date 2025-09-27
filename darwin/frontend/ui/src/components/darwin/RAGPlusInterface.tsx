"use client"

import React, { useState } from 'react'
import { GrFormSearch, GrCode, GrBook, GrStatusGood, GrRefresh, GrCopy } from 'react-icons/gr'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Badge } from '../ui/badge'
import { ScrollArea } from '../ui/scroll-area'
import * as Select from '@radix-ui/react-select'
import { Separator } from '../ui/separator'
import * as Form from '@radix-ui/react-form'

interface Source {
  id: string
  title: string
  snippet: string
  score: number
  source: string
  url?: string
}

interface RAGResult {
  query: string
  answer: string
  sources: Source[]
  method: string
  retrievedDocs: number
  processingTime: number
}

interface RAGPlusInterfaceProps {
  onQuery: (query: string, type: string) => void
  isLoading?: boolean
  results?: RAGResult
}

export function RAGPlusInterface({ onQuery, isLoading = false, results }: RAGPlusInterfaceProps) {
  const [query, setQuery] = useState('')
  const [searchType, setSearchType] = useState('rag_plus')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onQuery(query.trim(), searchType)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const searchTypes = [
    { value: 'rag_simple', label: 'Simple RAG' },
    { value: 'rag_plus', label: 'RAG++ Enhanced' },
    { value: 'iterative', label: 'Iterative Search' },
    { value: 'discovery', label: 'Scientific Discovery' }
  ]

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-darwin-display font-bold" 
            style={{ color: 'hsl(var(--darwin-primary))' }}>
          RAG++ Enhanced Research
        </h1>
        <p className="text-muted-foreground font-darwin-body">
          Advanced scientific queries with iterative reasoning and context discovery
        </p>
      </div>

      {/* Query Interface */}
      <Card className="darwin-card">
        <CardHeader>
          <CardTitle className="font-darwin-display flex items-center space-x-2">
            <GrFormSearch style={{ color: 'hsl(var(--darwin-primary))' }} />
            <span>Scientific Query Interface</span>
          </CardTitle>
          <CardDescription className="font-darwin-body">
            Enter your research question and select the search method
          </CardDescription>
        </CardHeader>

        <CardContent>
          <Form.Root onSubmit={handleSubmit}>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="query" className="font-darwin-body font-medium">
                  Research Question
                </Label>
                <Input
                  id="query"
                  placeholder="e.g., What are the latest advances in biomaterial scaffold porosity?"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="darwin-input"
                  disabled={isLoading}
                />
              </div>

              <div className="space-y-2">
                <Label className="font-darwin-body font-medium">Search Method</Label>
                <Select.Root value={searchType} onValueChange={setSearchType}>
                  <Select.Trigger className="darwin-input w-full">
                    <Select.Value />
                  </Select.Trigger>
                  <Select.Portal>
                    <Select.Content className="darwin-card border p-2 min-w-[200px]">
                      <Select.Viewport>
                        {searchTypes.map((type) => (
                          <Select.Item
                            key={type.value}
                            value={type.value}
                            className="p-2 hover:bg-muted/50 rounded cursor-pointer font-darwin-body"
                          >
                            <Select.ItemText>{type.label}</Select.ItemText>
                          </Select.Item>
                        ))}
                      </Select.Viewport>
                    </Select.Content>
                  </Select.Portal>
                </Select.Root>
              </div>

              <Button 
                type="submit" 
                className="w-full darwin-button-primary font-darwin-body"
                disabled={isLoading || !query.trim()}
              >
                {isLoading ? (
                  <>
                    <GrRefresh className="mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <GrFormSearch className="mr-2" />
                    Search
                  </>
                )}
              </Button>
            </div>
          </Form.Root>
        </CardContent>
      </Card>

      {/* Results */}
      {results && (
        <div className="space-y-4">
          {/* Answer */}
          <Card className="darwin-card">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="font-darwin-display flex items-center space-x-2">
                  <GrStatusGood style={{ color: 'hsl(var(--darwin-success))' }} />
                  <span>Answer</span>
                </CardTitle>
                <div className="flex items-center space-x-2">
                  <Badge className="darwin-status-success">
                    {results.method.replace('_', ' ').toUpperCase()}
                  </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(results.answer)}
                    className="font-darwin-body"
                  >
                    <GrCopy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="darwin-rag-result">
                <p className="font-darwin-body leading-relaxed">{results.answer}</p>
              </div>
              
              <div className="flex items-center space-x-4 mt-4 text-sm text-muted-foreground font-darwin-mono">
                <span>Retrieved: {results.retrievedDocs} docs</span>
                <span>•</span>
                <span>Time: {results.processingTime}s</span>
              </div>
            </CardContent>
          </Card>

          {/* Sources */}
          <Card className="darwin-card">
            <CardHeader>
              <CardTitle className="font-darwin-display flex items-center space-x-2">
                <GrBook style={{ color: 'hsl(var(--darwin-accent))' }} />
                <span>Sources ({results.sources.length})</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                <div className="space-y-4">
                  {results.sources.map((source, index) => (
                    <div key={source.id} className="border-l-4 border-accent pl-4 py-2">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium font-darwin-body text-sm">
                          {source.title}
                        </h4>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            Score: {(source.score * 100).toFixed(1)}%
                          </Badge>
                          {source.url && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => window.open(source.url, '_blank')}
                              className="h-6 px-2 text-xs"
                            >
                              <GrCode className="h-3 w-3" />
                            </Button>
                          )}
                        </div>
                      </div>
                      
                      <p className="text-sm text-muted-foreground font-darwin-body mb-2">
                        {source.snippet}
                      </p>
                      
                      <p className="text-xs text-muted-foreground font-darwin-mono">
                        Source: {source.source}
                      </p>
                      
                      {index < results.sources.length - 1 && (
                        <Separator className="mt-4" />
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Quick Examples */}
      {!results && (
        <Card className="darwin-card">
          <CardHeader>
            <CardTitle className="font-darwin-display">Example Queries</CardTitle>
            <CardDescription className="font-darwin-body">
              Try these sample research questions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {[
                "What are the optimal porosity ranges for biomaterial scaffolds?",
                "How does surface roughness affect cell adhesion in tissue engineering?",
                "What are the latest advances in 3D bioprinting materials?",
                "Compare mechanical properties of natural vs synthetic scaffolds"
              ].map((example, index) => (
                <Button
                  key={index}
                  variant="ghost"
                  className="w-full justify-start text-left h-auto p-3 font-darwin-body"
                  onClick={() => setQuery(example)}
                >
                  <GrFormSearch className="mr-2 flex-shrink-0" />
                  <span className="text-sm">{example}</span>
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}