"use client"

import React, { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Sparkles, Brain } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Message {
  id: string
  type: 'user' | 'agent'
  content: string
  timestamp: Date
  agentName?: string
  agentType?: string
}

interface Agent {
  id: string
  name: string
  type: string
  status: 'active' | 'idle' | 'busy'
  description: string
}

interface ConsciousnessChatProps {
  onAgentSwitch: (agent: Agent) => void
}

const mockAgents: Agent[] = [
  {
    id: '1',
    name: 'Dr. Neural',
    type: 'Neural Networks Specialist',
    status: 'active',
    description: 'Especialista em redes neurais e deep learning'
  },
  {
    id: '2',
    name: 'Prof. Quantum',
    type: 'Quantum Computing Expert',
    status: 'idle',
    description: 'Especialista em computação quântica e algoritmos'
  },
  {
    id: '3',
    name: 'Dr. Bio',
    type: 'Bioinformatics Researcher',
    status: 'busy',
    description: 'Especialista em bioinformática e análise genômica'
  }
]

export const ConsciousnessChat: React.FC<ConsciousnessChatProps> = ({
  onAgentSwitch
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'agent',
      content: 'Olá! Sou Dr. Neural, especialista em redes neurais. Como posso ajudá-lo com sua pesquisa hoje?',
      timestamp: new Date(),
      agentName: 'Dr. Neural',
      agentType: 'Neural Networks Specialist'
    }
  ])
  const [currentMessage, setCurrentMessage] = useState('')
  const [selectedAgent, setSelectedAgent] = useState(mockAgents[0])
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: currentMessage,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setCurrentMessage('')
    setIsTyping(true)

    // Simulate agent response
    setTimeout(() => {
      const agentResponse: Message = {
        id: (Date.now() + 1).toString(),
        type: 'agent',
        content: generateAgentResponse(currentMessage, selectedAgent),
        timestamp: new Date(),
        agentName: selectedAgent.name,
        agentType: selectedAgent.type
      }

      setMessages(prev => [...prev, agentResponse])
      setIsTyping(false)
    }, 1500)
  }

  const generateAgentResponse = (userMessage: string, agent: Agent): string => {
    const responses = {
      'Dr. Neural': [
        'Interessante pergunta sobre redes neurais! Baseado na minha experiência, posso sugerir algumas abordagens...',
        'Essa é uma área fascinante do deep learning. Vamos explorar as possibilidades...',
        'Do ponto de vista de arquiteturas neurais, podemos considerar...'
      ],
      'Prof. Quantum': [
        'Excelente questão sobre computação quântica! Os algoritmos quânticos oferecem vantagens únicas...',
        'Na perspectiva da mecânica quântica, isso envolve conceitos de superposição e entrelaçamento...',
        'Considerando os princípios quânticos, podemos abordar isso através de...'
      ],
      'Dr. Bio': [
        'Muito relevante para bioinformática! Essa questão conecta-se com análise genômica...',
        'Do ponto de vista biológico, precisamos considerar as interações moleculares...',
        'Na análise de proteínas, isso nos leva a considerar...'
      ]
    }

    const agentResponses = responses[agent.name as keyof typeof responses] || responses['Dr. Neural']
    return agentResponses[Math.floor(Math.random() * agentResponses.length)]
  }

  const handleAgentSwitch = (agent: Agent) => {
    setSelectedAgent(agent)
    onAgentSwitch(agent)
    
    // Add system message about agent switch
    const switchMessage: Message = {
      id: Date.now().toString(),
      type: 'agent',
      content: `Olá! Agora você está conversando com ${agent.name}. ${agent.description}. Como posso ajudá-lo?`,
      timestamp: new Date(),
      agentName: agent.name,
      agentType: agent.type
    }
    
    setMessages(prev => [...prev, switchMessage])
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border/50 bg-background/50">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Agent Chat
            </h2>
            <p className="text-sm text-muted-foreground">
              Conversando com {selectedAgent.name}
            </p>
          </div>
          
          {/* Agent Selector */}
          <div className="flex gap-2">
            {mockAgents.map((agent) => (
              <button
                key={agent.id}
                onClick={() => handleAgentSwitch(agent)}
                className={cn(
                  "px-3 py-2 rounded-lg text-sm font-medium transition-all",
                  selectedAgent.id === agent.id
                    ? "bg-purple-500 text-white"
                    : "bg-background/50 border border-border/50 hover:bg-background/80"
                )}
              >
                {agent.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              "flex gap-3",
              message.type === 'user' ? "justify-end" : "justify-start"
            )}
          >
            {message.type === 'agent' && (
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 text-white" />
              </div>
            )}
            
            <div
              className={cn(
                "max-w-[70%] rounded-lg p-4",
                message.type === 'user'
                  ? "bg-purple-500 text-white"
                  : "bg-background/50 border border-border/50"
              )}
            >
              {message.type === 'agent' && (
                <div className="text-xs text-muted-foreground mb-2">
                  {message.agentName} • {message.agentType}
                </div>
              )}
              
              <div className="text-sm">{message.content}</div>
              
              <div className={cn(
                "text-xs mt-2",
                message.type === 'user' ? "text-purple-100" : "text-muted-foreground"
              )}>
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>

            {message.type === 'user' && (
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-green-500 to-blue-500 flex items-center justify-center flex-shrink-0">
                <User className="w-4 h-4 text-white" />
              </div>
            )}
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex gap-3 justify-start">
            <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
              <Bot className="w-4 h-4 text-white" />
            </div>
            
            <div className="bg-background/50 border border-border/50 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                </div>
                <span className="text-sm text-muted-foreground">
                  {selectedAgent.name} está digitando...
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-6 border-t border-border/50 bg-background/50">
        <div className="flex gap-3">
          <textarea
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={`Faça uma pergunta para ${selectedAgent.name}...`}
            className="flex-1 p-3 bg-background/50 border border-border/50 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
            rows={1}
            style={{ minHeight: '44px', maxHeight: '120px' }}
          />
          
          <button
            onClick={handleSendMessage}
            disabled={!currentMessage.trim() || isTyping}
            className={cn(
              "px-4 py-3 rounded-lg font-medium transition-all flex items-center gap-2",
              "bg-purple-500 hover:bg-purple-600 text-white",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        
        <div className="mt-2 text-xs text-muted-foreground text-center">
          Pressione Enter para enviar, Shift+Enter para nova linha
        </div>
      </div>
    </div>
  )
}

export default ConsciousnessChat
