"use client"

import * as React from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import { Brain, Rocket, Cloud, Zap, BookOpen, Code, HeartPulse, CheckCircle, XCircle, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { useActiveProfile } from "@/hooks/useActiveProfile"
import { api } from "@/lib/api" // Importar o cliente de API

interface HealthStatus {
  status: string;
  message?: string;
  latency?: number;
}

export default function HomePage() {
  const [backendHealth, setBackendHealth] = React.useState<HealthStatus | null>(null);
  const [researchTeamHealth, setResearchTeamHealth] = React.useState<HealthStatus | null>(null);
  const [loadingHealth, setLoadingHealth] = React.useState(true);
  const [errorHealth, setErrorHealth] = React.useState<string | null>(null);

  const profile = useActiveProfile(); // Mantido para compatibilidade, mas não usado diretamente nesta landing page

  React.useEffect(() => {
    const fetchHealthStatus = async () => {
      setLoadingHealth(true);
      setErrorHealth(null);
      try {
        const startGlobal = Date.now();
        const globalStatus = await api.getHealth();
        const endGlobal = Date.now();
        setBackendHealth({ ...globalStatus, latency: endGlobal - startGlobal });

        const startResearch = Date.now();
        const researchStatus = await api.getResearchTeamHealth();
        const endResearch = Date.now();
        setResearchTeamHealth({ ...researchStatus, latency: endResearch - startResearch });
      } catch (e: any) {
        console.error("Erro ao buscar status de saúde:", e);
        setErrorHealth("Não foi possível conectar ao backend. Tente novamente mais tarde.");
        setBackendHealth({ status: "offline", message: e.message });
        setResearchTeamHealth({ status: "offline", message: e.message });
      } finally {
        setLoadingHealth(false);
      }
    };

    fetchHealthStatus();
    const interval = setInterval(fetchHealthStatus, 30000); // Atualiza a cada 30 segundos
    return () => clearInterval(interval);
  }, []);

  const renderHealthStatus = (status: HealthStatus | null, title: string) => {
    if (!status) return null;

    const isOnline = status.status === "online" || status.status === "healthy";
    const Icon = isOnline ? CheckCircle : XCircle;
    const colorClass = isOnline ? "text-green-500" : "text-red-500";
    const badgeVariant = isOnline ? "default" : "destructive";

    return (
      <Card className="flex-1">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">{title}</CardTitle>
          <Icon className={`h-4 w-4 ${colorClass}`} />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold flex items-center gap-2">
            <Badge variant={badgeVariant}>{status.status.toUpperCase()}</Badge>
          </div>
          {status.latency !== undefined && (
            <p className="text-xs text-muted-foreground">Latência: {status.latency}ms</p>
          )}
          {status.message && (
            <p className="text-xs text-muted-foreground mt-1">{status.message}</p>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-16 py-12 md:py-24">
      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-8 px-4"
      >
        <motion.div
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium"
        >
          <Brain className="h-4 w-4" />
          DARWIN AI · Plataforma de Pesquisa Inteligente
        </motion.div>
        <h1 className="text-4xl md:text-6xl font-bold text-gradient leading-tight">
          Acelere sua Descoberta Científica com IA
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          DARWIN é a sua plataforma de IA para orquestração de agentes, pesquisa avançada e análise de dados em escala.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4">
          <Button asChild size="lg" className="px-6 shadow-glow">
            <Link href="/prompt-lab" className="inline-flex items-center gap-2">
              <Rocket className="h-4 w-4" />
              Testar Colaboração
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg" className="px-6">
            <a href="https://docs.agourakis.med.br" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              Ver Documentação
            </a>
          </Button>
          <Button asChild variant="ghost" size="lg" className="px-6">
            <a href="https://api.agourakis.med.br/openapi.json" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2">
              <Code className="h-4 w-4" />
              API OpenAPI
            </a>
          </Button>
        </div>
      </motion.section>

      {/* Features Section */}
      <section className="max-w-6xl mx-auto px-4 space-y-12">
        <h2 className="text-3xl md:text-4xl font-bold text-center">Recursos Principais</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <Card className="hover-lift">
            <CardHeader>
              <Brain className="h-8 w-8 text-primary mb-2" />
              <CardTitle>Agentes Orquestrados</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Utilize o poder de agentes autônomos com AutoGen e GroupChat para resolver problemas complexos de pesquisa.
              </p>
            </CardContent>
          </Card>
          <Card className="hover-lift">
            <CardHeader>
              <Cloud className="h-8 w-8 text-primary mb-2" />
              <CardTitle>Integrações Nativas</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Conecte-se facilmente com Vertex AI para modelos avançados e BigQuery para análise de dados em larga escala.
              </p>
            </CardContent>
          </Card>
          <Card className="hover-lift">
            <CardHeader>
              <Zap className="h-8 w-8 text-primary mb-2" />
              <CardTitle>JAX Ultra-Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Experimente o desempenho inigualável do JAX para computação numérica e machine learning de alta velocidade.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* System Status Section */}
      <section className="max-w-6xl mx-auto px-4 space-y-8">
        <h2 className="text-3xl md:text-4xl font-bold text-center">Status do Sistema</h2>
        {loadingHealth ? (
          <div className="grid md:grid-cols-2 gap-4">
            <Card className="flex-1">
              <CardHeader><Skeleton className="h-5 w-3/4" /></CardHeader>
              <CardContent><Skeleton className="h-8 w-1/2" /></CardContent>
            </Card>
            <Card className="flex-1">
              <CardHeader><Skeleton className="h-5 w-3/4" /></CardHeader>
              <CardContent><Skeleton className="h-8 w-1/2" /></CardContent>
            </Card>
          </div>
        ) : errorHealth ? (
          <Card className="border-destructive/50 bg-destructive/5">
            <CardContent className="p-6 text-destructive flex items-center gap-2">
              <XCircle className="h-5 w-5" />
              <span>{errorHealth}</span>
            </CardContent>
          </Card>
        ) : (
          <div className="grid md:grid-cols-2 gap-4">
            {renderHealthStatus(backendHealth, "Status do Backend (Global)")}
            {renderHealthStatus(researchTeamHealth, "Status da Equipe de Pesquisa (AutoGen)")}
          </div>
        )}
        <p className="text-center text-sm text-muted-foreground">
          Última atualização: {new Date().toLocaleTimeString()}
        </p>
      </section>

      {/* Call to Action - Secondary */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="text-center space-y-6 px-4"
      >
        <h2 className="text-3xl md:text-4xl font-bold">Pronto para Transformar sua Pesquisa?</h2>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Junte-se a pesquisadores que estão utilizando DARWIN para inovar e acelerar suas descobertas.
        </p>
        <Button asChild size="lg" className="px-8 shadow-glow">
          <Link href="/prompt-lab" className="inline-flex items-center gap-2">
            <Rocket className="h-5 w-5" />
            Começar Agora
          </Link>
        </Button>
      </motion.section>
    </div>
  );
}
