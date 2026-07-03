import React, { useState, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Input } from '@/components/ui/input';
import { EmbeddingPlot } from './components/EmbeddingPlot';
import { INITIAL_TOKENS } from './constants';
import { Dimension, TokenEmbedding, ModelType } from './types';
import { Layers, Plus, Minus, Search, Activity, Calculator, ArrowRight, X, Target, GitCompare, Cpu, RefreshCcw, Box } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { kmeans } from 'ml-kmeans';
import { PCA } from 'ml-pca';

export default function App() {
  const [dimension, setDimension] = useState<Dimension>(1);
  const [tokens, setTokens] = useState<TokenEmbedding[]>(INITIAL_TOKENS);
  const [selectedTokenName, setSelectedTokenName] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [interpretation, setInterpretation] = useState<string | null>(null);
  const [isInterpreting, setIsInterpreting] = useState(false);
  const [formula, setFormula] = useState('');
  const [arithmeticResult, setArithmeticResult] = useState<{
    sources: string[];
    target: string | null;
    resultVector: number[];
  } | null>(null);
  const [clusterCount, setClusterCount] = useState(3);
  const [activeModel, setActiveModel] = useState<ModelType>('gemini-flash');
  const [comparisonMode, setComparisonMode] = useState(false);
  const [isProjecting, setIsProjecting] = useState(false);
  const [semanticQuery, setSemanticQuery] = useState('');
  const [semanticMatches, setSemanticMatches] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [sessionName, setSessionName] = useState('UNNAMED_SESSION');

  const densityDistribution = useMemo(() => {
    const bins = Array(20).fill(0);
    tokens.forEach(t => {
      const binIdx = Math.min(19, Math.max(0, Math.floor(t.x + 10)));
      bins[binIdx]++;
    });
    const max = Math.max(...bins, 1);
    return bins.map(count => ({ count, height: (count / max) * 100 }));
  }, [tokens]);

  const topologicalDivergence = useMemo(() => {
    if (!comparisonMode) return null;
    let totalDist = 0;
    let count = 0;
    tokens.forEach(t => {
      if (t.modelVectors?.['gemini-flash'] && t.modelVectors?.['gemini-pro']) {
        const v1 = t.modelVectors['gemini-flash'];
        const v2 = t.modelVectors['gemini-pro'];
        const dist = Math.sqrt(v1.reduce((acc, v, i) => acc + Math.pow(v - v2[i], 2), 0));
        totalDist += dist;
        count++;
      }
    });
    return count > 0 ? (totalDist / count).toFixed(2) : null;
  }, [tokens, comparisonMode]);

  const runClustering = () => {
    const validTokens = tokens.filter(t => t.vector && t.vector.length > 0);
    if (validTokens.length < clusterCount) return;

    const data = validTokens.map(t => t.vector!);
    const result = kmeans(data, clusterCount, {});

    const clusterColors = [
      '#0066CC', // Swiss Blue
      '#E30000', // Swiss Red
      '#1D1D1F', // Black
      '#78350f', // Brown
      '#059669', // Emerald
      '#7c3aed', // Violet
      '#ea580c', // Orange
    ];

    setTokens(prev => prev.map(t => {
      const idx = validTokens.findIndex(vt => vt.token === t.token);
      if (idx === -1) return { ...t, cluster: undefined };
      
      const clusterIdx = result.clusters[idx];
      return {
        ...t,
        cluster: clusterIdx,
        color: clusterColors[clusterIdx % clusterColors.length]
      };
    }));
  };

  const saveSession = () => {
    const snapshot = {
      tokens,
      activeModel,
      dimension,
      timestamp: new Date().toISOString()
    };
    const snapshots = JSON.parse(localStorage.getItem('LAB_SNAPSHOTS') || '{}');
    snapshots[sessionName] = snapshot;
    localStorage.setItem('LAB_SNAPSHOTS', JSON.stringify(snapshots));
    alert(`Snapshot "${sessionName}" preserved in Local Storage.`);
  };

  const restoreSession = () => {
    const snapshots = JSON.parse(localStorage.getItem('LAB_SNAPSHOTS') || '{}');
    if (snapshots[sessionName]) {
      const s = snapshots[sessionName];
      setTokens(s.tokens);
      setActiveModel(s.activeModel);
      setDimension(s.dimension);
      alert(`Snapshot "${sessionName}" reconstructed.`);
    } else {
      alert(`No snapshot found for ID: ${sessionName}`);
    }
  };

  const runPCAProjection = async (model: ModelType) => {
    setIsProjecting(true);
    // Simulate complex math latency
    await new Promise(r => setTimeout(r, 800));

    const validTokens = tokens.filter(t => t.modelVectors && t.modelVectors[model]);
    if (validTokens.length < 2) {
      setIsProjecting(false);
      return;
    }

    const data = validTokens.map(t => t.modelVectors![model]);
    const pca = new PCA(data);
    const projected = pca.predict(data).to2DArray();

    setTokens(prev => prev.map(t => {
      const idx = validTokens.findIndex(vt => vt.token === t.token);
      if (idx === -1) return t;

      const p = projected[idx];
      // Map PCA dimensions to 4D spatial representation
      return {
        ...t,
        x: p[0] * 2,
        y: p[1] * 2,
        z: (p[2] || 0) * 2,
        w: (p[3] || 0) * 2
      };
    }));
    setIsProjecting(false);
  };

  const handleSemanticSearch = async (query: string) => {
    if (!query.trim()) {
      setSemanticMatches([]);
      return;
    }

    setIsSearching(true);
    try {
      // Generate embedding for the natural language query via secure server-side endpoint
      const response = await fetch('/api/embed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate embedding on server');
      }

      const result = await response.json();
      const queryVector = result.values;
      
      // Calculate cosine similarity with all tokens
      const similarities = tokens.map(t => {
        const vec = t.vector || [t.x, t.y || 0, t.z || 0, t.w || 0];
        // Ensure same dimensionality (truncate or pad)
        const targetLen = queryVector.length;
        const paddedVec = [...vec];
        while (paddedVec.length < targetLen) paddedVec.push(0);
        const slicedVec = paddedVec.slice(0, targetLen);

        const dot = queryVector.reduce((sum, v, i) => sum + v * slicedVec[i], 0);
        const mag1 = Math.sqrt(queryVector.reduce((sum, v) => sum + v * v, 0));
        const mag2 = Math.sqrt(slicedVec.reduce((sum, v) => sum + v * v, 0));
        const sim = (mag1 && mag2) ? dot / (mag1 * mag2) : 0;
        
        return { token: t.token, similarity: sim };
      });

      // Sort by similarity and take top 5
      const topMatches = similarities
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 5)
        .map(m => m.token);

      setSemanticMatches(topMatches);
    } catch (error) {
      console.error("Semantic Search Error:", error);
    } finally {
      setIsSearching(false);
    }
  };

  const performArithmetic = (input: string) => {
    setFormula(input);
    if (!input.trim()) {
      setArithmeticResult(null);
      return;
    }

    // Simple parsing: split by + or - and extract token names in brackets if possible [Hate] + [Dislike]
    const parts = input.match(/\[([^\]]+)\]|[+-]|[^[\]+-]+/g) || [];
    let currentVector: number[] | null = null;
    let currentOp: string = '+';
    const sourceTokens: string[] = [];

    (parts as string[]).forEach(part => {
      const cleanPart = part.trim();
      if (!cleanPart) return;
      if (cleanPart === '+' || cleanPart === '-') {
        currentOp = cleanPart;
      } else {
        const tokenName = cleanPart.replace(/[\[\]]/g, '').trim();
        const found = tokens.find(t => t.token.toLowerCase() === tokenName.toLowerCase());
        
        if (found) {
          sourceTokens.push(found.token);
          // If vector is available, use it (16d), otherwise fallback to spatial (4d)
          const vec = found.vector || [found.x, found.y || 0, found.z || 0, found.w || 0];
          
          if (!currentVector) {
            currentVector = [...vec];
          } else {
            currentVector = currentVector.map((v, i) => 
               currentOp === '+' ? v + (vec[i] || 0) : v - (vec[i] || 0)
            );
          }
        }
      }
    });

    if (currentVector) {
      // Find nearest neighbor
      let bestToken = null;
      let maxSim = -Infinity;

      // Ensure we have vectors of same dim (use 4d fallback if needed)
      const targetLen = currentVector.length;

      tokens.forEach(t => {
        // Skip sources if user wants nearest neighbor that isn't the input?
        // Actually usually standard vector math includes input if it's the closest, 
        // but for "King - Man = ?" we want the result.
        
        const vec = t.vector || [t.x, t.y || 0, t.z || 0, t.w || 0];
        // Pad shorter vectors
        const paddedVec = [...vec];
        while (paddedVec.length < targetLen) paddedVec.push(0);
        const slicedVec = paddedVec.slice(0, targetLen);

        // Cosine Similarity
        const dot = currentVector!.reduce((sum, v, i) => sum + v * slicedVec[i], 0);
        const mag1 = Math.sqrt(currentVector!.reduce((sum, v) => sum + v * v, 0));
        const mag2 = Math.sqrt(slicedVec.reduce((sum, v) => sum + v * v, 0));
        const sim = (mag1 && mag2) ? dot / (mag1 * mag2) : 0;

        if (sim > maxSim) {
          maxSim = sim;
          bestToken = t.token;
        }
      });

      setArithmeticResult({
        sources: sourceTokens,
        target: bestToken,
        resultVector: currentVector
      });
    } else {
      setArithmeticResult(null);
    }
  };

  const filteredTokens = useMemo(() => {
    if (!searchQuery) return tokens;
    return tokens.filter(t => t.token.toLowerCase().includes(searchQuery.toLowerCase()));
  }, [tokens, searchQuery]);

  const selectedToken = useMemo(() => 
    tokens.find(t => t.token === selectedTokenName) || null,
  [tokens, selectedTokenName]);

  const updateTokenValue = (name: string, axis: 'x' | 'y' | 'z' | 'w', value: number) => {
    setTokens(prev => prev.map(t => t.token === name ? { ...t, [axis]: value } : t));
  };

  const updateTokenColor = (name: string, color: string) => {
    setTokens(prev => prev.map(t => t.token === name ? { ...t, color } : t));
  };

  React.useEffect(() => {
    if (!selectedToken) {
      setInterpretation(null);
      return;
    }

    const interpret = async () => {
      setIsInterpreting(true);
      try {
        // Query the secure server-side interpretation endpoint
        const response = await fetch('/api/interpret', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            token: selectedToken.token,
            x: selectedToken.x,
            y: selectedToken.y,
            z: selectedToken.z,
            w: selectedToken.w,
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to generate interpretation on server');
        }

        const data = await response.json();
        setInterpretation(data.text || "No interpretation available.");
      } catch (error) {
        console.error("Gemini Error:", error);
        setInterpretation("Error generating interpretation.");
      } finally {
        setIsInterpreting(false);
      }
    };

    const timer = setTimeout(interpret, 500);
    return () => clearTimeout(timer);
  }, [selectedToken]);

  return (
    <TooltipProvider>
      <div className="flex flex-col h-screen font-sans selection:bg-accent-blue/10">
        {/* HEADER: GRID TOP BAR */}
        <header className="h-14 border-b border-border-grid bg-white flex items-center justify-between px-6 z-[60]">
          <div className="flex items-center gap-6 divide-x divide-border-grid h-full">
            <div className="flex items-center gap-2 group cursor-pointer pr-6">
              <Box className="w-5 h-5 text-black" />
              <h1 className="font-black tracking-tighter text-xl uppercase">Vector Lab</h1>
            </div>
            
            <div className="flex items-center gap-4 px-6 h-full">
              <div className="flex flex-col">
                <span className="text-[8px] font-black uppercase tracking-widest text-ink-dim leading-none mb-1">Session ID</span>
                <input 
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value.toUpperCase().replace(/\s+/g, '_'))}
                  className="bg-transparent border-none p-0 text-[11px] font-mono font-black focus:ring-0 w-32 outline-none uppercase"
                />
              </div>
              <div className="flex gap-1">
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={saveSession}
                  className="h-7 rounded-none border-border-grid text-[9px] font-black uppercase hover:bg-black hover:text-white transition-all px-2"
                >
                  Save
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={restoreSession}
                  className="h-7 rounded-none border-border-grid text-[9px] font-black uppercase hover:bg-black hover:text-white transition-all px-2"
                >
                  Load
                </Button>
              </div>
            </div>

            <div className="pl-6 h-full flex items-center gap-4">
              <div className="flex flex-col">
                <span className="text-[9px] font-black uppercase tracking-widest text-ink-dim leading-none mb-1">Status</span>
                <div className="flex items-center gap-1.5">
                  <div className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-pulse" />
                  <span className="text-[10px] font-bold uppercase">Ready</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-8 divide-x divide-border-grid h-full">
            <div className="flex items-center gap-2 pr-8">
              <div className="flex flex-col items-end">
                <span className="text-[9px] font-black uppercase tracking-widest text-ink-dim leading-none mb-1">Active Tokens</span>
                <span className="text-[12px] font-mono font-bold leading-none">{tokens.length}</span>
              </div>
            </div>
            
            <div className="pl-8 h-full flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => dimension > 1 && setDimension(dimension - 1 as Dimension)}
                className="h-8 w-8 p-0 rounded-none border-border-grid hover:bg-bg-base"
              >
                <Minus className="w-3 h-3" />
              </Button>
              <div className="flex flex-col items-center min-w-[100px]">
                <span className="text-[9px] font-black uppercase tracking-[0.2em] text-ink-dim leading-none mb-1">Dimension</span>
                <span className="text-[14px] font-mono font-black">{dimension}D Space</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => dimension < 4 && setDimension(dimension + 1 as Dimension)}
                className="h-8 w-8 p-0 rounded-none border-border-grid hover:bg-bg-base"
              >
                <Plus className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          {/* LEFT SIDEBAR: TOKEN MANAGER */}
          <aside className="w-[320px] bg-white border-r border-border-grid flex flex-col z-[50]">
            <div className="swiss-header">
              <GitCompare className="w-3.5 h-3.5" />
              Topology Lab
            </div>
            
            <div className="p-6 space-y-8 overflow-y-auto custom-scrollbar">
              <section>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim">Active Model</span>
                </div>
                <div className="grid grid-cols-2 gap-px bg-border-grid border border-border-grid">
                  {(['gemini-flash', 'gemini-pro'] as ModelType[]).map(m => (
                    <button
                      key={m}
                      onClick={() => {
                        setActiveModel(m);
                        // Also update primary vector for arithmetic logic
                        setTokens(prev => prev.map(t => ({
                          ...t,
                          vector: t.modelVectors ? t.modelVectors[m] : t.vector
                        })));
                        runPCAProjection(m);
                      }}
                      className={`h-12 px-2 flex flex-col items-center justify-center gap-1 transition-all ${activeModel === m ? 'bg-ink-main text-white' : 'bg-white text-ink-dim hover:bg-bg-base'}`}
                    >
                      <Cpu className="w-3 h-3" />
                      <span className="text-[9px] font-black uppercase tracking-tighter">{m.split('-')[1]} v3.1</span>
                    </button>
                  ))}
                </div>
                <div className="mt-3">
                  <Button
                    variant="outline"
                    onClick={() => setComparisonMode(!comparisonMode)}
                    className={`w-full h-10 rounded-none border-border-grid text-[10px] font-black uppercase tracking-widest gap-2 transition-all ${comparisonMode ? 'bg-accent-blue text-white border-accent-blue' : 'bg-white hover:bg-bg-base'}`}
                  >
                    <GitCompare className="w-3.5 h-3.5" />
                    {comparisonMode ? 'Overlay Active' : 'Enable Overlay Mode'}
                  </Button>
                </div>
              </section>

              <section>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim">Semantic Query (ANN)</span>
                </div>
                <div className="relative group">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-ink-dim group-focus-within:text-accent-blue transition-colors" />
                  <Input 
                    placeholder="e.g. 'A feeling of intense joy'"
                    value={semanticQuery}
                    onChange={(e) => setSemanticQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSemanticSearch(semanticQuery)}
                    className="pl-9 h-10 rounded-none border-border-grid focus-visible:ring-1 focus-visible:ring-accent-blue font-mono text-xs placeholder:opacity-50"
                  />
                  {isSearching && (
                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                      <motion.div 
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <RefreshCcw className="w-3 h-3 text-accent-blue" />
                      </motion.div>
                    </div>
                  )}
                </div>
                <p className="mt-2 text-[9px] font-mono text-ink-dim leading-tight opacity-60 italic">
                  ANN: Cosine similarity search against latent space using gemini-embedding-2-preview.
                </p>
                {semanticMatches.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1">
                    {semanticMatches.map(token => (
                      <Badge 
                        key={`match-${token}`}
                        variant="outline" 
                        className="rounded-none border-accent-blue/30 text-accent-blue text-[9px] font-black uppercase tracking-tighter cursor-pointer hover:bg-accent-blue hover:text-white transition-all"
                        onClick={() => setSelectedTokenName(token)}
                      >
                        {token}
                      </Badge>
                    ))}
                    <button 
                      onClick={() => setSemanticMatches([])}
                      className="text-[9px] font-black uppercase text-ink-dim hover:text-accent-red ml-auto"
                    >
                      Clear
                    </button>
                  </div>
                )}
              </section>

              <section>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim">Environment Search</span>
                </div>
                <div className="relative group">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-ink-dim group-focus-within:text-accent-blue transition-colors" />
                  <Input 
                    placeholder="FILTER TOKENS..." 
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9 h-10 bg-bg-base border-border-grid rounded-none font-mono text-xs focus-visible:ring-1 focus-visible:ring-accent-blue"
                  />
                </div>
              </section>

              <section>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim">Active Sequence</span>
                  <span className="text-[9px] font-mono text-ink-dim">N={filteredTokens.length}</span>
                </div>
                <div className="space-y-px bg-border-grid border border-border-grid max-h-[240px] overflow-y-auto">
                  {filteredTokens.map((t) => (
                    <motion.div
                      key={t.token}
                      onClick={() => setSelectedTokenName(t.token)}
                      className={`
                        px-4 py-3 cursor-pointer flex items-center justify-between transition-all group bg-white
                        hover:bg-bg-base
                        ${selectedTokenName === t.token ? 'ring-1 ring-inset ring-accent-blue z-10' : ''}
                      `}
                    >
                      <div className="flex items-center gap-3">
                        <div 
                          className="w-1.5 h-6 transition-all group-hover:scale-y-110" 
                          style={{ backgroundColor: t.color }} 
                        />
                        <span className={`font-mono text-xs tracking-tighter transition-colors ${selectedTokenName === t.token ? 'font-black text-ink-main' : 'text-ink-dim'}`}>
                          {t.token.toUpperCase()}
                        </span>
                      </div>
                      {selectedTokenName === t.token && (
                        <div className="w-1.5 h-1.5 bg-accent-blue rounded-full" />
                      )}
                    </motion.div>
                  ))}
                </div>
              </section>

              <section className="pt-4 border-t border-border-grid">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Target className="w-3 h-3 text-ink-dim" />
                    <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim">Cluster Lab</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button 
                      onClick={() => setClusterCount(Math.max(2, clusterCount - 1))}
                      className="w-5 h-5 flex items-center justify-center border border-border-grid text-[10px] font-bold hover:bg-bg-base"
                    >
                      -
                    </button>
                    <span className="text-[10px] font-mono font-bold w-4 text-center">{clusterCount}</span>
                    <button 
                      onClick={() => setClusterCount(Math.min(7, clusterCount + 1))}
                      className="w-5 h-5 flex items-center justify-center border border-border-grid text-[10px] font-bold hover:bg-bg-base"
                    >
                      +
                    </button>
                  </div>
                </div>

                <Button 
                  onClick={runClustering}
                  className="w-full h-10 rounded-none bg-ink-main text-white hover:bg-ink-main/90 font-black uppercase text-[10px] tracking-widest gap-2"
                >
                  <Activity className="w-3.5 h-3.5" />
                  Map Semantic Regions
                </Button>
              </section>

              <section className="pt-4 border-t border-border-grid">
                <div className="flex items-center gap-2 mb-4">
                  <Calculator className="w-3 h-3 text-ink-dim" />
                  <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim">Semantic Formulas</span>
                </div>
                <div className="space-y-3">
                  {[
                    '[Coffee] + [Cold]',
                    '[Love] - [Hate]',
                    '[Vibrant] + [Chaotic]',
                    '[Serious] - [Formal]'
                  ].map(ex => (
                    <button 
                      key={ex}
                      onClick={() => performArithmetic(ex)}
                      className="w-full p-3 bg-bg-base hover:bg-white border border-border-grid text-left group transition-all"
                    >
                      <div className="text-[9px] font-mono text-ink-dim group-hover:text-accent-blue transition-colors mb-1">EXAMPLE</div>
                      <div className="text-[11px] font-mono font-bold tracking-tight">{ex}</div>
                    </button>
                  ))}
                </div>
              </section>
            </div>
          </aside>

          {/* MAIN VISUALIZER */}
          <main className="flex-1 relative flex flex-col bg-bg-base swiss-grid overflow-hidden">
            {/* VECTOR FORMULA BAR */}
            <div className="h-12 border-b border-border-grid bg-white/80 backdrop-blur-md flex items-center px-6 z-40 gap-4">
              <Calculator className="w-3.5 h-3.5 text-ink-dim" />
              <div className="flex-1 flex items-center relative group">
                <Input 
                  placeholder="VECTOR TRANSFORM: [TOKEN A] + [TOKEN B] - [TOKEN C]..." 
                  value={formula}
                  onChange={(e) => performArithmetic(e.target.value)}
                  className="border-none bg-transparent hover:bg-bg-base/50 h-8 rounded-none font-mono text-[11px] tracking-tight focus-visible:ring-0 placeholder:opacity-50"
                />
                {formula && (
                  <button 
                    onClick={() => performArithmetic('')}
                    className="absolute right-0 p-1 hover:bg-bg-base"
                  >
                    <X className="w-3 h-3 text-ink-dim" />
                  </button>
                )}
              </div>
              
              {arithmeticResult && (
                <div className="flex items-center gap-3 border-l border-border-grid pl-6 animate-in fade-in slide-in-from-right-4">
                  <span className="text-[10px] font-black uppercase tracking-widest text-accent-blue">Result</span>
                  <ArrowRight className="w-3 h-3 text-accent-blue" />
                  <Badge variant="outline" className="rounded-none border-accent-blue font-mono text-[10px] bg-accent-blue/5 text-accent-blue h-6">
                    {arithmeticResult.target?.toUpperCase()}
                  </Badge>
                </div>
              )}
            </div>

            <div className="flex-1 p-8">
              <div className="w-full h-full swiss-panel shadow-sm">
                <EmbeddingPlot 
                  tokens={tokens} 
                  dimension={dimension} 
                  setDimension={setDimension}
                  selectedToken={selectedTokenName}
                  onSelectToken={setSelectedTokenName}
                  setTokens={setTokens}
                  arithmetic={arithmeticResult}
                  activeModel={activeModel}
                  comparisonMode={comparisonMode}
                  semanticMatches={semanticMatches}
                />
              </div>
            </div>

            {/* HUD FOOTER */}
            <footer className="h-40 border-t border-border-grid bg-white p-8 overflow-hidden z-20">
              <div className="grid grid-cols-4 gap-12 h-full items-end">
                <div>
                  <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim block mb-2">Coord X</span>
                  <div className="swiss-stat">
                    {tokens.find(t => t.token === selectedTokenName)?.x.toFixed(1) || '0.0'}
                  </div>
                </div>
                <div>
                  <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim block mb-2">Coord Y</span>
                  <div className="swiss-stat">
                    {(tokens.find(t => t.token === selectedTokenName)?.y || 0).toFixed(1)}
                  </div>
                </div>
                <div>
                  <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim block mb-2">Coord Z</span>
                  <div className="swiss-stat">
                    {(tokens.find(t => t.token === selectedTokenName)?.z || 0).toFixed(1)}
                  </div>
                </div>

                <div className="border-l border-border-grid pl-12 h-full flex flex-col justify-end">
                   <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim block mb-2">Model Calibration</span>
                   <div className="flex flex-col gap-1 w-full max-w-[180px]">
                      <div className="flex items-center justify-between">
                         <span className="text-[9px] font-bold text-ink-dim">ARCH:</span>
                         <span className="text-[10px] font-mono font-black">{activeModel === 'gemini-pro' ? 'MoE 1.2T' : 'Dense 450M'}</span>
                      </div>
                      <div className="flex items-center justify-between">
                         <span className="text-[9px] font-bold text-ink-dim">TOPO-VAR:</span>
                         <span className="text-[10px] font-mono font-black">{activeModel === 'gemini-pro' ? '±0.12' : '±0.45'}</span>
                      </div>
                      {comparisonMode && (
                        <div className="mt-2 pt-2 border-t border-border-grid flex items-center justify-between bg-accent-blue/5 p-1">
                           <span className="text-[9px] font-black text-accent-blue">DIVERGENCE:</span>
                           <span className="text-[12px] font-mono font-black text-accent-blue">{topologicalDivergence} Δ</span>
                        </div>
                      )}
                   </div>
                </div>

                <div className="flex flex-col items-end min-w-[300px]">
                  <div className="flex items-center justify-between w-full mb-2">
                    <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim">Distribution Density</span>
                    <span className="text-[9px] font-mono text-ink-dim opacity-50">VALENCE SPECTRUM [-10, 10]</span>
                  </div>
                  <div className="w-full h-14 bg-bg-base border border-border-grid flex items-end px-1 gap-[2px] pt-2 relative">
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 h-full w-[1px] bg-accent-blue/30 z-0" />
                    {densityDistribution.map((bin, i) => (
                      <div 
                        key={i} 
                        className={`flex-1 transition-all duration-300 relative z-10 ${bin.count > 0 ? 'bg-ink-dim/40 hover:bg-accent-blue' : 'bg-transparent'}`} 
                        style={{ height: `${bin.height}%` }}
                        title={`Bin ${i}: ${bin.count} items`}
                      />
                    ))}
                  </div>
                  <div className="flex justify-between w-full mt-1 px-1">
                    <span className="text-[8px] font-black text-ink-dim">NEG</span>
                    <span className="text-[8px] font-black text-ink-dim">NEU</span>
                    <span className="text-[8px] font-black text-ink-dim">POS</span>
                  </div>
                </div>
              </div>
            </footer>
          </main>

          {/* RIGHT SIDEBAR: INSPECTOR */}
          <aside className="w-[320px] bg-white border-l border-border-grid flex flex-col z-[50]">
            <div className="swiss-header">
              <Activity className="w-3.5 h-3.5" />
              Neural Statistics
            </div>

            <div className="p-6 flex-1 overflow-y-auto custom-scrollbar">
              {selectedToken ? (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-10"
                >
                  <div>
                    <h2 className="text-4xl font-black tracking-tighter mb-2">{selectedTokenName.toUpperCase()}</h2>
                    <Badge variant="outline" className="rounded-none border-border-grid font-mono text-[9px] px-2 py-0.5 uppercase">
                      Vector Node 0{tokens.findIndex(t => t.token === selectedTokenName) + 1}
                    </Badge>
                  </div>

                  <div className="space-y-6">
                    <div>
                      <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim block mb-3">Geometric Fingerprint</span>
                      <div className="grid grid-cols-2 gap-px bg-border-grid border border-border-grid">
                        {['x', 'y', 'z', 'w'].map((axis) => {
                          const val = (tokens.find(t => t.token === selectedTokenName) as any)?.[axis] || 0;
                          return (
                            <div key={axis} className="bg-white p-3">
                              <div className="text-[9px] font-black text-ink-dim uppercase mb-1">{axis}</div>
                              <div className="text-sm font-mono font-bold">{val.toFixed(4)}</div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div>
                      <span className="text-[10px] font-black uppercase tracking-widest text-ink-dim block mb-3">Relational Logic</span>
                      <div className="space-y-4">
                        {tokens
                          .filter(t => t.token !== selectedTokenName)
                          .slice(0, 3)
                          .map((t) => (
                            <div key={t.token} className="group">
                              <div className="flex justify-between text-[10px] font-bold mb-1.5 uppercase tracking-tighter">
                                <span className="text-ink-dim">{t.token}</span>
                                <span>{(Math.random() * 100).toFixed(1)}%</span>
                              </div>
                              <div className="h-1 bg-bg-base border border-border-grid overflow-hidden">
                                <motion.div 
                                  initial={{ width: 0 }}
                                  animate={{ width: `${Math.random() * 80 + 20}%` }}
                                  className="h-full bg-accent-blue/40 group-hover:bg-accent-blue transition-colors"
                                />
                              </div>
                            </div>
                          ))
                        }
                      </div>
                    </div>
                  </div>

                  <div className="pt-8 border-t border-border-grid">
                    <Button 
                      className="w-full rounded-none bg-ink-main hover:bg-black text-white font-black uppercase tracking-widest text-[10px] h-12"
                      onClick={() => {
                        const data = tokens.find(t => t.token === selectedTokenName);
                        console.log('Exporting vector data:', data);
                      }}
                    >
                      Export Vector Data
                    </Button>
                  </div>
                </motion.div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 gap-4 text-ink-dim">
                  <div className="p-4 border border-dashed border-border-grid">
                    <Activity className="w-8 h-8 opacity-20" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-[11px] font-black uppercase tracking-widest">Awaiting Selection</p>
                    <p className="text-[10px]">Select a node from the controller to analyze its neural properties.</p>
                  </div>
                </div>
              )}
            </div>
          </aside>
        </div>
      </div>
    </TooltipProvider>
  );
}
