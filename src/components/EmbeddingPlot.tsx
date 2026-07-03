import React, { useMemo, useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Grid, Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import { TokenEmbedding, Dimension, ModelType } from '../types';
import { motion, AnimatePresence } from 'motion/react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Layers, Box, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import * as d3 from 'd3';
import { PCA } from 'ml-pca';
import { RefreshCcw, ScatterChart } from 'lucide-react';

interface Props {
  tokens: TokenEmbedding[];
  dimension: Dimension;
  setDimension: (d: Dimension) => void;
  selectedToken: string | null;
  onSelectToken: (token: string) => void;
  setTokens: React.Dispatch<React.SetStateAction<TokenEmbedding[]>>;
  arithmetic?: { sources: string[], target: string | null } | null;
  activeModel?: ModelType;
  comparisonMode?: boolean;
  semanticMatches?: string[];
}

const Token3D = ({ 
  token, 
  isSelected, 
  onSelect, 
  dimension,
  onHover,
  onHoverEnd,
  isSemanticMatch
}: { 
  token: TokenEmbedding; 
  isSelected: boolean; 
  onSelect: () => void; 
  dimension: Dimension;
  onHover?: () => void;
  onHoverEnd?: () => void;
  isSemanticMatch?: boolean;
}) => {
  const [hovered, setHovered] = useState(false);
  const wScale = dimension === 4 ? 1 + ((token.w || 0) + 10) / 20 : 1;

  return (
    <group position={[token.x / 2, (token.y || 0) / 2, (token.z || 0) / 2]}>
      {isSemanticMatch && (
        <mesh>
          <sphereGeometry args={[isSelected ? 0.35 : 0.25, 32, 32]} />
          <meshBasicMaterial color="#0066CC" transparent opacity={0.2} />
        </mesh>
      )}
      <mesh 
        onClick={onSelect}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          onHover?.();
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
          onHoverEnd?.();
        }}
        scale={hovered ? wScale * 1.1 : wScale}
      >
        <sphereGeometry args={[isSelected ? 0.25 : 0.15, 32, 32]} />
        <meshStandardMaterial 
          color={isSelected ? '#0066CC' : (hovered ? '#1D1D1F' : token.color || '#D2D2D7')} 
          roughness={0.1}
          metalness={0.1}
        />
      </mesh>

      {(hovered || isSelected) && (
        <Html position={[0, 0.4 * wScale, 0]} center distanceFactor={10}>
          <div className="bg-white border border-border-grid p-2 shadow-sm pointer-events-none whitespace-nowrap">
            <div className="text-[10px] font-black uppercase tracking-tighter">{token.token}</div>
            <div className="text-[9px] font-mono text-ink-dim">
              [{token.x.toFixed(1)}, {(token.y || 0).toFixed(1)}, {(token.z || 0).toFixed(1)}]
            </div>
          </div>
        </Html>
      )}
    </group>
  );
};

export const EmbeddingPlot: React.FC<Props> = ({ 
  tokens, 
  dimension, 
  setDimension, 
  selectedToken, 
  onSelectToken, 
  setTokens, 
  arithmetic,
  activeModel = 'gemini-flash',
  comparisonMode = false,
  semanticMatches = []
}) => {
  const [viewMode, setViewMode] = useState<'1d' | '2d' | '3d'>(dimension === 1 ? '1d' : (dimension >= 3 ? '3d' : '2d'));
  const [isProjecting, setIsProjecting] = useState(false);
  const [hoveredTokenName, setHoveredTokenName] = useState<string | null>(null);
  const width = 800;
  const height = 400;
  const padding = 60;

  const svgRef = useRef<SVGSVGElement>(null);
  const [zoomTransform, setZoomTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);

  const xScale = d3.scaleLinear().domain([-10, 10]).range([padding, width - padding]);
  const yScale = d3.scaleLinear().domain([-10, 10]).range([height - padding, padding]);

  // Calculate alternative model manifold for comparison overlay
  const comparisonManifold = useMemo(() => {
    if (!comparisonMode) return null;
    const otherModel = activeModel === 'gemini-flash' ? 'gemini-pro' : 'gemini-flash';
    const validTokens = tokens.filter(t => t.modelVectors && t.modelVectors[otherModel]);
    if (validTokens.length < 2) return null;

    const data = validTokens.map(t => t.modelVectors![otherModel]);
    const pca = new PCA(data);
    const projected = pca.predict(data).to2DArray();

    return validTokens.map((t, idx) => {
      const p = projected[idx];
      return {
        token: t.token,
        x: p[0] * 2,
        y: (p[1] || 0) * 2,
        z: (p[2] || 0) * 2
      };
    });
  }, [tokens, comparisonMode, activeModel]);

  useEffect(() => {
    if (dimension === 1) {
      setViewMode('1d');
    } else if (viewMode === '1d' && dimension >= 2) {
      setViewMode('2d');
    } else if (viewMode === '2d' && dimension >= 3) {
      setViewMode('3d');
    } else if (viewMode === '3d' && dimension < 3) {
      setViewMode('2d');
    }
  }, [dimension]);

  useEffect(() => {
    if (!svgRef.current || viewMode === '3d') return;

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        setZoomTransform(event.transform);
      });

    const selection = d3.select(svgRef.current);
    selection.call(zoom);
    selection.transition().duration(750).call(zoom.transform, d3.zoomIdentity);

    return () => {
      selection.on('.zoom', null);
    };
  }, [dimension, viewMode]);

  const get1DView = (v: string) => v === '1d';
  const get2DView = (v: string) => v === '2d';
  const get3DView = (v: string) => v === '3d';

  const clusterHulls = useMemo(() => {
    if (viewMode === '3d' || viewMode === '1d') return [];
    
    const clusters: Record<number, [number, number][]> = {};
    tokens.forEach(t => {
      if (t.cluster !== undefined) {
        if (!clusters[t.cluster]) clusters[t.cluster] = [];
        const x = xScale(t.x);
        const y = yScale(t.y || 0);
        clusters[t.cluster].push([x, y]);
      }
    });

    return Object.entries(clusters).map(([id, points]) => {
      if (points.length < 3) return null;
      const hull = d3.polygonHull(points);
      if (!hull) return null;
      
      const clusterId = parseInt(id);
      const representative = tokens.find(t => t.cluster === clusterId);
      
      return { 
        id: clusterId, 
        path: d3.line()(hull) + 'Z',
        color: representative?.color || '#D2D2D7'
      };
    }).filter((h): h is { id: number, path: string, color: string } => h !== null);
  }, [tokens, viewMode, dimension, xScale, yScale, height]);

  const runPCA = () => {
    const validTokens = tokens.filter(t => t.vector && t.vector.length > 0);
    if (validTokens.length < 2) return;

    setIsProjecting(true);
    
    // Tiny delay to show UI state
    setTimeout(() => {
      try {
        const data = validTokens.map(t => t.vector!);
        const pca = new PCA(data);
        const projected = pca.predict(data).to2DArray();

        // Map projected coordinates to -10 to 10 range
        const mapRange = (val: number, min: number, max: number) => {
          if (max === min) return 0;
          return ((val - min) / (max - min)) * 20 - 10;
        };

        const xCol = projected.map(p => p[0]);
        const yCol = projected.length > 0 && projected[0].length > 1 ? projected.map(p => p[1]) : Array(projected.length).fill(0);
        const zCol = projected.length > 0 && projected[0].length > 2 ? projected.map(p => p[2]) : Array(projected.length).fill(0);
        const wCol = projected.length > 0 && projected[0].length > 3 ? projected.map(p => p[3]) : Array(projected.length).fill(0);

        const minX = Math.min(...xCol);
        const maxX = Math.max(...xCol);
        const minY = Math.min(...yCol);
        const maxY = Math.max(...yCol);
        const minZ = Math.min(...zCol);
        const maxZ = Math.max(...zCol);
        const minW = Math.min(...wCol);
        const maxW = Math.max(...wCol);

        setTokens(prev => prev.map(t => {
          const idx = validTokens.findIndex(vt => vt.token === t.token);
          if (idx === -1) return t;
          
          return {
            ...t,
            x: mapRange(projected[idx][0], minX, maxX),
            y: projected[idx].length > 1 ? mapRange(projected[idx][1], minY, maxY) : 0,
            z: projected[idx].length > 2 ? mapRange(projected[idx][2], minZ, maxZ) : 0,
            w: projected[idx].length > 3 ? mapRange(projected[idx][3], minW, maxW) : 0
          };
        }));
      } catch (err) {
        console.error("PCA Error:", err);
      } finally {
        setIsProjecting(false);
      }
    }, 800);
  };

  if (get3DView(viewMode)) {
    return (
      <div className="w-full h-full relative bg-white">
        <div className="absolute top-4 left-4 z-20 flex gap-4">
          <div className="bg-white/90 border border-border-grid p-3 flex flex-col gap-2 shadow-sm min-w-[140px]">
            <div className="text-[9px] font-black uppercase tracking-widest text-ink-dim">Coordinate Key</div>
            {[
              { label: 'X Axis', color: 'bg-ink-main' },
              { label: 'Y Axis', color: 'bg-accent-blue' },
              { label: 'Z Axis', color: 'bg-accent-red' },
            ].map(item => (
              <div key={item.label} className="flex items-center gap-2 text-[10px] font-bold">
                <div className={`w-2 h-2 ${item.color}`} />
                <span>{item.label}</span>
              </div>
            ))}
            <div className="mt-2 pt-2 border-t border-border-grid">
               <Button 
                variant="outline" 
                size="sm" 
                className="w-full text-[9px] font-black uppercase h-7 gap-2 rounded-none border-ink-main/10 hover:bg-bg-base"
                onClick={runPCA}
                disabled={isProjecting}
               >
                 <RefreshCcw className={`w-3 h-3 ${isProjecting ? 'animate-spin' : ''}`} />
                 {isProjecting ? 'Projecting...' : 'Run PCA Lab'}
               </Button>
            </div>
          </div>
        </div>

        <Canvas camera={{ position: [10, 10, 10], fov: 45 }}>
          <color attach="background" args={['#F5F5F7']} />
          <ambientLight intensity={1.5} />
          <pointLight position={[10, 10, 10]} intensity={200} />
          <Grid 
            infiniteGrid 
            fadeDistance={30} 
            sectionSize={1} 
            sectionThickness={1} 
            cellSize={0.5} 
            cellThickness={0.5} 
            sectionColor="#D2D2D7" 
            cellColor="#E5E7EB" 
          />
          
          <group>
            <primitive object={new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-10, 0, 0), 21, 0x1D1D1F, 0.4, 0.2)} />
            <primitive object={new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, -10, 0), 21, 0x0066CC, 0.4, 0.2)} />
            <primitive object={new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -10), 21, 0xE30000, 0.4, 0.2)} />
          </group>

          {comparisonMode && comparisonManifold && (
            <group>
              {comparisonManifold.map(t => (
                <group key={`other-3d-${t.token}`} position={[t.x / 2, t.y / 2, t.z / 2]}>
                  <mesh>
                    <sphereGeometry args={[0.08, 16, 16]} />
                    <meshBasicMaterial color="#D2D2D7" transparent opacity={0.4} />
                  </mesh>
                </group>
              ))}
              {tokens.map(t => {
                const other = comparisonManifold.find(cm => cm.token === t.token);
                if (!other) return null;
                return (
                  <Line
                    key={`thread-3d-${t.token}`}
                    points={[
                      [t.x / 2, (t.y || 0) / 2, (t.z || 0) / 2],
                      [other.x / 2, other.y / 2, other.z / 2]
                    ]}
                    color="#D2D2D7"
                    lineWidth={1}
                    dashed
                    dashScale={10}
                    dashSize={0.2}
                    gapSize={0.2}
                    transparent
                    opacity={0.3}
                  />
                );
              })}
            </group>
          )}

          {/* Arithmetic 3D Connectors */}
          {arithmetic && arithmetic.target && (
            <group>
              {arithmetic.sources.map((source, i) => {
                const sToken = tokens.find(t => t.token === source);
                const tToken = tokens.find(t => t.token === arithmetic.target);
                if (!sToken || !tToken) return null;

                return (
                  <Line
                    key={`line3d-${source}-${i}`}
                    points={[
                      [sToken.x / 2, (sToken.y || 0) / 2, (sToken.z || 0) / 2],
                      [tToken.x / 2, (tToken.y || 0) / 2, (tToken.z || 0) / 2]
                    ]}
                    color="#0066CC"
                    lineWidth={3}
                    dashed
                    dashScale={10}
                    dashSize={0.2}
                    gapSize={0.2}
                  />
                );
              })}
            </group>
          )}

          {tokens.map((t) => (
            <Token3D 
              key={t.token} 
              token={t} 
              isSelected={selectedToken === t.token} 
              isSemanticMatch={semanticMatches.includes(t.token)}
              onSelect={() => onSelectToken(t.token)} 
              dimension={dimension} 
              onHover={() => setHoveredTokenName(t.token)}
              onHoverEnd={() => setHoveredTokenName(null)}
            />
          ))}

          {dimension === 4 && (
            <Line
              points={tokens.map(t => [t.x / 2, (t.y || 0) / 2, (t.w || 0) / 2]) as [number, number, number][]}
              color="#424245"
              lineWidth={2}
              transparent
              opacity={0.3}
            />
          )}

          <OrbitControls makeDefault enableDamping dampingFactor={0.05} />
        </Canvas>

        <div className="absolute top-4 right-4 flex bg-white border border-border-grid p-1 shadow-sm z-20">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setViewMode('1d')}
            className={`h-8 px-4 gap-2 text-[10px] font-black uppercase rounded-none transition-all ${get1DView(viewMode) ? 'bg-bg-base text-ink-main' : 'text-ink-dim hover:text-ink-main'}`}
          >
            <Layers className="w-3 h-3" />
            1D AXIS
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setViewMode('2d')}
            className={`h-8 px-4 gap-2 text-[10px] font-black uppercase rounded-none transition-all ${get2DView(viewMode) ? 'bg-bg-base text-ink-main' : 'text-ink-dim hover:text-ink-main'}`}
          >
            <Square className="w-3 h-3" />
            2D MANIFOLD
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setViewMode('3d')}
            className={`h-8 px-4 gap-2 text-[10px] font-black uppercase rounded-none transition-all ${get3DView(viewMode) ? 'bg-bg-base text-ink-main' : 'text-ink-dim hover:text-ink-main'}`}
          >
            <Box className="w-3 h-3" />
            3D SPACE
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative flex items-center justify-center p-8 bg-white cursor-grab active:cursor-grabbing overflow-hidden">
      <div className="absolute top-4 left-4 z-20 flex flex-col gap-3">
        <div className="bg-white/90 border border-border-grid p-3 flex flex-col gap-2 shadow-sm min-w-[140px]">
          <div className="text-[9px] font-black uppercase tracking-widest text-ink-dim">Coordinate Key</div>
          {[
            { label: 'Valence (X)', color: 'bg-ink-main' },
            { label: 'Arousal (Y)', color: 'bg-accent-blue' },
          ].map(item => (
            <div key={item.label} className="flex items-center gap-2 text-[10px] font-bold">
              <div className={`w-2 h-2 ${item.color}`} />
              <span>{item.label}</span>
            </div>
          ))}
          <div className="mt-2 pt-2 border-t border-border-grid">
              <Button 
              variant="outline" 
              size="sm" 
              className="w-full text-[9px] font-black uppercase h-7 gap-2 rounded-none border-ink-main/10 hover:bg-bg-base"
              onClick={runPCA}
              disabled={isProjecting}
              >
                <RefreshCcw className={`w-3 h-3 ${isProjecting ? 'animate-spin' : ''}`} />
                {isProjecting ? 'Projecting...' : 'Run PCA Lab'}
              </Button>
          </div>
        </div>
      </div>

      <div className="absolute top-4 right-4 flex bg-white border border-border-grid p-1 shadow-sm z-20">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setViewMode('1d')}
          className={`h-8 px-4 gap-2 text-[10px] font-black uppercase rounded-none transition-all ${get1DView(viewMode) ? 'bg-bg-base text-ink-main' : 'text-ink-dim hover:text-ink-main'}`}
        >
          <Layers className="w-3 h-3" />
          1D AXIS
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setViewMode('2d')}
          className={`h-8 px-4 gap-2 text-[10px] font-black uppercase rounded-none transition-all ${get2DView(viewMode) ? 'bg-bg-base text-ink-main' : 'text-ink-dim hover:text-ink-main'}`}
        >
          <Square className="w-3 h-3" />
          2D MANIFOLD
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setViewMode('3d')}
          className={`h-8 px-4 gap-2 text-[10px] font-black uppercase rounded-none transition-all ${get3DView(viewMode) ? 'bg-bg-base text-ink-main' : 'text-ink-dim hover:text-ink-main'}`}
        >
          <Box className="w-3 h-3" />
          3D SPACE
        </Button>
      </div>

      <svg 
        ref={svgRef}
        width="100%" 
        height="100%" 
        viewBox={`0 0 ${width} ${height}`} 
        className="overflow-hidden touch-none"
      >
        <g transform={zoomTransform.toString()}>
          {/* Subtle Grid Lines */}
          {get2DView(viewMode) ? (
            [-10, -5, 0, 5, 10].map(val => (
              <React.Fragment key={`grid-${val}`}>
                <line x1={xScale(val)} y1={-height * 5} x2={xScale(val)} y2={height * 5} stroke="#D2D2D7" strokeWidth="0.5" strokeDasharray="2 2" />
                <line x1={-width * 5} y1={yScale(val)} x2={width * 5} y2={yScale(val)} stroke="#D2D2D7" strokeWidth="0.5" strokeDasharray="2 2" />
              </React.Fragment>
            ))
          ) : (
             <React.Fragment>
               {[-10, -5, 0, 5, 10].map(val => (
                 <g key={`grid-v-${val}`} transform={`translate(${xScale(val)}, ${height/2})`}>
                    <line y1={-20} y2={20} stroke={val === 0 ? "#1D1D1F" : "#D2D2D7"} strokeWidth={val === 0 ? "2" : "1"} />
                    <text 
                      y={40} 
                      textAnchor="middle" 
                      className="text-[9px] font-mono font-bold fill-ink-dim"
                    >
                      {val > 0 ? `+${val}` : val}
                    </text>
                 </g>
               ))}
               
               {/* Descriptive Labels */}
               <g transform={`translate(${xScale(-10)}, ${height/2 - 50})`}>
                 <text className="text-[11px] font-black fill-ink-dim tracking-[0.2em] uppercase">← NEGATIVE VALENCE</text>
                 <text y={15} className="text-[9px] font-mono fill-ink-dim opacity-50 uppercase">Scale Extremity: Unpleasant / Distress</text>
               </g>
               <g transform={`translate(${xScale(10)}, ${height/2 - 50})`}>
                 <text textAnchor="end" className="text-[11px] font-black fill-ink-main tracking-[0.2em] uppercase">POSITIVE VALENCE →</text>
                 <text y={15} textAnchor="end" className="text-[9px] font-mono fill-ink-main opacity-50 uppercase">Scale Extremity: Pleasant / Joy</text>
               </g>

               {/* Rug Plot Distribution */}
               {tokens.map((t, i) => (
                 <line 
                   key={`rug-${t.token}-${i}`}
                   x1={xScale(t.x)}
                   x2={xScale(t.x)}
                   y1={height/2 + 60}
                   y2={height/2 + 75}
                   stroke={selectedToken === t.token ? "#0066CC" : "#D2D2D7"}
                   strokeWidth="1"
                   opacity="0.5"
                 />
               ))}
               <text x={xScale(0)} y={height/2 + 90} textAnchor="middle" className="text-[8px] font-black uppercase tracking-widest fill-ink-dim opacity-30">Distribution Density</text>
             </React.Fragment>
          )}

          {/* Persistent Axes */}
          <line x1={-width * 5} y1={height/2} x2={width * 5} y2={height/2} stroke="#1D1D1F" strokeWidth={get1DView(viewMode) ? "5" : "1"} strokeLinecap="square" />
          
          {/* Cluster Halos (Semantic Hulls) */}
          <AnimatePresence>
            {clusterHulls.map((hull) => (
              <motion.path
                key={`hull-${hull.id}`}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 0.15, scale: 1 }}
                exit={{ opacity: 0 }}
                d={hull.path}
                fill={hull.color}
                stroke={hull.color}
                strokeWidth="60"
                strokeLinejoin="round"
                className="pointer-events-none"
                style={{ filter: 'blur(10px)' }}
              />
            ))}
          </AnimatePresence>

          {get2DView(viewMode) && (
            <React.Fragment>
              <line x1={-width * 5} y1={yScale(0)} x2={width * 5} y2={yScale(0)} stroke="#1D1D1F" strokeWidth="1" />
              <line x1={xScale(0)} y1={-height * 5} x2={xScale(0)} y2={height * 5} stroke="#0066CC" strokeWidth="1" opacity={dimension >= 2 ? 1 : 0.1} />
            </React.Fragment>
          )}

          {/* Sequential Connectors */}
          <path
            d={d3.line<TokenEmbedding>()
              .x(d => xScale(d.x))
              .y(d => get2DView(viewMode) && dimension >= 2 ? yScale(d.y || 0) : height / 2)
              (tokens) || ''}
            fill="none"
            stroke="#1D1D1F"
            strokeWidth={get1DView(viewMode) ? "2.5" : "1"}
            strokeOpacity={get1DView(viewMode) ? 0.6 : 0.2}
            className="pointer-events-none"
          />

          {/* Topological Comparison Threads (Connecting Models) */}
          {comparisonMode && comparisonManifold && (
            <g className="topology-threads">
              {tokens.map(t => {
                const other = comparisonManifold.find(cm => cm.token === t.token);
                if (!other) return null;
                const x1 = xScale(t.x);
                const y1 = get2DView(viewMode) && dimension >= 2 ? yScale(t.y || 0) : height / 2;
                const x2 = xScale(other.x);
                const y2 = get2DView(viewMode) && dimension >= 2 ? yScale(other.y) : height / 2;
                
                return (
                  <line 
                    key={`thread-${t.token}`}
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke="#D2D2D7"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                    opacity="0.4"
                  />
                );
              })}
            </g>
          )}

          {/* Alternative Manifold Nodes */}
          {comparisonMode && comparisonManifold && (
            <g className="other-model-nodes">
              {comparisonManifold.map(t => {
                const x = xScale(t.x);
                const y = get2DView(viewMode) && dimension >= 2 ? yScale(t.y) : height / 2;
                return (
                  <circle 
                    key={`other-${t.token}`}
                    cx={x} cy={y} r="3"
                    fill="transparent"
                    stroke="#D2D2D7"
                    strokeWidth="1"
                  />
                );
              })}
            </g>
          )}

          {/* Arithmetic Transformation Arrows */}
          <AnimatePresence>
            {arithmetic && arithmetic.target && (
              <g className="arithmetic-layer">
                {arithmetic.sources.map((source, i) => {
                  const sToken = tokens.find(t => t.token === source);
                  const tToken = tokens.find(t => t.token === arithmetic.target);
                  if (!sToken || !tToken) return null;

                  const x1 = xScale(sToken.x);
                  const y1 = get2DView(viewMode) && dimension >= 2 ? yScale(sToken.y || 0) : height / 2;
                  const x2 = xScale(tToken.x);
                  const y2 = get2DView(viewMode) && dimension >= 2 ? yScale(tToken.y || 0) : height / 2;

                  return (
                    <motion.line
                      key={`arithmetic-${source}-${i}`}
                      initial={{ pathLength: 0, opacity: 0 }}
                      animate={{ pathLength: 1, opacity: 1 }}
                      exit={{ opacity: 0 }}
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke="#0066CC"
                      strokeWidth="2"
                      strokeDasharray="4 4"
                      markerEnd="url(#arrowhead)"
                    />
                  );
                })}
              </g>
            )}
          </AnimatePresence>

          {/* SVG Arrowhead Definition */}
          <defs>
            <marker 
              id="arrowhead" 
              markerWidth="10" 
              markerHeight="7" 
              refX="10" 
              refY="3.5" 
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#0066CC" />
            </marker>
          </defs>

          {tokens.map((t) => {
            const x = xScale(t.x);
            const y = get2DView(viewMode) && dimension >= 2 ? yScale(t.y || 0) : height / 2;
            const isSelected = selectedToken === t.token;
            const isArithmeticTarget = arithmetic?.target === t.token;
            const isSemanticMatch = semanticMatches.includes(t.token);

            return (
              <Tooltip key={t.token}>
                <TooltipTrigger>
                  <motion.g
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1, x, y }}
                    className="cursor-pointer group"
                    onClick={() => onSelectToken(t.token)}
                  >
                    {(isArithmeticTarget || isSemanticMatch) && (
                      <motion.circle
                        initial={{ scale: 0.8, opacity: 0.2 }}
                        animate={{ scale: 1.8, opacity: 0 }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        r={isSemanticMatch ? 12 : 8}
                        fill="#0066CC"
                      />
                    )}
                    <circle
                      r={isSelected || isArithmeticTarget ? (get1DView(viewMode) ? 10 : 8) : (get1DView(viewMode) ? 6 : (isSemanticMatch ? 7 : 4))}
                      fill={isArithmeticTarget || isSemanticMatch ? '#0066CC' : (isSelected ? '#0066CC' : (hoveredTokenName === t.token ? '#1D1D1F' : t.color || '#D2D2D7'))}
                      stroke={isSemanticMatch ? "rgba(0,102,204,0.5)" : "white"}
                      strokeWidth={isArithmeticTarget || isSemanticMatch ? "3" : "2"}
                    />
                    <text
                      y={-15}
                      textAnchor="middle"
                      className={`text-[10px] font-black uppercase tracking-tighter transition-all ${isSelected || isArithmeticTarget ? 'fill-ink-main' : 'fill-ink-dim opacity-0 group-hover:opacity-100'}`}
                    >
                      {t.token}
                    </text>
                  </motion.g>
                </TooltipTrigger>
                <TooltipContent className="bg-white border border-border-grid text-ink-main shadow-sm rounded-none p-2">
                  <div className="font-black uppercase tracking-tighter text-[10px]">{t.token}</div>
                  <div className="font-mono text-[9px] text-ink-dim">VALENCE: {t.x.toFixed(2)}</div>
                  {get2DView(viewMode) && dimension >= 2 && <div className="font-mono text-[9px] text-ink-dim">AROUSAL: {(t.y || 0).toFixed(2)}</div>}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </g>
      </svg>
    </div>
  );
};
