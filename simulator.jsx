import React, { useState, useEffect, useRef, useCallback } from 'react';

// Physics helpers
const wrapPosition = (pos, length) => ((pos % length) + length) % length;

const signedDistance = (pos1, pos2, length) => {
  let diff = pos2 - pos1;
  if (diff > length / 2) diff -= length;
  if (diff < -length / 2) diff += length;
  return diff;
};

const resolveCollision = (m1, v1, m2, v2) => {
  const total = m1 + m2;
  return [
    ((m1 - m2) * v1 + 2 * m2 * v2) / total,
    ((m2 - m1) * v2 + 2 * m1 * v1) / total
  ];
};

// Color palettes
const ABSTRACT_COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6', '#3b82f6', '#8b5cf6', '#ec4899'];

export default function PolysomeSimulator() {
  const [mode, setMode] = useState('abstract'); // 'abstract' or 'polyribosome'
  const [isRunning, setIsRunning] = useState(false);
  const [time, setTime] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [collisionCount, setCollisionCount] = useState(0);
  
  // Abstract mode
  const [numParticles, setNumParticles] = useState(5);
  const [particles, setParticles] = useState([]);
  
  // Polyribosome mode
  const [ribosomes, setRibosomes] = useState([]);
  const [completedTranslations, setCompletedTranslations] = useState(0);
  const [initiationRate, setInitiationRate] = useState(0.1);
  
  // Refs for animation
  const animRef = useRef(null);
  const lastTimeRef = useRef(null);
  const stateRef = useRef({ particles: [], ribosomes: [], collisions: 0, completed: 0 });
  
  const TRACK_LENGTH = mode === 'abstract' ? 1000 : 400;
  const TRACK_RADIUS = 140;
  const CENTER = { x: 180, y: 180 };
  
  // Initialize abstract particles
  const initAbstract = useCallback(() => {
    const newParticles = [];
    const spacing = TRACK_LENGTH / numParticles;
    
    for (let i = 0; i < numParticles; i++) {
      const diameter = 5 + Math.random() * 15;
      newParticles.push({
        id: i,
        position: spacing * i + spacing * 0.3 + Math.random() * spacing * 0.4,
        velocity: (20 + Math.random() * 60) * (Math.random() < 0.5 ? 1 : -1),
        diameter,
        mass: Math.pow(diameter, 3) / 1000
      });
    }
    
    setParticles(newParticles);
    stateRef.current = { particles: newParticles, ribosomes: [], collisions: 0, completed: 0 };
    setCollisionCount(0);
    setTime(0);
  }, [numParticles, TRACK_LENGTH]);
  
  // Initialize polyribosomes
  const initPolyribosome = useCallback(() => {
    const startCodon = 50;
    const newRibosomes = [];
    
    for (let i = 0; i < 3; i++) {
      newRibosomes.push({
        id: i,
        position: startCodon + i * 40,
        velocity: 5 + Math.random() * 2,
        chainLength: i * 60, // Different chain lengths for heterogeneity
        get diameter() { return 10 + this.chainLength * 0.02; },
        get mass() { return 4.2 + this.chainLength * 0.0001; }
      });
    }
    
    setRibosomes(newRibosomes);
    stateRef.current = { particles: [], ribosomes: newRibosomes, collisions: 0, completed: 0 };
    setCollisionCount(0);
    setCompletedTranslations(0);
    setTime(0);
  }, []);
  
  // Initialize based on mode
  const initialize = useCallback(() => {
    setIsRunning(false);
    if (mode === 'abstract') initAbstract();
    else initPolyribosome();
  }, [mode, initAbstract, initPolyribosome]);
  
  useEffect(() => { initialize(); }, [mode]);
  
  // Physics update
  const update = useCallback((dt) => {
    const L = TRACK_LENGTH;
    
    if (mode === 'abstract') {
      const current = stateRef.current.particles.map(p => ({ ...p }));
      
      // Move
      for (const p of current) {
        p.position = wrapPosition(p.position + p.velocity * dt, L);
      }
      
      // Collisions
      const sorted = [...current].sort((a, b) => a.position - b.position);
      for (let i = 0; i < sorted.length; i++) {
        const p1 = sorted[i];
        const p2 = sorted[(i + 1) % sorted.length];
        
        const dist = Math.abs(signedDistance(p1.position, p2.position, L));
        const minDist = (p1.diameter + p2.diameter) / 2;
        
        if (dist < minDist) {
          const sDist = signedDistance(p1.position, p2.position, L);
          if ((p1.velocity - p2.velocity) * sDist > 0) {
            const [v1, v2] = resolveCollision(p1.mass, p1.velocity, p2.mass, p2.velocity);
            current.find(p => p.id === p1.id).velocity = v1;
            current.find(p => p.id === p2.id).velocity = v2;
            stateRef.current.collisions++;
          }
        }
      }
      
      stateRef.current.particles = current;
      return current;
      
    } else {
      const startCodon = 50;
      const stopCodon = 350;
      let current = stateRef.current.ribosomes.map(r => ({ ...r }));
      
      // Stochastic initiation
      if (Math.random() < initiationRate * dt) {
        const blocked = current.some(r => 
          Math.abs(signedDistance(r.position, startCodon, L)) < 20
        );
        if (!blocked && current.length < 15) {
          current.push({
            id: Date.now(),
            position: startCodon,
            velocity: 5 + Math.random() * 2,
            chainLength: 0,
            get diameter() { return 10 + this.chainLength * 0.02; },
            get mass() { return 4.2 + this.chainLength * 0.0001; }
          });
        }
      }
      
      // Move and grow chains
      for (const r of current) {
        r.position += r.velocity * dt;
        r.chainLength = Math.min(300, r.chainLength + r.velocity * dt);
        
        // Termination
        if (r.position >= stopCodon) {
          stateRef.current.completed++;
          r.position = startCodon;
          r.chainLength = 0;
        }
      }
      
      // Collisions
      const sorted = [...current].sort((a, b) => a.position - b.position);
      for (let i = 0; i < sorted.length; i++) {
        const r1 = sorted[i];
        const r2 = sorted[(i + 1) % sorted.length];
        
        const dist = Math.abs(signedDistance(r1.position, r2.position, L));
        const minDist = (r1.diameter + r2.diameter) / 2;
        
        if (dist < minDist) {
          const sDist = signedDistance(r1.position, r2.position, L);
          if ((r1.velocity - r2.velocity) * sDist > 0) {
            const [v1, v2] = resolveCollision(r1.mass, r1.velocity, r2.mass, r2.velocity);
            current.find(r => r.id === r1.id).velocity = Math.max(0.5, v1);
            current.find(r => r.id === r2.id).velocity = Math.max(0.5, v2);
            stateRef.current.collisions++;
          }
        }
      }
      
      stateRef.current.ribosomes = current;
      return current;
    }
  }, [mode, TRACK_LENGTH, initiationRate]);
  
  // Animation loop
  useEffect(() => {
    if (!isRunning) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    
    const animate = (timestamp) => {
      if (!lastTimeRef.current) lastTimeRef.current = timestamp;
      const dt = Math.min((timestamp - lastTimeRef.current) / 1000, 0.05) * speed;
      lastTimeRef.current = timestamp;
      
      const updated = update(dt);
      
      if (mode === 'abstract') setParticles([...updated]);
      else {
        setRibosomes([...updated]);
        setCompletedTranslations(stateRef.current.completed);
      }
      
      setTime(t => t + dt);
      setCollisionCount(stateRef.current.collisions);
      
      animRef.current = requestAnimationFrame(animate);
    };
    
    animRef.current = requestAnimationFrame(animate);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [isRunning, mode, speed, update]);
  
  useEffect(() => { if (isRunning) lastTimeRef.current = null; }, [isRunning]);
  
  // Position to coordinates
  const toCoords = (position) => {
    const angle = (position / TRACK_LENGTH) * 2 * Math.PI - Math.PI / 2;
    return {
      x: CENTER.x + TRACK_RADIUS * Math.cos(angle),
      y: CENTER.y + TRACK_RADIUS * Math.sin(angle)
    };
  };
  
  // Compute stats
  const totalEnergy = particles.reduce((sum, p) => sum + 0.5 * p.mass * p.velocity ** 2, 0);
  const totalMomentum = particles.reduce((sum, p) => sum + p.mass * p.velocity, 0);
  
  const massHeterogeneity = ribosomes.length > 1 ? (() => {
    const masses = ribosomes.map(r => r.mass);
    const mean = masses.reduce((a, b) => a + b, 0) / masses.length;
    const std = Math.sqrt(masses.reduce((a, b) => a + (b - mean) ** 2, 0) / masses.length);
    return ((std / mean) * 100).toFixed(1);
  })() : '0.0';

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-2xl font-bold text-center mb-1">Polyribosome Dynamics Simulator</h1>
        <p className="text-gray-400 text-center text-sm mb-4">
          Paper 1: Neural Operators for Impulsive Systems | Paper 2: Chaotic Translation Dynamics
        </p>
        
        {/* Mode Toggle */}
        <div className="flex justify-center mb-4">
          <div className="bg-gray-800 rounded-lg p-1 flex gap-1">
            <button
              onClick={() => setMode('abstract')}
              className={`px-4 py-2 rounded text-sm font-medium transition ${
                mode === 'abstract' ? 'bg-blue-600' : 'hover:bg-gray-700'
              }`}
            >
              🔵 Abstract Hard Spheres
            </button>
            <button
              onClick={() => setMode('polyribosome')}
              className={`px-4 py-2 rounded text-sm font-medium transition ${
                mode === 'polyribosome' ? 'bg-green-600' : 'hover:bg-gray-700'
              }`}
            >
              🧬 Polyribosome
            </button>
          </div>
        </div>
        
        {/* Controls */}
        <div className="bg-gray-800 rounded-lg p-4 mb-4">
          <div className="flex flex-wrap gap-4 items-center justify-center">
            {mode === 'abstract' ? (
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-400">Particles:</span>
                <input type="range" min="2" max="10" value={numParticles}
                  onChange={(e) => setNumParticles(+e.target.value)}
                  className="w-20" disabled={isRunning} />
                <span className="w-6 text-center">{numParticles}</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-400">Init Rate:</span>
                <input type="range" min="0.01" max="0.3" step="0.01" value={initiationRate}
                  onChange={(e) => setInitiationRate(+e.target.value)} className="w-20" />
                <span className="w-16 text-xs">{initiationRate.toFixed(2)}/s</span>
              </div>
            )}
            
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Speed:</span>
              <input type="range" min="0.25" max="4" step="0.25" value={speed}
                onChange={(e) => setSpeed(+e.target.value)} className="w-20" />
              <span className="w-8">{speed}x</span>
            </div>
            
            <button onClick={() => setIsRunning(!isRunning)}
              className={`px-4 py-2 rounded font-medium ${isRunning ? 'bg-red-600' : 'bg-green-600'}`}>
              {isRunning ? 'Pause' : 'Start'}
            </button>
            
            <button onClick={initialize} className="px-4 py-2 rounded font-medium bg-blue-600">
              Reset
            </button>
          </div>
        </div>
        
        {/* Main Area */}
        <div className="flex flex-col lg:flex-row gap-4">
          {/* Visualization */}
          <div className="bg-gray-800 rounded-lg p-4 flex-1">
            <svg width="360" height="360" className="mx-auto">
              {/* Track */}
              <circle cx={CENTER.x} cy={CENTER.y} r={TRACK_RADIUS}
                fill="none" stroke={mode === 'polyribosome' ? '#64748b' : '#4b5563'} strokeWidth="20" />
              
              {/* Polyribosome markers */}
              {mode === 'polyribosome' && (
                <>
                  <circle {...toCoords(50)} r="6" fill="#22c55e" />
                  <circle {...toCoords(350)} r="6" fill="#ef4444" />
                  <text x={CENTER.x} y={CENTER.y - 30} textAnchor="middle" fill="#64748b" fontSize="10">
                    🟢 Start | 🔴 Stop
                  </text>
                </>
              )}
              
              {/* Particles */}
              {mode === 'abstract' ? particles.map((p, i) => {
                const coords = toCoords(p.position);
                const color = ABSTRACT_COLORS[i % ABSTRACT_COLORS.length];
                const r = Math.max(4, p.diameter * 0.4);
                return (
                  <g key={p.id}>
                    <circle cx={coords.x} cy={coords.y} r={r} fill={color} stroke="white" strokeWidth="1" />
                    <text x={coords.x} y={coords.y + 3} textAnchor="middle" fill="white" fontSize="9">{i + 1}</text>
                  </g>
                );
              }) : ribosomes.map((r) => {
                const coords = toCoords(r.position);
                const chainR = r.chainLength * 0.015;
                return (
                  <g key={r.id}>
                    {chainR > 0 && <circle cx={coords.x} cy={coords.y} r={8 + chainR} fill="#22c55e" opacity="0.4" />}
                    <circle cx={coords.x} cy={coords.y} r="8" fill="#3b82f6" stroke="white" strokeWidth="1" />
                    <text x={coords.x} y={coords.y + 3} textAnchor="middle" fill="white" fontSize="7">
                      {Math.round(r.chainLength)}
                    </text>
                  </g>
                );
              })}
              
              {/* Center info */}
              <text x={CENTER.x} y={CENTER.y + 10} textAnchor="middle" fill="#9ca3af" fontSize="11">
                {mode === 'abstract' ? `${particles.length} particles` : `${ribosomes.length} ribosomes`}
              </text>
            </svg>
          </div>
          
          {/* Stats */}
          <div className="bg-gray-800 rounded-lg p-4 lg:w-72">
            <h3 className="font-semibold mb-3">{mode === 'abstract' ? 'System State' : 'Translation Stats'}</h3>
            
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Time:</span>
                <span className="font-mono">{time.toFixed(2)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Collisions:</span>
                <span className="font-mono">{collisionCount}</span>
              </div>
              
              <hr className="border-gray-700 my-2" />
              
              {mode === 'abstract' ? (
                <>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total Energy:</span>
                    <span className="font-mono text-green-400">{totalEnergy.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total Momentum:</span>
                    <span className="font-mono text-blue-400">{totalMomentum.toFixed(2)}</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">(Should remain constant)</p>
                </>
              ) : (
                <>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Completed:</span>
                    <span className="font-mono text-green-400">{completedTranslations}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Mass Heterogeneity:</span>
                    <span className="font-mono text-yellow-400">{massHeterogeneity}%</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">Higher = more chaotic</p>
                </>
              )}
            </div>
            
            <hr className="border-gray-700 my-3" />
            
            <div className="text-xs text-gray-400">
              <p className="font-medium text-gray-300 mb-1">🔬 What to observe:</p>
              {mode === 'abstract' ? (
                <ul className="space-y-1">
                  <li>• Energy/momentum conservation</li>
                  <li>• Mass asymmetry in collisions</li>
                  <li>• Chaotic trajectory divergence</li>
                </ul>
              ) : (
                <ul className="space-y-1">
                  <li>• Growing nascent chains (numbers)</li>
                  <li>• Traffic jams near stop codon</li>
                  <li>• Heterogeneity creates chaos</li>
                </ul>
              )}
            </div>
          </div>
        </div>
        
        {/* Paper Info */}
        <div className="bg-gray-800 rounded-lg p-4 mt-4">
          <h3 className="font-semibold mb-2">📄 Publication Strategy</h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-700 rounded p-3">
              <h4 className="font-medium text-blue-400 mb-1">Paper 1: ML Methods</h4>
              <p className="text-gray-300 text-xs">
                "Neural Operators for Density Evolution in Impulsive Dynamical Systems"
              </p>
              <p className="text-gray-500 text-xs mt-1">
                Target: NeurIPS / ICML / J. Comp. Physics
              </p>
            </div>
            <div className="bg-gray-700 rounded p-3">
              <h4 className="font-medium text-green-400 mb-1">Paper 2: Biophysics</h4>
              <p className="text-gray-300 text-xs">
                "Chaotic Hamiltonian Dynamics of Polyribosome Traffic: Beyond TASEP"
              </p>
              <p className="text-gray-500 text-xs mt-1">
                Target: Biophysical Journal / PLOS Comp Bio
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
