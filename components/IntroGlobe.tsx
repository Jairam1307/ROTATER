
import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface IntroGlobeProps {
  onEnter: () => void;
}

const IntroGlobe: React.FC<IntroGlobeProps> = ({ onEnter }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const [timestamp, setTimestamp] = useState<string>(new Date().toISOString());

  useEffect(() => {
    const timer = setInterval(() => {
      setTimestamp(new Date().toISOString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;

    // --- Scene Setup ---
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // --- Textures ---
    const textureLoader = new THREE.TextureLoader();
    const earthTexture = textureLoader.load('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg');
    const bumpMap = textureLoader.load('https://unpkg.com/three-globe/example/img/earth-topology.png');
    const specularMap = textureLoader.load('https://unpkg.com/three-globe/example/img/earth-waterbodies.png');

    // --- Earth Group ---
    const globeGroup = new THREE.Group();
    scene.add(globeGroup);

    // --- Earth Object ---
    const geometry = new THREE.SphereGeometry(1, 64, 64);
    const material = new THREE.MeshPhongMaterial({
      map: earthTexture,
      bumpMap: bumpMap,
      bumpScale: 0.02,
      specularMap: specularMap,
      specular: new THREE.Color(0x333333),
      shininess: 15,
    });
    const earth = new THREE.Mesh(geometry, material);
    globeGroup.add(earth);

    // --- Atmosphere ---
    const atmosphereGeometry = new THREE.SphereGeometry(1.1, 64, 64);
    const atmosphereMaterial = new THREE.ShaderMaterial({
      transparent: true,
      side: THREE.BackSide,
      uniforms: {
        glowColor: { value: new THREE.Color(0x00f0ff) },
        viewVector: { value: camera.position }
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewPosition;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vViewPosition = -mvPosition.xyz;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 glowColor;
        varying vec3 vNormal;
        varying vec3 vViewPosition;
        void main() {
          float intensity = pow(0.6 - dot(vNormal, normalize(vViewPosition)), 6.0);
          gl_FragColor = vec4(glowColor, intensity);
        }
      `
    });
    const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    globeGroup.add(atmosphere);

    // --- Orbital Rings (Satellite Tracks) ---
    const createRing = (radius: number, rotationX: number, rotationZ: number, color: number) => {
      const ringGeometry = new THREE.RingGeometry(radius, radius + 0.005, 128);
      const ringMaterial = new THREE.MeshBasicMaterial({ 
        color: color, 
        side: THREE.DoubleSide, 
        transparent: true, 
        opacity: 0.3 
      });
      const ring = new THREE.Mesh(ringGeometry, ringMaterial);
      ring.rotation.x = rotationX;
      ring.rotation.z = rotationZ;
      return ring;
    };

    const rings = [
      createRing(1.3, Math.PI / 2.5, 0.5, 0x00f0ff),
      createRing(1.5, -Math.PI / 3, -0.2, 0x0066ff),
      createRing(1.2, Math.PI / 4, 1.2, 0x00f0ff)
    ];
    rings.forEach(r => globeGroup.add(r));

    // --- Starfield ---
    const starGeometry = new THREE.BufferGeometry();
    const starMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.005, sizeAttenuation: true });
    const starVertices = [];
    for (let i = 0; i < 8000; i++) {
      const x = (Math.random() - 0.5) * 20;
      const y = (Math.random() - 0.5) * 20;
      const z = (Math.random() - 0.5) * 20;
      starVertices.push(x, y, z);
    }
    starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
    const stars = new THREE.Points(starGeometry, starMaterial);
    scene.add(stars);

    // --- Lights ---
    const ambientLight = new THREE.AmbientLight(0x404040, 1);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 2);
    mainLight.position.set(5, 3, 5);
    scene.add(mainLight);

    const blueLight = new THREE.PointLight(0x00f0ff, 1, 10);
    blueLight.position.set(-2, 1, 2);
    scene.add(blueLight);

    // --- Mouse Interaction ---
    let mouseX = 0;
    let mouseY = 0;
    const handleMouseMove = (event: MouseEvent) => {
      mouseX = (event.clientX - window.innerWidth / 2) / 1000;
      mouseY = (event.clientY - window.innerHeight / 2) / 1000;
    };
    window.addEventListener('mousemove', handleMouseMove);

    // --- Animation Loop ---
    const animate = () => {
      if (!rendererRef.current) return;
      
      // Rotations
      earth.rotation.y += 0.0012;
      rings[0].rotation.z += 0.002;
      rings[1].rotation.z -= 0.001;
      rings[2].rotation.z += 0.003;

      // Smooth camera follow or globe tilt
      globeGroup.rotation.y += (mouseX - globeGroup.rotation.y) * 0.05;
      globeGroup.rotation.x += (mouseY - globeGroup.rotation.x) * 0.05;

      rendererRef.current.render(scene, camera);
      requestAnimationFrame(animate);
    };

    animate();

    // --- Resize Handling ---
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      if (rendererRef.current) {
        rendererRef.current.setSize(window.innerWidth, window.innerHeight);
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      if (rendererRef.current) {
        rendererRef.current.dispose();
        containerRef.current?.removeChild(rendererRef.current.domElement);
      }
    };
  }, []);

  return (
    <div 
      className="relative w-full h-screen overflow-hidden bg-[#00040a] flex flex-col items-center justify-center cursor-pointer group" 
      onClick={onEnter}
    >
      <div ref={containerRef} className="absolute inset-0 z-0" />
      
      {/* HUD Elements */}
      <div className="absolute inset-0 pointer-events-none border-[1px] border-cyan-500/10 m-12 rounded-2xl overflow-hidden">
        <div className="absolute top-0 left-0 w-32 h-32 border-t-2 border-l-2 border-cyan-500/40 rounded-tl-2xl" />
        <div className="absolute bottom-0 right-0 w-32 h-32 border-b-2 border-r-2 border-cyan-500/40 rounded-br-2xl" />
      </div>

      <div className="z-10 text-center select-none transition-all duration-1000 group-hover:scale-105 pointer-events-none">
        <div className="mb-4 inline-block px-4 py-1 border border-cyan-500/30 rounded-full bg-cyan-950/20 backdrop-blur-sm animate-pulse">
           <span className="text-[10px] font-orbitron font-bold text-cyan-400 tracking-[0.3em] uppercase">System Uplink Active</span>
        </div>
        
        <h1 className="text-8xl md:text-[10rem] font-orbitron font-black text-transparent bg-clip-text bg-gradient-to-b from-white via-cyan-400 to-blue-900 tracking-tighter leading-none mb-2 drop-shadow-[0_0_30px_rgba(0,240,255,0.4)]">
          ROTATER
        </h1>
        
        <p className="text-cyan-200 font-exo text-sm md:text-base tracking-[0.2em] uppercase opacity-70 mb-16 font-light max-w-2xl px-4">
          Satellite-Based Climate Intelligence, Historical Analysis & Early Warning System
        </p>
        
        <div className="mt-12 pointer-events-auto">
          <button className="group/btn relative overflow-hidden font-orbitron text-white border border-cyan-400/50 px-12 py-5 rounded-sm bg-cyan-500/5 hover:bg-cyan-500 transition-all duration-500 hover:text-black hover:shadow-[0_0_40px_rgba(0,240,255,0.6)]">
            <span className="relative z-10 font-bold tracking-[0.3em] uppercase">Initialize Session</span>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover/btn:translate-x-full transition-transform duration-1000" />
          </button>
        </div>
      </div>
      
      {/* Telemetry Data (Left) - NOW DYNAMIC */}
      <div className="absolute bottom-12 left-12 text-[9px] font-mono text-cyan-500/60 space-y-2 uppercase tracking-widest pointer-events-none hidden md:block">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-ping" />
          <span>Orbital_Sync: <span className="text-cyan-400">Locked</span></span>
        </div>
        <div>Sector_Scan: <span className="text-cyan-400">100% Complete</span></div>
        <div>Model_State: <span className="text-cyan-400">Gemini_Flash_v3</span></div>
        <div className="pt-2 border-t border-cyan-900/50 font-light text-[8px]">
          Timestamp: <span className="text-cyan-400">{timestamp}</span>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 w-full max-w-4xl text-center pointer-events-none hidden md:block">
        <div className="flex flex-col items-center gap-2">
          <span className="text-[10px] font-mono font-light text-cyan-700/60 uppercase tracking-[0.4em]">DEVELOPMENT_BY</span>
          <div className="flex flex-wrap justify-center gap-x-12 gap-y-2 text-[14px] font-orbitron font-bold text-cyan-400 tracking-wider">
             <span className="drop-shadow-[0_0_10px_rgba(0,240,255,0.3)] hover:text-white transition-colors">CH. JAI RAM (NRIIT)</span>
             <span className="drop-shadow-[0_0_10px_rgba(0,240,255,0.3)] hover:text-white transition-colors">M. VISHNU VARDHAN (VIT-AP)</span>
             <span className="drop-shadow-[0_0_10px_rgba(0,240,255,0.3)] hover:text-white transition-colors">J. V. SIVA BALAJI (KLU)</span>
             <span className="drop-shadow-[0_0_10px_rgba(0,240,255,0.3)] hover:text-white transition-colors">A. ASHISH (KLU)</span>
          </div>
        </div>
      </div>

      <div className="absolute top-16 right-16 text-[9px] font-mono text-cyan-500/40 text-right uppercase tracking-[0.2em] pointer-events-none">
        Array_Coord: 09-Alpha-7<br/>
        Network_Priority: High<br/>
        Visual_Ref: Orbital_Blue_Marble
      </div>
    </div>
  );
};

export default IntroGlobe;
