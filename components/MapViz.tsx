
import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { feature } from 'topojson-client';
import { BoundingBox } from '../types';
import { MousePointer2, Move, Maximize2 } from 'lucide-react';

interface MapVizProps {
  lat: number;
  lon: number;
  bbox: BoundingBox | null;
  onLocationSelect: (lat: number, lon: number, bbox: BoundingBox | null) => void;
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  name: string;
}

const MapViz: React.FC<MapVizProps> = ({ lat, lon, bbox, onLocationSelect }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [mode, setMode] = useState<'select' | 'pan'>('select');
  const [startPoint, setStartPoint] = useState<[number, number] | null>(null);
  const [currentPoint, setCurrentPoint] = useState<[number, number] | null>(null);
  const [hoverCoords, setHoverCoords] = useState<[number, number] | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState>({ visible: false, x: 0, y: 0, name: '' });

  // D3 persistent refs
  const projectionRef = useRef(d3.geoEquirectangular());
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const gRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null);

  // Function to center map on coordinates
  const centerOnCoords = (targetLat: number, targetLon: number, scale?: number) => {
    if (!svgRef.current || !zoomRef.current || !projectionRef.current) return;
    
    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    
    const [x, y] = projectionRef.current([targetLon, targetLat]) || [width / 2, height / 2];
    const currentScale = scale || d3.zoomTransform(svgRef.current).k;
    
    svg.transition()
      .duration(750)
      .call(
        zoomRef.current.transform,
        d3.zoomIdentity
          .translate(width / 2, height / 2)
          .scale(currentScale)
          .translate(-x, -y)
      );
  };

  useEffect(() => {
    if (!svgRef.current) return;

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const svg = d3.select(svgRef.current);
    
    svg.selectAll("*").remove();

    // Initial Projection Setup
    const projection = d3.geoEquirectangular()
      .scale(width / 6.3)
      .translate([width / 2, height / 2]);
    projectionRef.current = projection;

    const path = d3.geoPath().projection(projection);

    // Group hierarchy
    const g = svg.append('g').attr('class', 'map-root');
    gRef.current = g;
    const mainGroup = g.append('g').attr('class', 'main-map');
    const selectionGroup = g.append('g').attr('class', 'selections');

    // Zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 15])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    zoomRef.current = zoom;
    svg.call(zoom);

    // Only enable pan drag if in pan mode
    if (mode === 'select') {
      svg.on('.zoom', null); // Disable standard zoom-drag
      svg.call(zoom).on('mousedown.zoom', null); // Re-enable wheel but kill drag-to-pan
    } else {
      svg.call(zoom);
    }

    // Fetch 50m resolution for better "state/region" visibility
    fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json')
      .then(response => response.json())
      .then(topology => {
        const countries = feature(topology, topology.objects.countries);

        mainGroup.selectAll('path')
          .data((countries as any).features)
          .enter().append('path')
          .attr('d', path as any)
          .attr('class', 'country')
          .attr('fill', '#0a192f')
          .attr('stroke', '#1e3a8a')
          .attr('stroke-width', 0.5)
          .style('cursor', 'pointer')
          .on('mouseover', function(event, d: any) {
            // Update Tooltip
            const countryName = d.properties?.name || "Unidentified Territory";
            setTooltip({
              visible: true,
              x: event.clientX,
              y: event.clientY,
              name: countryName
            });

            d3.select(this).transition().duration(200)
              .attr('fill', '#0f2a4a')
              .attr('stroke', '#00f0ff')
              .attr('stroke-width', 1);
          })
          .on('mousemove', function(event) {
            setTooltip(prev => ({
              ...prev,
              x: event.clientX,
              y: event.clientY
            }));
          })
          .on('mouseout', function() {
            setTooltip(prev => ({ ...prev, visible: false }));

            d3.select(this).transition().duration(200)
              .attr('fill', '#0a192f')
              .attr('stroke', '#1e3a8a')
              .attr('stroke-width', 0.5);
          })
          .on('click', function(event, d: any) {
            // Get centroid of the country
            const centroid = d3.geoCentroid(d);
            const [cLon, cLat] = centroid;
            
            // If in selection mode, treat as point selection
            onLocationSelect(cLat, cLon, null);
            
            // Center the map on clicked country
            centerOnCoords(cLat, cLon, 4); // Zoom in slightly to country
            event.stopPropagation();
          });

        renderOverlays(selectionGroup, projection);
      });

    const handleMouseDown = (event: any) => {
      if (mode !== 'select') return;
      const [x, y] = d3.pointer(event);
      setStartPoint([x, y]);
      setCurrentPoint([x, y]);
      setIsDrawing(true);
    };

    const handleMouseMove = (event: any) => {
      const [x, y] = d3.pointer(event);
      
      // Update coordinate tracking
      const transform = d3.zoomTransform(svg.node()!);
      const [tx, ty] = transform.invert([x, y]);
      const geo = projection.invert?.([tx, ty]);
      if (geo) setHoverCoords([geo[1], geo[0]]);

      if (!isDrawing) return;
      setCurrentPoint([x, y]);
    };

    const handleMouseUp = (event: any) => {
      if (!isDrawing || !startPoint || !currentPoint) return;
      
      const transform = d3.zoomTransform(svg.node()!);
      const [startX, startY] = transform.invert([startPoint[0], startPoint[1]]);
      const [endX, endY] = transform.invert([currentPoint[0], currentPoint[1]]);
      
      const dx = Math.abs(currentPoint[0] - startPoint[0]);
      const dy = Math.abs(currentPoint[1] - startPoint[1]);
      
      const coordsStart = projection.invert?.([startX, startY]);
      const coordsEnd = projection.invert?.([endX, endY]);

      if (coordsStart && coordsEnd) {
        if (dx < 5 && dy < 5) {
          // Point click selection
          onLocationSelect(coordsEnd[1], coordsEnd[0], null);
        } else {
          // Bounding box selection
          const newBbox: BoundingBox = {
            lonMin: Math.min(coordsStart[0], coordsEnd[0]),
            lonMax: Math.max(coordsStart[0], coordsEnd[0]),
            latMin: Math.min(coordsStart[1], coordsEnd[1]),
            latMax: Math.max(coordsStart[1], coordsEnd[1]),
          };
          const centerLat = (newBbox.latMin + newBbox.latMax) / 2;
          const centerLon = (newBbox.lonMin + newBbox.lonMax) / 2;
          onLocationSelect(centerLat, centerLon, newBbox);
        }
      }

      setIsDrawing(false);
      setStartPoint(null);
      setCurrentPoint(null);
    };

    svg.on('mousedown', handleMouseDown);
    svg.on('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
    };

  }, [onLocationSelect, mode, isDrawing, startPoint, currentPoint]);

  // Effect to re-render overlays when props change
  useEffect(() => {
    if (!gRef.current || !projectionRef.current) return;
    const selectionGroup = gRef.current.select('.selections');
    renderOverlays(selectionGroup, projectionRef.current);
  }, [lat, lon, bbox]);

  const renderOverlays = (group: any, projection: any) => {
    group.selectAll("*").remove();

    if (bbox) {
      const p1 = projection([bbox.lonMin, bbox.latMax]);
      const p2 = projection([bbox.lonMax, bbox.latMin]);
      if (p1 && p2) {
        group.append('rect')
          .attr('x', Math.min(p1[0], p2[0]))
          .attr('y', Math.min(p1[1], p2[1]))
          .attr('width', Math.abs(p2[0] - p1[0]))
          .attr('height', Math.abs(p2[1] - p1[1]))
          .attr('fill', 'rgba(0, 240, 255, 0.15)')
          .attr('stroke', '#00f0ff')
          .attr('stroke-width', 1.5)
          .attr('stroke-dasharray', '4,2');
      }
    } else {
      const coords = projection([lon, lat]);
      if (coords) {
        group.append('circle')
          .attr('cx', coords[0])
          .attr('cy', coords[1])
          .attr('r', 5)
          .attr('fill', '#00f0ff')
          .attr('filter', 'drop-shadow(0 0 5px #00f0ff)');
          
        group.append('circle')
          .attr('cx', coords[0])
          .attr('cy', coords[1])
          .attr('r', 10)
          .attr('fill', 'none')
          .attr('stroke', '#00f0ff')
          .attr('stroke-width', 1)
          .append('animate')
          .attr('attributeName', 'r')
          .attr('from', 5)
          .attr('to', 15)
          .attr('dur', '1.5s')
          .attr('repeatCount', 'indefinite');
      }
    }
  };

  const resetZoom = () => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current).transition().duration(750).call(
      zoomRef.current.transform,
      d3.zoomIdentity
    );
  };

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden rounded-lg border border-cyan-900 bg-black/40 shadow-inner group">
      <svg ref={svgRef} className="w-full h-full" />
      
      {/* Tooltip */}
      {tooltip.visible && (
        <div 
          className="fixed pointer-events-none z-[100] px-3 py-1.5 glass-panel rounded border border-cyan-500/50 shadow-[0_0_10px_rgba(0,240,255,0.3)] transform -translate-x-1/2 -translate-y-[120%]"
          style={{ 
            left: tooltip.x, 
            top: tooltip.y,
            transition: 'left 0.05s linear, top 0.05s linear'
          }}
        >
          <div className="text-[10px] font-orbitron font-bold text-cyan-400 whitespace-nowrap tracking-wider uppercase">
            {tooltip.name}
          </div>
          <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-cyan-900 border-r border-b border-cyan-500/50 rotate-45"></div>
        </div>
      )}

      {/* Real-time selection box (un-transformed) */}
      {isDrawing && startPoint && currentPoint && (
        <div 
          className="absolute border-2 border-cyan-400 bg-cyan-500/10 pointer-events-none z-10"
          style={{
            left: Math.min(startPoint[0], currentPoint[0]),
            top: Math.min(startPoint[1], currentPoint[1]),
            width: Math.abs(currentPoint[0] - startPoint[0]),
            height: Math.abs(currentPoint[1] - startPoint[1]),
          }}
        />
      )}

      {/* Mode Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 z-20">
        <button 
          onClick={() => setMode('select')}
          className={`p-2 rounded border ${mode === 'select' ? 'bg-cyan-600 border-cyan-400 text-black' : 'bg-black/60 border-cyan-900 text-cyan-500'} transition-all`}
          title="Selection Mode"
        >
          <MousePointer2 size={18} />
        </button>
        <button 
          onClick={() => setMode('pan')}
          className={`p-2 rounded border ${mode === 'pan' ? 'bg-cyan-600 border-cyan-400 text-black' : 'bg-black/60 border-cyan-900 text-cyan-500'} transition-all`}
          title="Explore Mode (Pan/Zoom)"
        >
          <Move size={18} />
        </button>
        <button 
          onClick={resetZoom}
          className="p-2 rounded border bg-black/60 border-cyan-900 text-cyan-500 hover:border-cyan-400 transition-all"
          title="Reset Zoom"
        >
          <Maximize2 size={18} />
        </button>
      </div>

      {/* Info Overlays */}
      <div className="absolute top-4 left-4 font-mono text-[10px] text-cyan-500 bg-black/80 p-2 border border-cyan-900 rounded select-none pointer-events-none">
        {bbox ? (
          <>
            REGION SELECTED:<br/>
            N: {bbox.latMax.toFixed(2)}° | S: {bbox.latMin.toFixed(2)}°<br/>
            E: {bbox.lonMax.toFixed(2)}° | W: {bbox.lonMin.toFixed(2)}°
          </>
        ) : (
          <>
            TARGET POINT:<br/>
            LAT: {lat.toFixed(4)}°N <br/>
            LON: {lon.toFixed(4)}°E
          </>
        )}
        {hoverCoords && (
          <div className="mt-2 pt-2 border-t border-cyan-900 text-cyan-700">
            CURSOR: {hoverCoords[0].toFixed(2)}°, {hoverCoords[1].toFixed(2)}°
          </div>
        )}
      </div>

      <div className="absolute bottom-4 right-4 text-[9px] text-cyan-700 font-mono italic pointer-events-none">
        {mode === 'select' ? 'CLICK COUNTRY OR DRAG TO SELECT REGION' : 'DRAG TO PAN • SCROLL TO ZOOM'}
      </div>
    </div>
  );
};

export default MapViz;
