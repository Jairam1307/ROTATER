
import React, { useState, useEffect, useCallback } from 'react';
import { Search, Bell, Activity, MapPin, Loader2, RadioTower, FileSpreadsheet, Calendar, CloudRain, Thermometer, Wind, Sun, Droplets, LayoutDashboard, Database, ShieldCheck, Heart, ExternalLink } from 'lucide-react';
import MapViz from './MapViz';
import ClimateCharts from './ClimateCharts';
import ChatAssistant from './ChatAssistant';
import MajorAlerts from './MajorAlerts';
import { fetchClimateData, fetchCalamityHistory, fetchClimatologyData } from '../services/nasaService';
import { fetchRealTimeWeather } from '../services/weatherService';
import { getResearchIntelligence, resolveGeospatialQuery } from '../services/geminiService';
import { ClimateStats, Calamity, BoundingBox, WeatherData, ClimatologyStats, ResearchIntelligence } from '../types';
import { AnimatePresence, motion } from 'framer-motion';

const Dashboard: React.FC = () => {
  const currentYear = new Date().getFullYear();
  const [lat, setLat] = useState<number>(20.5937);
  const [lon, setLon] = useState<number>(78.9629);
  const [bbox, setBbox] = useState<BoundingBox | null>(null);
  const [startYear, setStartYear] = useState<number>(2018);
  const [endYear, setEndYear] = useState<number>(2025);
  
  const [data, setData] = useState<ClimateStats[]>([]);
  const [climatology, setClimatology] = useState<ClimatologyStats[]>([]);
  const [realTimeWeather, setRealTimeWeather] = useState<WeatherData | null>(null);
  const [calamities, setCalamities] = useState<Calamity[]>([]);
  const [intelligence, setIntelligence] = useState<ResearchIntelligence | null>(null);
  
  const [loading, setLoading] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('System Ready');
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'mission' | 'disasters' | 'infrastructure' | 'ngos'>('mission');

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      setStatusMessage('Syncing Orbital Arrays...');
      const [clim, weather, stats] = await Promise.all([
        fetchClimatologyData(lat, lon),
        fetchRealTimeWeather(lat, lon),
        fetchClimateData(lat, lon, startYear, endYear)
      ]);
      
      const events = fetchCalamityHistory(lat, lon);
      setData(stats);
      setClimatology(clim);
      setRealTimeWeather(weather);
      setCalamities(events);

      setStatusMessage('Consulting Neural Research Core...');
      const intel = await getResearchIntelligence(stats, clim, weather, lat, lon);
      setIntelligence(intel);
      
      setStatusMessage('Uplink Synchronized');
    } catch (error) {
      console.error(error);
      setStatusMessage('Sensor Fault Detected');
    } finally {
      setLoading(false);
    }
  }, [lat, lon, startYear, endYear]);

  useEffect(() => { loadData(); }, [lat, lon]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      const result = await resolveGeospatialQuery(searchQuery);
      setLat(result.lat); setLon(result.lon); setBbox(result.bbox);
      setSearchQuery('');
    } catch (e) { console.error(e); }
  };

  const exportCSV = () => {
    const csvRows = [
      ["Date", "Temperature", "Rainfall", "NDVI", "Anomaly"],
      ...data.map(row => [row.date, row.temperature, row.rainfall, row.ndvi, row.anomaly])
    ];
    const csvContent = "data:text/csv;charset=utf-8," + csvRows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `ROTATER_SECTOR_${lat}_${lon}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-[#02040a] text-white p-4 md:p-6 flex flex-col gap-6 font-exo">
      
      <header className="flex justify-between items-center border-b border-cyan-900/40 pb-4">
        <div className="flex items-center gap-3">
          <RadioTower className="text-cyan-400 animate-pulse" size={24} />
          <h2 className="text-3xl font-orbitron font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">ROTATER</h2>
          <div className="px-2 py-0.5 border border-cyan-500/30 rounded text-[9px] font-mono text-cyan-500 bg-cyan-950/20">{statusMessage.toUpperCase()}</div>
        </div>
        <button className="p-2 glass-panel rounded-full text-cyan-400 border border-cyan-500/20 transition-transform active:scale-90"><Bell size={18} /></button>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        <div className="lg:col-span-8 space-y-6">
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2 glass-panel p-4 rounded border border-cyan-500/10 flex items-center gap-4">
              <Calendar size={16} className="text-cyan-500" />
              <div className="flex items-center gap-2">
                <span className="text-[9px] font-orbitron text-cyan-500 uppercase">Analysis Window:</span>
                <input type="number" value={startYear} onChange={e => setStartYear(Number(e.target.value))} className="bg-black border border-cyan-900/50 rounded px-2 py-1 w-16 text-[10px] text-cyan-400 font-mono focus:border-cyan-500" />
                <span className="text-cyan-900">TO</span>
                <input type="number" value={endYear} onChange={e => setEndYear(Number(e.target.value))} className="bg-black border border-cyan-900/50 rounded px-2 py-1 w-16 text-[10px] text-cyan-400 font-mono focus:border-cyan-500" />
              </div>
              <button onClick={loadData} className="ml-auto bg-cyan-600 hover:bg-cyan-500 text-black font-bold text-[9px] font-orbitron px-4 py-2 rounded transition-all active:scale-95 shadow-lg">REFRESH CORE</button>
            </div>

            <div className="glass-panel p-4 rounded border border-cyan-500/20 flex flex-col justify-center scanline-effect">
              <div className="flex justify-between text-[8px] font-orbitron text-cyan-400 mb-2">
                <span>SECTOR TELEMETRY</span>
                <CloudRain size={12} className="animate-bounce" />
              </div>
              {realTimeWeather ? (
                <div className="flex justify-between items-center">
                  <div className="text-xl font-bold font-mono text-cyan-400">{realTimeWeather.temp}°C</div>
                  <div className="text-[9px] font-mono opacity-60">HUM: {realTimeWeather.humidity}% | WIND: {realTimeWeather.windSpeed}kph</div>
                </div>
              ) : <div className="text-[10px] opacity-20">PENDING...</div>}
            </div>
          </div>

          <div className="h-[400px] w-full relative group">
             <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 w-full max-w-md px-4">
                <div className="glass-panel p-1 rounded-full border border-cyan-500/30 flex items-center bg-black/60 shadow-2xl transition-all focus-within:border-cyan-400 focus-within:scale-105">
                  <Search size={14} className="text-cyan-500 ml-4 mr-2" />
                  <input type="text" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSearch()} placeholder="RESEARCH SECTOR (e.g. Florida, Coastal India)..." className="bg-transparent border-none outline-none text-[10px] text-white w-full font-mono py-2" />
                </div>
             </div>
             <MapViz lat={lat} lon={lon} bbox={bbox} onLocationSelect={(la, lo, bb) => { setLat(la); setLon(lo); setBbox(bb); }} />
          </div>

          <ClimateCharts data={data} calamities={calamities} />
        </div>

        <div className="lg:col-span-4 flex flex-col gap-4">
          <div className="glass-panel flex-1 rounded border border-cyan-500/10 flex flex-col overflow-hidden scanline-effect">
            <div className="flex border-b border-white/5 bg-black/20">
              {[
                { id: 'mission', icon: LayoutDashboard, label: 'MISSION' },
                { id: 'disasters', icon: Database, label: 'HISTORY' },
                { id: 'infrastructure', icon: ShieldCheck, label: 'RELIEF' },
                { id: 'ngos', icon: Heart, label: 'NGO' }
              ].map(tab => (
                <button 
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex-1 py-3 flex flex-col items-center gap-1 transition-all relative ${activeTab === tab.id ? 'text-cyan-400 bg-cyan-500/5' : 'text-gray-600 hover:text-gray-400 hover:bg-white/5'}`}
                >
                  <tab.icon size={14} className={activeTab === tab.id ? 'animate-pulse' : ''} />
                  <span className="text-[8px] font-orbitron font-bold tracking-tighter">{tab.label}</span>
                  {activeTab === tab.id && <motion.div layoutId="activeTab" className="absolute bottom-0 left-0 right-0 h-0.5 bg-cyan-500 shadow-[0_0_10px_#00f0ff]" />}
                </button>
              ))}
            </div>

            <div className="p-5 flex-1 overflow-y-auto scrollbar-hide text-xs leading-relaxed">
              {intelligence ? (
                <div className="space-y-4">
                  {activeTab === 'mission' && (
                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                      <h4 className="text-[10px] font-orbitron text-cyan-400 uppercase tracking-widest flex items-center gap-2">
                        <Activity size={10} /> Sector Summary
                      </h4>
                      <p className="text-gray-400 italic text-[11px] border-l border-cyan-500/30 pl-3">"{intelligence.summary}"</p>
                      <div className="space-y-2">
                        {intelligence.predictions.map((p, i) => (
                          <div key={i} className="p-2 bg-white/5 rounded border border-white/5 flex justify-between items-center hover:bg-cyan-500/5 transition-colors group">
                            <span className="font-mono text-[9px] uppercase group-hover:text-cyan-400">{p.month}</span>
                            <span className={`text-[8px] px-1.5 py-0.5 rounded font-bold ${p.riskLevel === 'Critical' ? 'bg-red-500 text-black shadow-[0_0_10px_rgba(239,68,68,0.5)]' : 'text-cyan-400 border border-cyan-500/20'}`}>{p.riskLevel}</span>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}

                  {activeTab === 'disasters' && (
                    <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">
                       <h4 className="text-[10px] font-orbitron text-red-400 uppercase tracking-widest">Event Timeline (2018-2025)</h4>
                       {intelligence.disasters.map((d, i) => (
                         <div key={i} className="border-l-2 border-red-900/50 pl-3 py-1 space-y-1 hover:border-red-500 transition-colors">
                           <div className="flex justify-between items-center">
                             <span className="font-bold text-white text-[10px] uppercase">{d.type}</span>
                             <span className="text-[8px] font-mono text-gray-500">{d.date}</span>
                           </div>
                           <p className="text-[9px] text-gray-400 leading-tight">{d.impact}</p>
                           <div className="flex gap-1 flex-wrap">
                             {d.areas.map(a => <span key={a} className="text-[7px] bg-red-950 text-red-200 px-1 rounded border border-red-500/20">{a}</span>)}
                           </div>
                         </div>
                       ))}
                    </motion.div>
                  )}

                  {activeTab === 'infrastructure' && (
                    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="space-y-4">
                       <h4 className="text-[10px] font-orbitron text-emerald-400 uppercase tracking-widest">Strategic Relief Points</h4>
                       {intelligence.infrastructure.map((inf, i) => (
                         <div key={i} className="p-3 bg-white/5 rounded border border-white/10 flex flex-col gap-1 hover:border-emerald-500/50 transition-all hover:bg-emerald-500/5">
                            <div className="flex justify-between">
                              <span className="font-bold text-[10px] text-emerald-100">{inf.name}</span>
                              <span className="text-[8px] text-emerald-500 bg-emerald-950 px-1 rounded border border-emerald-500/30">ACTIVE</span>
                            </div>
                            <div className="text-[9px] text-gray-500 flex items-center gap-1"><MapPin size={8} className="text-emerald-500" /> {inf.location}</div>
                            <a href={inf.uri} target="_blank" rel="noreferrer" className="mt-2 text-[8px] text-cyan-500 hover:text-white flex items-center gap-1 uppercase font-bold tracking-tighter">
                              OPEN SATELLITE COORDS <ExternalLink size={8} />
                            </a>
                         </div>
                       ))}
                    </motion.div>
                  )}

                  {activeTab === 'ngos' && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
                       <h4 className="text-[10px] font-orbitron text-pink-400 uppercase tracking-widest">Active Aid Network</h4>
                       {intelligence.ngos.map((ngo, i) => (
                         <div key={i} className="p-2 border-b border-white/5 flex justify-between items-center hover:bg-pink-500/5 transition-colors">
                           <div>
                             <div className="font-bold text-[10px] text-pink-100">{ngo.name}</div>
                             <div className="text-[8px] text-gray-500 uppercase">{ngo.aidType}</div>
                           </div>
                           <div className="text-right">
                             <div className="text-[8px] text-gray-400">{ngo.origin}</div>
                             <div className="text-[7px] text-pink-400 font-mono font-bold">{ngo.mode}</div>
                           </div>
                         </div>
                       ))}
                    </motion.div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full opacity-20 gap-4">
                  <Activity size={32} className="animate-pulse text-cyan-500" />
                  <span className="font-mono text-[9px] tracking-widest">AWAITING NEURAL SYNC</span>
                </div>
              )}
            </div>

            <button onClick={exportCSV} className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 text-black text-[10px] font-orbitron font-bold flex items-center justify-center gap-2 border-t border-cyan-400/30 transition-all active:scale-95">
               <FileSpreadsheet size={14} /> EXPORT RESEARCH_LOG.CSV
            </button>
          </div>
          
          <MajorAlerts calamities={calamities} predictions={intelligence?.predictions || []} />
        </div>
      </div>
      
      <ChatAssistant lat={lat} lon={lon} predictions={intelligence?.predictions || []} calamities={calamities} />
    </div>
  );
};

export default Dashboard;
